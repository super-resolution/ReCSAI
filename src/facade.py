from src.models.wavelet_model import WaveletAI
import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt #todo: plot in main?
from src.data import crop_generator_u_net, DataServing
from src.custom_metrics import JaccardIndex
from src.utility import bin_localisations_v2, get_reconstruct_coords,FRC_loss, timing
from src.models.loss_functions import compute_loss_decode, compute_loss_decode_ncs
#from scipy.ndimage.filters import gaussian_filter, uniform_filter
from src.emitters import Emitter


class NetworkFacade():
    IMAGE_ENCODING = 0
    COORDINATE_ENCODING = 1
    def __init__(self, network_class, path, denoising_chkpt, shape=128):
        self.network = network_class()
        self.denoising = WaveletAI(shape=shape)
        self._training_path = path
        self.dataset_path = r"/dataset_low_ph"
        #self.dataset_path = r"/dataset_low_ph"
        self.denoising.load_weights(denoising_chkpt)
        self.learning_rate = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)#todo for transerlearning
        self.metrics = JaccardIndex(self._training_path)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer , net=self.network)
        self.manager = tf.train.CheckpointManager(self.ckpt, self._training_path, max_to_keep=6)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        self.train_loops = 60
        self.threshold = 0.3

    @property
    def sigma(self):
        return self.network.sigma

    @sigma.setter
    def sigma(self, value):
        self.network.sigma = value


    @tf.function
    def train_step_d(self, train_image, noiseless_gt, coords):
        with tf.GradientTape() as tape:
            #logits, update_list = self.network(train_image, training=True)
            logits = self.network(train_image, training=True)
            #mat = self.network.mat
            loss = compute_loss_decode(coords, logits, noiseless_gt,  train_image)#todo, undo
            #loss = self.network.compute_loss(logits, update_list, coords, noiseless_gt,  train_image)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

    def loop_d(self, dataset, save=True):
        """
        Training and validation
        :param dataset: Dataset to train on
        :param save: Save training?
        :return:
        """
        for i,(train_image,noiseless_gt, coords,t, sigma) in enumerate(dataset):
            if i%4 !=0:
                # todo: time it
                self.sigma = sigma.numpy()
                loss_value = self.train_step_d(train_image, noiseless_gt, coords)
                self.ckpt.step.assign_add(1)
                if int(self.ckpt.step) % 10 == 0 and save:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()) )
        for i,(train_image,noiseless_gt, coords,t, sigma) in enumerate(dataset):
            if i%4 ==0:
                #train_image,noiseless_gt, coords,t = iterator.get_next()
                pred = self.network(train_image, training=False)
                vloss = compute_loss_decode_ncs(coords, pred)
                print(f"validation loss = {vloss}" )
                self.metrics.update_state(coords.numpy(), pred.numpy(), validation=vloss)#todo: save accuracy with metrics
        accuracy = self.metrics.result(int(self.ckpt.step))
        print("jaccard index {:1.2f}".format(accuracy[0])+ " rmse {:1.2f}".format(accuracy[1]) +
              " fp {:1.2f}".format(accuracy[2]) + " fn {:1.2f}".format(accuracy[3]))
        self.metrics.reset()
        self.metrics.save()


    def get_current_dataset(self):
        self.data_factory = DataServing(self.dataset_path)
        dataset = tf.data.Dataset.from_generator(self.data_factory(), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=self.data_factory.shape)
        return dataset

    def train_saved_data(self):
        """
        Train network on defined dataset
        """
        self.data_factory = DataServing(self.dataset_path)
        dataset = tf.data.Dataset.from_generator(self.data_factory(), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=self.data_factory.shape)
        for j in range(self.train_loops):
            print(self.ckpt.step//50)
            self.loop_d(dataset)
        with open(self._training_path +"/parameters.txt", "w") as text_file:
            text_file.write("dataset: " + self.dataset_path+ "\n")
            text_file.write("optimizer: " + self.optimizer.__repr__() + "\n")
            text_file.write(f"learning rate: {self.learning_rate}" )

    def pretrain_current_sigma_d(self):
        #deprecated
        sigma = self.sigma
        generator = crop_generator_u_net(9, sigma_x=sigma, noiseless_ground_truth=True)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=((1 * 100, 9, 9, 3),(1*100, 9, 9, 3), (1 * 100, 10, 3),(1*100, 9, 9, 4) ))
        iterator = iter(dataset)

        self.loop_d(iterator)

    #@timing
    def predict(self, image, raw=False):
        """
        Predict localisations of image stack
        :param image: path to image or image as np array
        :param raw: should data be returned as feature map or emitter set?
        :return:
        """
        self.data_factory = DataServing(self.dataset_path)
        coords = []
        result_full = []
        gen,offset = self.data_factory.generate_image_serving_generator(image)
        dataset = tf.data.Dataset.from_generator(gen, tf.float64)
        j = 0
        for image in dataset:
            crop_tensor, _, coord_list = bin_localisations_v2(image, self.denoising, th=self.wavelet_thresh)
            coord_list[:, 0:2] -= offset
            for z in range(len(coord_list)):
                coord_list[z][2] += j * 2000
            result_tensor = self.network.predict(crop_tensor)
            result_full.append(result_tensor)
            coords.append(np.array(coord_list))

            j+=1
            print(j)
        result_full = np.concatenate(result_full,axis=0)
        coords = np.concatenate(coords,axis=0)
        if raw:
            return result_full, coords
        elif self.network.TYPE == self.IMAGE_ENCODING:
            return Emitter.from_result_tensor(result_full, self.threshold, coords)
        elif self.network.TYPE == self.COORDINATE_ENCODING:
            pass
        else:
            raise EnvironmentError("Unknown Network output")

    def compute_frc(self, image1, image2):
        FRC_loss(image1, image2)


class NetworkFacade_C():
    IMAGE_ENCODING = 0
    COORDINATE_ENCODING = 1
    def __init__(self, network_class, path, denoising_chkpt, shape=128):
        self.network = network_class()
        self.denoising = WaveletAI(shape=shape)
        self.data_factory = DataGeneratorFactory(r"/dataset_correct_params_large")

        self.denoising.load_weights(denoising_chkpt)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)#todo for transerlearning
        self.metrics = JaccardIndex(path)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer , net=self.network)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=6)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        self.train_loops = 60
        self.threshold = 0.3
        self.wavelet_thresh = 0.1
        self.sigma_thresh = 0.4
        self.photon_filter= 0.0
        #todo: threshold for jaccard index

    @property
    def sigma(self):
        return self.network.sigma

    @sigma.setter
    def sigma(self, value):
        self.network.sigma = value


    def train_saved_data(self):
        # dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_EX, (tf.float32, tf.float32, tf.float32),
        #                                         output_shapes=((1 * 100, 9, 9, 3), (1 * 100, 9, 9, 4),(1 * 100, 9, 9, 3)))

        dataset = tf.data.Dataset.from_generator(self.data_factory(), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=self.data_factory.shape)
        iterator = iter(dataset)
        for j in range(self.train_loops):
            print(self.ckpt.step//50)

            #sigma = np.random.randint(100, 250)+np.random.rand(1)*20-10

            self.loop_d(dataset)
        self.test_d()


    def get_localizations_from_image_tensor(self, result_tensor, coord_list):
        drift_path = None#r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv"
        if drift_path:
            drift = pd.read_csv(drift_path).as_matrix()
        result_array = []
        test=[]
        for i in range(result_tensor.shape[0]):
            # current_drift = drift[int(coord_list[i][2] * 0.4), 1:3]
            # current_drift[1] *= -1

            classifier =result_tensor[i, :, :, 2]

            #classifier =uniform_filter(result_tensor[i, :, :, 2], size=3)
            #plt.imshow(classifier)
            # plt.show()
            #classifier = result_tensor[i, :, :, 2]
            if np.sum(classifier) > self.threshold:
                #classifier[np.where(classifier < self.threshold)] = 0
                #indices = np.where(np.logical_and(classifier>self.threshold,classifier<1))

                #indices = get_coords(classifier,neighbors=3).T
                indices = get_reconstruct_coords(classifier, self.threshold)
                #todo: cross filter on coords and second threshold
                x = result_tensor[i, indices[0], indices[1], 0]
                y = result_tensor[i, indices[0], indices[1], 1]
                p = result_tensor[i, indices[0], indices[1], 2]
                dx = result_tensor[i, indices[0], indices[1], 3]#todo: if present
                dy = result_tensor[i, indices[0], indices[1], 4]
                N = result_tensor[i, indices[0], indices[1], 5]

                for j in range(indices[0].shape[0]):
                    if np.sum(result_tensor[i, indices[0][j]-1:indices[0][j]+1, indices[0][j]-1:indices[0][j]+1, 2])>0.3:
                        if dx[j]<self.sigma_thresh and dy[j]<self.sigma_thresh and N[j]>self.photon_filter:
                            result_array.append(np.array([coord_list[i][0] +float(indices[0][j]) + (x[j])
                                ,coord_list[i][1] +float(indices[1][j]) + y[j], coord_list[i][2], dx[j], dy[j], N[j], p[j]]))
                            test.append(np.array([x[j],y[j]]))
        print(np.mean(np.array(test),axis=0))
        return np.array(result_array)

    def get_localizations_from_coordinate_tensor(self, result_tensor, coord_list):
        result_array = []
        for i in range(result_tensor.shape[0]):
            #current_drift = drift[int(coord_list[i][2] * 0.4), 1:3]
            # current_drift[1] *= -1
            #                if coord_list[i][2] == frame:
            # limit = [4,1,0.3]
            limit = [0.8, 0.6, 0.5]
            for n in range(3):
                if result_tensor[i, 6 + n] > limit[n]:
                    result_array.append(np.array([
                        coord_list[i][0] + result_tensor[i, 2 * n] / 8, coord_list[i][1] + result_tensor[i, 2 * n + 1] / 8,coord_list[i][2]]))
        return np.array(result_array)

    def predict(self, image, raw=False):
        coords = []
        #todo: implement drift
        result_full = []
        gen,offset = generate_generator(image)
        dataset = tf.data.Dataset.from_generator(gen, tf.float64)
        j = 0
        for image in dataset.take(4):
            # plt.imshow(image[2,:,:,0])
            # plt.show()
            # plt.imshow(image[2,:,:,1])
            # plt.show()
            crop_tensor, _, coord_list = bin_localisations_v2(image, self.denoising, th=self.wavelet_thresh)
            coord_list[:, 0:2] -= offset
            for z in range(len(coord_list)):
                coord_list[z][2] += j * 2000
            result_tensor = self.network.predict(crop_tensor)#todo: save raw data in tmp folder
            if raw:
                result_full.append(result_tensor)
                coords.append(np.array(coord_list))
            elif self.network.TYPE == self.IMAGE_ENCODING:
                result = self.get_localizations_from_image_tensor(result_tensor, coord_list)
                result_full.append(result)
            elif self.network.TYPE == self.COORDINATE_ENCODING:
                result = self.get_localizations_from_coordinate_tensor(result_tensor, coord_list)
                result_full.append(result)
            else:
                raise EnvironmentError("Unknown Network output")
            j+=1
            print(j)
        result_full = np.concatenate(result_full,axis=0)
        coords = np.concatenate(coords,axis=0)
        if raw:
            return result_full, coords
        return result_full

    #@tf.function
    def loop(self, iterator, save=True):
        for j in range(3):
            train_image,truth, noiseless_gt = iterator.get_next()
            x = truth.numpy()
            print(tf.reduce_min(truth),tf.reduce_max(truth))

            for i in range(200):
                loss_value = self.train_step(train_image, truth, noiseless_gt)
                self.ckpt.step.assign_add(1)
                if int(self.ckpt.step) % 10 == 0 and save:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()) )

        train_image,truth, noiseless_gt = iterator.get_next()
        pred = self.network.predict(train_image)
        vloss = self.network.compute_loss_log(truth.numpy(), pred)
        print(f"validation loss = {vloss}" )
        self.metrics.update_state(truth.numpy(), pred.numpy())
        accuracy = self.metrics.result(int(self.ckpt.step))
        print("jaccard index {:1.2f}".format(accuracy[0])+ " rmse {:1.2f}".format(accuracy[1]) +
              " fp {:1.2f}".format(accuracy[2]) + " fn {:1.2f}".format(accuracy[3]))
        self.metrics.reset()
        self.metrics.save()

    def pretrain_current_sigma(self):
        sigma = self.sigma
        generator = crop_generator_u_net(9, sigma_x=sigma, noiseless_ground_truth=True)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32,tf.float32 ,tf.float32),
                                                 output_shapes=((1 * 100, 9, 9, 3), (1 * 100, 9, 9, 4), (1 * 100, 9, 9, 3)))
        iterator = iter(dataset)

        self.loop(iterator, save=False)

    def train(self):
        for j in range(self.train_loops):
            print(self.ckpt.step//50)
            sigma = np.random.randint(100, 250)
            generator = crop_generator_u_net(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
            #self.sigma = sigma#todo: reactivate!!!!!

            self.loop(dataset)
        self.test()



    @tf.function
    def train_step(self, train_image,truth, noiseless_gt):
        with tf.GradientTape() as tape:
            logits, cs_out = self.network(train_image, training=True)
            loss = self.network.compute_loss_log_cs_out(truth, logits, cs_out, truth)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

    def test(self):
        sigma = np.random.randint(150, 200)
        generator = crop_generator_u_net(9, sigma_x=sigma)

        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((1*1000, 9, 9, 3),  (1*1000, 9,9,4)))

        self.sigma = sigma
        for train_image, truth in dataset.take(1):
            truth = truth.numpy()
            result = self.network.predict(train_image)
            for i in range(truth.shape[0]):
                fig,axs = plt.subplots(3,3)
                axs[0][0].imshow(truth[i, :, :, 2])
                axs[0][1].imshow(result[i,:,:,2])
                axs[0][2].imshow(result[i,:,:,5])


                axs[1][0].imshow(truth[i, :, :, 1])
                axs[1][1].imshow(result[i,:,:,1])
                axs[1][2].imshow(result[i,:,:,3])

                axs[2][0].imshow(truth[i, :, :, 0])
                axs[2][1].imshow(result[i,:,:,0])
                axs[2][2].imshow(result[i,:,:,4])

                axs[0][0].set_title("Ground truth")
                axs[0][1].set_title("Prediction")
                axs[0][0].set_ylabel("Classifier")
                axs[1][0].set_ylabel("Delta x")
                axs[2][0].set_ylabel("Delta y")
                plt.show()