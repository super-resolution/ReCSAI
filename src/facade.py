from src.models.wavelet_model import WaveletAI
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #todo: plot in main?
from src.data import crop_generator,CROP_TRANSFORMS, crop_generator_u_net, generate_generator, crop_generator_saved_file_EX, crop_generator_saved_file_coords
from src.custom_metrics import JaccardIndex
from src.utility import bin_localisations_v2, get_coords, get_root_path
from scipy.ndimage.filters import gaussian_filter, uniform_filter


class NetworkFacade():
    IMAGE_ENCODING = 0
    COORDINATE_ENCODING = 1
    def __init__(self, network_class, path, denoising_chkpt):
        self.network = network_class()
        self.denoising = WaveletAI()

        self.denoising.load_weights(denoising_chkpt)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
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
        #todo: threshold for jaccard index

    @property
    def sigma(self):
        return self.network.sigma

    @sigma.setter
    def sigma(self, value):
        self.network.sigma = value

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

    def get_localizations_from_image_tensor(self, result_tensor, coord_list):
        result_array = []
        test=[]
        for i in range(result_tensor.shape[0]):

            classifier =uniform_filter(result_tensor[i, :, :, 2], size=3)*9
            #classifier = result_tensor[i, :, :, 2]
            if np.sum(classifier) > self.threshold:
                classifier[np.where(classifier < 0.3)] = 0
                #indices = np.where(classifier>self.threshold)

                indices = get_coords(classifier).T
                x = result_tensor[i, indices[0], indices[1], 0]
                y = result_tensor[i, indices[0], indices[1], 1]
                dx = result_tensor[i, indices[0], indices[1], 3]#todo: if present
                dy = result_tensor[i, indices[0], indices[1], 4]

                for j in range(indices[0].shape[0]):
                    if dx[j]<self.sigma_thresh and dy[j]<self.sigma_thresh:
                        result_array.append(np.array([coord_list[i][0] +float(indices[0][j]) + (x[j])
                            ,coord_list[i][1] +float(indices[1][j]) + y[j], coord_list[i][2], dx[j], dy[j]]))
                        test.append(np.array([x[j],y[j]]))
        print(np.mean(np.array(test),axis=0))
        return np.array(result_array)

    def predict(self, image, drift_path=None):
        if drift_path:
            drift = pd.read_csv(drift_path).as_matrix()#r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv"
        #todo: implement drift
        result_full = []
        gen = generate_generator(image)
        dataset = tf.data.Dataset.from_generator(gen, tf.float64)
        j = 0
        for image in dataset:
            # plt.imshow(image[2,:,:,0])
            # plt.show()
            # plt.imshow(image[2,:,:,1])
            # plt.show()
            crop_tensor, _, coord_list = bin_localisations_v2(image, self.denoising, th=self.wavelet_thresh)
            for z in range(len(coord_list)):
                coord_list[z][2] += j * 5000
            print(crop_tensor.shape[0])
            result_tensor = self.network.predict(crop_tensor)
            if self.network.TYPE == self.IMAGE_ENCODING:
                result = self.get_localizations_from_image_tensor(result_tensor, coord_list)
            elif self.network.TYPE == self.COORDINATE_ENCODING:
                result = self.get_localizations_from_coordinate_tensor(result_tensor, coord_list)
            else:
                raise EnvironmentError("Unknown Network output")
            result_full.append(result)
        result_full = np.concatenate(result_full,axis=0)
        return result_full

    #@tf.function
    def loop(self, iterator, save=True):
        for j in range(3):
            train_image,truth, noiseless_gt = iterator.get_next()
            x = truth.numpy()
            print(tf.reduce_min(truth),tf.reduce_max(truth))

            for i in range(50):
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

    def loop_d(self, iterator, save=True):
        for j in range(3):
            train_image,noiseless_gt, coords = iterator.get_next()#todo: noiseless image here
            for i in range(50):
                loss_value = self.train_step_d(train_image, noiseless_gt, coords)
                self.ckpt.step.assign_add(1)
                if int(self.ckpt.step) % 10 == 0 and save:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()) )

        train_image,noiseless_gt, coords = iterator.get_next()
        pred,cs_out = self.network(train_image, training=True)
        vloss = self.network.compute_loss_decode(coords, pred, noiseless_gt, cs_out)
        print(f"validation loss = {vloss}" )
        self.metrics.update_state(coords.numpy(), pred.numpy())
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
        # self.sigma = sigma#todo: reactivate!!!!!
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

    def train_saved_data(self):
        sigma = np.load(get_root_path() + r"/crop_dataset_sigma.npy", allow_pickle=True).astype(np.float32)
        # dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_EX, (tf.float32, tf.float32, tf.float32),
        #                                         output_shapes=((1 * 100, 9, 9, 3), (1 * 100, 9, 9, 4),(1 * 100, 9, 9, 3)))
        dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_coords, (tf.float32, tf.float32, tf.float32),
                                                  output_shapes=((1 * 100, 9, 9, 3),(1*100, 9, 9, 3), (1 * 100, 3, 3)))
        iterator = iter(dataset)
        for j in range(self.train_loops):
            print(self.ckpt.step//50)

            #sigma = np.random.randint(100, 250)+np.random.rand(1)*20-10

            self.sigma = sigma[j//4]#todo: vary sigma in data
            self.loop_d(iterator)
        self.test_d()
    @tf.function
    def train_step_d(self, train_image,noiseless_gt, coords):
        with tf.GradientTape() as tape:
            logits, cs_out = self.network(train_image, training=True)
            loss = self.network.compute_loss_decode(coords, logits, noiseless_gt, cs_out)#todo, cs_out, noiseless_gt)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

    @tf.function
    def train_step(self, train_image,truth, noiseless_gt):
        with tf.GradientTape() as tape:
            logits, cs_out = self.network(train_image, training=True)
            loss = self.network.compute_loss_log_cs_out(truth, logits, cs_out, truth)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

    def test_d(self):
        sigma = np.random.randint(150, 200)
        x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
        X, Y = tf.meshgrid(x, x)

        dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_coords, (tf.float32, tf.float32, tf.float32),
                                                  output_shapes=((1 * 100, 9, 9, 3),(1*100, 9, 9, 3), (1 * 100, 3, 3)))

        self.sigma = sigma
        for train_image, truth_i,truth_c in dataset.take(1):
            truth = truth_i.numpy()
            result = self.network.predict(train_image)
            for i in range(truth.shape[0]):
                j=0
                fig,axs = plt.subplots(3,3)
                axs[0][0].imshow(truth[i, :, :, 2])
                axs[0][1].imshow(result[i,:,:,2])

                axs[1][0].imshow(truth[i, :, :, 1])
                axs[1][1].imshow(result[i,:,:,1])#) - (truth_c[i,j:j+1,1:2]-X))
                axs[1][2].imshow(result[i,:,:,3])

                axs[2][0].imshow(truth[i, :, :, 0])
                axs[2][1].imshow(result[i,:,:,0] )#- (truth_c[i,j:j+1,0:1]-Y))
                axs[2][2].imshow(result[i,:,:,4])

                axs[0][0].set_title("Ground truth")
                axs[0][1].set_title("Prediction")
                axs[0][0].set_ylabel("Classifier")
                axs[1][0].set_ylabel("Delta x")
                axs[2][0].set_ylabel("Delta y")
                plt.show()


    def test(self):
        sigma = np.random.randint(150, 200)
        generator = crop_generator_u_net(9, sigma_x=sigma)

        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))

        self.sigma = sigma
        for train_image, truth in dataset.take(1):
            truth = truth.numpy()
            result = self.network.predict(train_image)
            for i in range(truth.shape[0]):
                fig,axs = plt.subplots(3,3)
                axs[0][0].imshow(truth[i, :, :, 2])
                axs[0][1].imshow(result[i,:,:,2])

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