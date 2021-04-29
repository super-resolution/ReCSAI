from src.models.wavelet_model import WaveletAI
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #todo: plot in main?
from src.data import crop_generator,CROP_TRANSFORMS, crop_generator_u_net, generate_generator
from src.custom_metrics import JaccardIndex
from src.utility import bin_localisations_v2, get_coords

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
        for i in range(result_tensor.shape[0]):

            classifier = result_tensor[i, :, :, 2]
            if np.sum(classifier) > self.threshold:
                classifier[np.where(classifier < 0.1)] = 0
                indices = get_coords(classifier).T
                x = result_tensor[i, indices[0], indices[1], 0]
                y = result_tensor[i, indices[0], indices[1], 1]
                dx = result_tensor[i, indices[0], indices[1], 3]#todo: if present
                dy = result_tensor[i, indices[0], indices[1], 4]

                for j in range(indices[0].shape[0]):
                    result_array.append(np.array([coord_list[i][0] +float(indices[0][j]) + x[j]
                        ,coord_list[i][1] +float(indices[1][j]) + y[j], coord_list[i][2], dx[j], dy[j]]))
        return np.array(result_array)

    def predict(self, path, drift_path=None):
        if drift_path:
            drift = pd.read_csv(drift_path).as_matrix()#r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv"
        #todo: implement drift
        result_full = []
        gen = generate_generator(path)
        dataset = tf.data.Dataset.from_generator(gen, tf.float64)
        j = 0
        for image in dataset:
            # plt.imshow(image[2,:,:,0])
            # plt.show()
            # plt.imshow(image[2,:,:,1])
            # plt.show()
            crop_tensor, _, coord_list = bin_localisations_v2(image, self.denoising, th=0.2)
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
    def loop(self, dataset):
        for train_image, truth in dataset.take(3):
            for i in range(50):
                loss_value = self.train_step(train_image, truth)
                self.ckpt.step.assign_add(1)
                if int(self.ckpt.step) % 10 == 0:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()) )

        for train_image, truth in dataset.take(1):
            pred = self.network.predict(train_image)
            self.metrics.update_state(truth.numpy(), pred)
            accuracy = self.metrics.result(int(self.ckpt.step))
            print("jaccard index {:1.2f}".format(accuracy[0])+ " rmse {:1.2f}".format(accuracy[1]))
            self.metrics.reset()
            self.metrics.save()

    def train(self):
        for j in range(self.train_loops):
            sigma = np.random.randint(100, 250)
            generator = crop_generator_u_net(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
            self.sigma = sigma
            self.loop(dataset)
        self.test()

    @tf.function
    def train_step(self, train_image, truth):
        with tf.GradientTape() as tape:
            logits = self.network(train_image)
            loss = self.network.compute_loss(truth, logits)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

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
                fig,axs = plt.subplots(3,2)
                axs[0][0].imshow(truth[i, :, :, 2])
                axs[0][1].imshow(result[i,:,:,2])
                axs[1][0].imshow(truth[i, :, :, 1])
                axs[1][1].imshow(result[i,:,:,1])
                axs[2][0].imshow(truth[i, :, :, 0])
                axs[2][1].imshow(result[i,:,:,0])
                axs[0][0].set_title("Ground truth")
                axs[0][1].set_title("Prediction")
                axs[0][0].set_ylabel("Classifier")
                axs[1][0].set_ylabel("Delta x")
                axs[2][0].set_ylabel("Delta y")
                plt.show()