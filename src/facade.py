from src.models.wavelet_model import WaveletAI
import tensorflow as tf
import numpy as np
from src.data import DataServing
from src.custom_metrics import JaccardIndex
from src.utility import bin_localisations_v2, get_reconstruct_coords,FRC_loss, timing
from src.models.loss_functions import compute_loss_decode, compute_loss_decode_ncs
from src.emitters import Emitter


class NetworkFacade():
    IMAGE_ENCODING = 0
    COORDINATE_ENCODING = 1
    def __init__(self, network_class, path, denoising_chkpt, shape=128):
        self.network = network_class()
        self.denoising = WaveletAI(shape=shape)
        self._training_path = path
        self.dataset_path = r"/test_data_creation"
        #self.dataset_path = r"/dataset_low_ph"
        self.denoising.load_weights(denoising_chkpt)
        self.learning_rate = 1e-3
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

