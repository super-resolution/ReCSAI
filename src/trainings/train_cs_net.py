from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet,CompressedSensingInceptionNet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #todo: plot in main?
from src.data import crop_generator,CROP_TRANSFORMS, crop_generator_u_net
from src.custom_metrics import JaccardIndex
import copy

class TrainBase():
    def __init__(self):
        self.network = None
        self.optimizer = None
        self.metrics = None
        self.manager = None
        self.ckpt = None
        self.train_loops = 60


    @tf.function
    def train_step(self, train_image, truth):
        with tf.GradientTape() as tape:
            logits = self.network(train_image)
            loss = self.network.compute_loss(truth, logits)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

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
            #try:
            self.metrics.update_state(truth.numpy(), pred)
            accuracy = self.metrics.result(int(self.ckpt.step))
            print("jaccard index {:1.2f}".format(accuracy[0])+ " rmse {:1.2f}".format(accuracy[1]))
            self.metrics.reset()
            self.metrics.save()
            # except:
            #     print("no metric...")

    def train(self):
        for j in range(self.train_loops):
            sigma = 150#np.random.randint(100, 250)
            generator = crop_generator_u_net(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
            self.network.update(sigma, 100)
            self.loop(dataset)
        sigma = np.random.randint(150, 200)
        generator = crop_generator_u_net(9, sigma_x=sigma)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
        self.network.update(sigma, 100)
        self.test(dataset)

    def test(self, dataset):
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

class TrainInceptionNet(TrainBase):
    def __init__(self):
        super(TrainInceptionNet, self).__init__()
        self.network = CompressedSensingInceptionNet()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.metrics = JaccardIndex("./cs_training_inception_increased_depth")
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer , net=self.network)
        self.manager = tf.train.CheckpointManager(self.ckpt, './cs_training_inception_increased_depth', max_to_keep=6)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

class TrainCVNet(TrainBase):
    def __init__(self):
        super(TrainCVNet, self).__init__()
        self.network = CompressedSensingCVNet()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.metrics = JaccardIndex("./cs_training_u_nmask_loss")
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer , net=self.network)
        self.manager = tf.train.CheckpointManager(self.ckpt, './cs_training_u_nmask_loss', max_to_keep=6)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


def train_cs_inception_net():
    #test dataset from generator
    metrics = JaccardIndex("./cs_training_inception_sigmoid")
    @tf.function
    def train_step(train_image, truth):
        with tf.GradientTape() as tape:
            logits = cs_net(train_image)
            loss = cs_net.compute_loss(truth, logits)
        gradients = tape.gradient(loss, cs_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cs_net.trainable_variables))
        # step.assign_add(1)

        # accuracy_value = accuracy(truth, tf.argmax(logits, -1))
        return loss  # , accuracy_value

    @tf.function
    def loop(dataset):
        for train_image, truth in dataset.take(3):
            for i in range(50):
                loss_value = train_step(train_image, truth)
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()) )

        for train_image, truth in dataset.take(1):
            pred = cs_net.predict(train_image)
            try:
                metrics.update_state(truth.numpy(), pred)
                accuracy = metrics.result(int(ckpt.step))
                print("jaccard index {:1.2f}".format(accuracy[0])+ " rmse {:1.2f}".format(accuracy[1]))
                metrics.reset()
                metrics.save()
            except:
                print("no metric...")

    def outer_loop():
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        for j in range(30):
            sigma = np.random.randint(100, 250)
            generator = crop_generator_u_net(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
            cs_net.update(sigma, 100)
            loop(dataset)
        sigma = np.random.randint(150, 200)
        generator = crop_generator_u_net(9, sigma_x=sigma)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
        cs_net.update(sigma, 100)
        test(dataset)

    def test(dataset):
        for train_image, truth in dataset.take(1):
            truth = truth.numpy()
            result = cs_net.predict(train_image)
            for i in range(truth.shape[0]):
                fig,axs = plt.subplots(4)
                axs[0].imshow(truth[i, :, :, 2])
                axs[1].imshow(result[i,:,:,2])
                axs[2].imshow(truth[i, :, :, 1])
                axs[3].imshow(result[i,:,:,1])
                plt.show()


    for i in range(10):
        cs_net = CompressedSensingInceptionNet()
        optimizer = tf.keras.optimizers.Adam(1e-4)
        step = tf.Variable(1, name="global_step")
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
        manager = tf.train.CheckpointManager(ckpt, './cs_training_inception_sigmoid', max_to_keep=6)
        outer_loop()


def train_cs_u_net():
    #test dataset from generator
    metrics = JaccardIndex("./cs_training_u")
    @tf.function
    def train_step(train_image, truth):
        with tf.GradientTape() as tape:
            logits = cs_net(train_image)
            loss = cs_net.compute_loss(truth, logits)
        gradients = tape.gradient(loss, cs_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cs_net.trainable_variables))
        # step.assign_add(1)

        # accuracy_value = accuracy(truth, tf.argmax(logits, -1))
        return loss  # , accuracy_value

    # @tf.function
    def loop(dataset):
        for train_image, truth in dataset.take(3):
            for i in range(50):
                loss_value = train_step(train_image, truth)
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()) )

        for train_image, truth in dataset.take(1):
            pred = cs_net.predict(train_image)
            try:
                metrics.update_state(truth.numpy(), pred)
                accuracy = metrics.result(int(ckpt.step))
                print("jaccard index {:1.2f}".format(accuracy[0])+ " rmse {:1.2f}".format(accuracy[1]))
                metrics.reset()
                metrics.save()
            except:
                print("no metric...")

    def outer_loop():
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        for j in range(30):
            sigma = np.random.randint(100, 250)
            generator = crop_generator_u_net(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
            cs_net.update(sigma, 100)
            loop(dataset)
        sigma = np.random.randint(150, 200)
        generator = crop_generator_u_net(9, sigma_x=sigma)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((1*100, 9, 9, 3),  (1*100, 9,9,4)))
        cs_net.update(sigma, 100)
        test(dataset)

    def test(dataset):
        for train_image, truth in dataset.take(1):
            truth = truth.numpy()
            result = cs_net.predict(train_image)
            for i in range(truth.shape[0]):
                fig,axs = plt.subplots(2)
                axs[0].imshow(truth[i, :, :, 2])
                axs[1].imshow(result[i,:,:,2])
                plt.show()


    for i in range(10):
        cs_net = CompressedSensingCVNet()

        optimizer = tf.keras.optimizers.Adam(1e-4)
        step = tf.Variable(1, name="global_step")
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
        manager = tf.train.CheckpointManager(ckpt, './cs_training_u', max_to_keep=6)

        outer_loop()

def train_cs_net():
    #test dataset from generator
    @tf.function
    def train_step(train_image, truth):
        with tf.GradientTape() as tape:
            truth_p = truth[:, 0:6] / 100
            logits = cs_net(train_image)
            logits_p = logits[:, 0:6] / 8
            #loss = cs_net.compute_loss(truth_p, logits_p, truth[:, 6:], logits[:, 6:], )#todo:change to permute loss
            loss = cs_net.compute_permute_loss(tf.concat([truth_p, truth[:,6:]],axis=-1), tf.concat([logits_p, logits[:,6:]],axis=-1) )
        gradients = tape.gradient(loss, cs_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cs_net.trainable_variables))
        # step.assign_add(1)

        # accuracy_value = accuracy(truth, tf.argmax(logits, -1))
        return loss  # , accuracy_value

    # @tf.function
    def loop(dataset):
        for train_image, truth in dataset.take(2):
            for i in range(50):
                loss_value = train_step(train_image, truth)
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss_value.numpy()))

    def outer_loop():
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        for j in range(5):
            sigma = np.random.randint(150, 200)
            generator = crop_generator(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((CROP_TRANSFORMS*100, 9, 9, 3),  (CROP_TRANSFORMS*100, 9)))
            cs_net.update(sigma, 100)
            loop(dataset)
        sigma = np.random.randint(150, 200)
        generator = crop_generator(9, sigma_x=sigma)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((CROP_TRANSFORMS*100, 9, 9, 3),  (CROP_TRANSFORMS*100, 9)))
        cs_net.update(sigma, 100)
        test(dataset)

    def test(dataset):
        for train_image, truth in dataset.take(1):
            truth = truth.numpy() / 100
            result = cs_net.predict(train_image) / 8
            for i in range(truth.shape[0]):
                plt.imshow(train_image[i, :, :, 1])
                for n in range(3):
                    plt.scatter(truth[i, 2 * n + 1], truth[i, 2 * n], c="r")
                    if result[i, 6 + n] > 0.1:
                        plt.scatter(result[i, 2 * n + 1], result[i, 2 * n], c="g")
                plt.show()


    for i in range(10):
        cs_net = CompressedSensingNet()

        optimizer = tf.keras.optimizers.Adam()
        step = tf.Variable(1, name="global_step")
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
        manager = tf.train.CheckpointManager(ckpt, './cs_training_permute_loss_downsample', max_to_keep=6)

        outer_loop()

if __name__ == '__main__':
    training = TrainInceptionNet()
    training.train()