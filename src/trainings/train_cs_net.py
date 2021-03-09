from src.models.cs_model import CompressedSensingNet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #todo: plot in main?



def train_cs_net(crop_generator):
    #test dataset from generator
    @tf.function
    def train_step(train_image, truth):
        with tf.GradientTape() as tape:
            truth_p = truth[:, 0:6] / 100
            logits = cs_net(train_image)
            logits_p = logits[:, 0:6] / 8
            #loss = cs_net.compute_loss(truth_p, logits_p, truth[:, 6:], logits[:, 6:], )#todo:change to permute loss
            loss = cs_net.compute_loss(truth_p,logits_p, truth[:,6:],  logits[:,6:])
        gradients = tape.gradient(loss, cs_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cs_net.trainable_variables))
        # step.assign_add(1)

        # accuracy_value = accuracy(truth, tf.argmax(logits, -1))
        return loss  # , accuracy_value

    # @tf.function
    def loop(dataset):
        for train_image, sigma, truth in dataset.take(15):
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
            sigma = np.random.randint(100, 250)
            generator = crop_generator(9, sigma_x=sigma)
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                     output_shapes=((100, 9, 9, 3),  (100, 9)))
            cs_net.update(sigma, 100)
            loop(dataset)
        sigma = np.random.randint(100, 250)
        generator = crop_generator(9, sigma_x=sigma)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                 output_shapes=((100, 9, 9, 3),  (100, 9)))
        cs_net.update(sigma, 100)
        test(dataset)

    def test(dataset):
        for train_image, truth in dataset.take(20):
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
        manager = tf.train.CheckpointManager(ckpt, './cs_training4', max_to_keep=3)

        outer_loop()