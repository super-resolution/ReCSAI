import matplotlib.pyplot as plt
from src.models.wavelet_model import WaveletAI
import numpy as np
import tensorflow as tf
import os
from src.data import crop_generator_save_file_wavelet

def train():
    ai = WaveletAI()
    #test.load_weights("training_lvl3/cp-1000.ckpt")

    ai.compile(optimizer='adam',
                loss=tf.keras.losses.MSE,
                metrics=['accuracy'])

    #todo: use more realistic data...
    dataset = tf.data.Dataset.from_generator(crop_generator_save_file_wavelet,
                                             (tf.float32, tf.float32),
                                             output_shapes=((1 * 100, 100, 100, 3), (1 * 100, 100, 100, 3)))
    #todo: use tfwavelet dataset
    iterator = iter(dataset)
    for j in range(40):
        #todo: iterate over dataset and train...
        # dataset = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32, tf.float64))
        image = np.zeros((100, 128, 128,3))
        truth = np.zeros((100, 128, 128,3))

        image[:, 14:114, 14:114], truth[:, 14:114, 14:114] = iterator.__next__()


        image_tf1 = tf.convert_to_tensor(image[0:90, :, :, :])
        image_tf2 = tf.convert_to_tensor(image[90:100, :, :, :])
        truth_tf1 = tf.convert_to_tensor(truth[0:90, :, :, :])
        truth_tf2 = tf.convert_to_tensor(truth[90:100, :, :, :])

        print("data loaded")
        #out_o = tfwavelets.nodes.dwt2d(image_tf1[0], tfwavelets.dwtcoeffs.haar)
        checkpoint_path = "training_lvl5/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=50)

        history = ai.fit(image_tf1[:,:,:,1:2], truth_tf1[:,:,:,1:2], epochs=1000, validation_data=(image_tf2[:,:,:,1:2], truth_tf2[:,:,:,1:2]), callbacks=[cp_callback])




    fig, axs = plt.subplots(2,5)
    #
    # coeffs2 = pywt.dwt2(image[0], 'haar')
    # LL, (LH, HL, HH) = coeffs2
    #
    # layer = DWT2()
    # layer2 = IDWT2(128) todo: include test for wavelet transform
    # print(image_tf1[0:1])

    # out = layer(image_tf1[1:2,:,:,0:1])
    # out = layer2(out)
    #axs[0,0].imshow(out[0,:,:,0])
    #axs[1,0].imshow(image_tf1[1,:,:,0])
    #plt.show()
    print(tf.reduce_sum(ai.trainable_weights[1][0]*ai.trainable_weights[1][1]))


    i = ai.predict(image_tf2[1:2,:,:,1:2])

    print(tf.reduce_sum(ai.trainable_weights[1][0]*ai.trainable_weights[1][1]))

    fig, axs = plt.subplots(3)

    axs[0].imshow(image_tf2[1,:,:,1])
    axs[1].imshow(i[0,:,:,0])
    axs[2].imshow(truth_tf2[1,:,:,1])
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

if __name__ == '__main__':
    train()