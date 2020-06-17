import tensorflow as tf
from custom_layers import *



def model():

    model = tf.keras.Sequential([
        FullWavelet(),]
        #tf.keras.layers.Conv2DTranspose(2, 64, strides=2,
        #                                padding='same',
        #                                kernel_initializer=tf.random_normal_initializer(0., 0.02),
        #                                use_bias=False),
        #tf.keras.layers.ReLU(),
        #IDWT2()]
    )
    return model
