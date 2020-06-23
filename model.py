import tensorflow as tf
from custom_layers import *



def model():
    inputs = tf.keras.layers.Input(shape=[64, 64, 1])
    layer = FullWavelet()
    x = inputs
    x = layer(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
