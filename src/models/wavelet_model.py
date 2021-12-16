from src.custom_layers.wavelet_layers import FullWavelet
import tensorflow as tf

def WaveletAI(test=False, shape=128):
    inputs = tf.keras.layers.Input(shape=[shape, shape, 1])#todo: use variable shape
    layer = FullWavelet(shape,level=4,)
    final = tf.keras.layers.ReLU()
    x = inputs/tf.keras.backend.max(inputs)
    x = layer(x, test=test)
    x = final(x)
    return tf.keras.Model(inputs=inputs, outputs=x)