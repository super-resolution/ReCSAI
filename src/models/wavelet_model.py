from src.custom_layers.wavelet_layers import FullWavelet
import tensorflow as tf

def WaveletAI():
    inputs = tf.keras.layers.Input(shape=[128, 128, 1])#todo: input 3 output 1
    layer = FullWavelet(128,level=4,)
    final = tf.keras.layers.ReLU()
    x = inputs/tf.keras.backend.max(inputs)
    x = layer(x)
    x = final(x)
    return tf.keras.Model(inputs=inputs, outputs=x)