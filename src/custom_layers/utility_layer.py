import tensorflow as tf


class Shifting(tf.keras.layers.Layer):
    def __init__(self):
        def layer(inp):
            return tfa.image.translate(inp[0],inp[1],"BILINEAR")
        self.shift_layer = tf.keras.layers.Lambda(layer)

    def restack(self, in1, in2):
        columns = []

        for i,col in enumerate(tf.unstack(in2, axis=1)):
            columns.extend([in1[:,i], col])
        columns.append(in1[:,i+1])
        return tf.stack(columns,axis=1)

    def __call__(self, input,shift):
        odd = True
        if input[0,0,0,0] ==0:
            odd = False
        substack1 = input[:,::2]
        substack2 = input[:,1::2]
        if odd:
            substack2 = self.shift_layer((substack2, shift))
        else:
            substack1 = self.shift_layer((substack1, shift))
        substack1 = tf.cast(substack1,tf.float64)
        substack2 = tf.cast(substack2,tf.float64)
        return self.restack(substack1, substack2)

def downsample(filters, size, apply_batchnorm=True, strides=2, activation=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    if activation:
        result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
