import custom_nodes as nodes
import tfwavelets
from tfwavelets.dwtcoeffs import haar, db3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

class BiasedSigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super(BiasedSigmoid, self).__init__()
        self.bias = tf.Variable(initial_value=[1.0])

    def __call__(self, inp):
        return tf.keras.activations.sigmoid(inp)

class OrthonormalConstraint(tf.keras.constraints.Constraint):
    def __call__(self, x):
        v,w = tf.unstack(x, axis=0)
        a =tf.greater(tf.abs(tf.reduce_sum(v[:,0,0]*w[:,0,0])),tf.constant([0.001]))
        w = tf.cond(tf.equal(a,tf.constant(True)),
            lambda: w - tf.reduce_sum(v*w)/tf.reduce_sum(v*v)*v,
            lambda: w)
        x = tf.stack([v,w])
        return x


class LFilter(tf.keras.layers.Layer):
    """Filterbase for Wavelet Layers"""
    def __init__(self, initial_coeffs_lp,initial_coeffs_hp, zero_lp, zero_hp, name, trainable=True):
        super(LFilter, self).__init__()
        self._coeffs_lp = initial_coeffs_lp.astype(np.float32)
        self._coeffs_hp = initial_coeffs_hp.astype(np.float32)


        self.zero_lp = zero_lp
        self.zero_hp = zero_hp
        def my_initializer(shape, dtype=None):
            return tf.constant([initial_coeffs_lp, initial_coeffs_hp])[:,:,tf.newaxis,tf.newaxis ]
        self.coeffs = tf.Variable(
            initial_value=my_initializer(0),
            trainable=trainable,
            name=name,
            dtype=tf.float32,
            constraint=OrthonormalConstraint()#todo: add new constraint here
        )

        # Erase stuff that will be invalid once the filter coeffs has changed
        self._coeffs = [None] * len(self._coeffs_lp)
        self.edge_matrices = None

    def __getitem__(self, item):
        """
        Returns filter coefficients at requested indeces. Indeces are offset by the filter
        origin

        Args:
            item (int or slice):    Item(s) to get

        Returns:
            np.ndarray: Item(s) at specified place(s)
        """
        if isinstance(item, slice):
            return self._coeffs.__getitem__(
                slice(item.start + self.zero, item.stop + self.zero, item.step)
            )
        else:
            return self._coeffs.__getitem__(item + self.zero)

    def num_pos(self):
        """
        Number of positive indexed coefficients in filter, including the origin. Ie,
        strictly speaking it's the number of non-negative indexed coefficients.

        Returns:
            int: Number of positive indexed coefficients in filter.
        """
        return len(self._coeffs) - self.zero

    def num_neg(self):
        """
        Number of negative indexed coefficients, excluding the origin.

        Returns:
            int: Number of negative indexed coefficients
        """
        return self.zero


class DWT2(tf.keras.layers.Layer):
    """Wavelet transform"""
    def __init__(self):
        super(DWT2, self).__init__()
        self.filter = LFilter(haar.decomp_lp._coeffs,haar.decomp_hp._coeffs, haar.decomp_lp.zero, haar.decomp_hp.zero, "filter")
        self.dwt2 = nodes.DWT2D(level=2)


    def call(self, input):
        #print(input.shape)
        t = self.dwt2(input, wavelet=self)
        #t = tf.keras.activations.sigmoid(input)
        return t


class IDWT2(tf.keras.layers.Layer):
    """Inverse Wavelet transform"""
    def __init__(self):
        super(IDWT2, self).__init__()
        self.filter = LFilter(haar.decomp_lp._coeffs, haar.decomp_hp._coeffs, haar.decomp_lp.zero,haar.decomp_hp.zero, "filter")
        self.idwt2d = nodes.IDWT2D(level=2)

    def call(self, input):
        #print(input.shape)
        t = self.idwt2d(input, wavelet=self)
        return t



class FullWavelet(tf.keras.layers.Layer):
    """Wavelet transform + activation Layer + Inverse Wavelet transform.
       Recon Wavelet coefficients depend on decomp Wavelet coefficients
    """
    def __init__(self, level=1):
        super(FullWavelet, self).__init__()
        self.filter = LFilter(db3.decomp_lp._coeffs, db3.decomp_hp._coeffs, db3.decomp_hp.zero, db3.decomp_lp.zero, "filter")
        #self.decomp_hp = LFilter(db3.decomp_hp._coeffs, db3.decomp_hp.zero, "decomp_hp")

        self.bias = tf.Variable( initial_value=tf.zeros(3*level+1),
            trainable=True,
            name="bias",
            dtype=tf.float32,)
        self.activation = tf.keras.layers.ReLU(max_value=1.0)
        self.idwt2d = nodes.IDWT2D(level=level)
        self.dwt2 = nodes.DWT2D(level=level)

    def call(self, inp):
        t = self.dwt2(inp, wavelet=self,)
        t = tf.math.subtract(t, self.bias)
        #t = self.split(t)


        t = tf.keras.activations.relu(t)
        #self.recon_lp.coeffs = tf.reverse(self.decomp_lp.coeffs, tf.constant([0], dtype=tf.int32))
        #self.recon_hp.coeffs = tf.reverse(self.decomp_hp.coeffs, tf.constant([0], dtype=tf.int32))
        t = self.idwt2d(t, wavelet=self)
        #t = tf.keras.activations.sigmoid(t)
        return t