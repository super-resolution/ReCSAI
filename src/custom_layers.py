from tfwavelets.dwtcoeffs import haar, db3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from .utility import *
from . import custom_nodes as nodes

class CompressedSensing(tf.keras.layers.Layer):
    def __init__(self):
        #todo: gaussian as initial value
        self.mat = tf.Variable(initial_value=self.psf_initializer())
        self.mu = tf.constant(np.ones((1)), dtype=tf.float64)
        self.lam = tf.Variable(initial_value=np.ones((1)), dtype=tf.float64)*0.12
        self.t = tf.Variable(np.array([1]),dtype=tf.float64)

    def psf_initializer(self):
        mat = create_psf_matrix(9, 8)
        return tf.constant(mat)[:,:,tf.newaxis]

    def softthresh(self, input, lam):
        one = simulate_where_add(input, lam, tf.constant([np.inf],dtype=tf.float64), -lam)
        two = simulate_where_add(input, tf.constant([-np.inf],dtype=tf.float64), -lam, lam)
        return one+two

    def __call__(self, input):
        #todo: fista here
        y = tf.constant(np.zeros((5184)))
        for i in range(100):
            #todo: has to work for image stack
            re =tf.linalg.matvec(self.mat[:,:,0], tf.keras.backend.flatten(input)- tf.linalg.matvec(tf.transpose(self.mat[:,:,0]), y))
            w = y+1/self.mu*re
            y_new = self.softthresh(w, self.lam/self.mu)
            t_n = (1+tf.math.sqrt(1+self.t**2))/2
            y = y_new+ (self.t-1)/t_n*(y_new-y)
            self.t = t_n
        return y


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


class DWT2(tf.keras.layers.Layer):
    """Wavelet transform"""
    def __init__(self):
        super(DWT2, self).__init__()
        self.filter = LFilter(haar.decomp_lp._coeffs,haar.decomp_hp._coeffs, haar.decomp_lp.zero, haar.decomp_hp.zero, "filter")
        self.dwt2 = nodes.DWT2D(level=3)


    def call(self, input):
        t = self.dwt2(input, wavelet=self)
        return t


class IDWT2(tf.keras.layers.Layer):
    """Inverse Wavelet transform"""
    def __init__(self):
        super(IDWT2, self).__init__()
        self.filter = LFilter(haar.decomp_lp._coeffs, haar.decomp_hp._coeffs, haar.decomp_lp.zero,haar.decomp_hp.zero, "filter")
        self.idwt2d = nodes.IDWT2D(level=3)

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
        t = tf.keras.activations.relu(t)
        t = self.idwt2d(t, wavelet=self)
        return t