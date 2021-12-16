import tensorflow as tf
import src.custom_nodes as nodes
import numpy as np
from tfwavelets.dwtcoeffs import haar, db3



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
    def __init__(self, shape):
        super(IDWT2, self).__init__()
        self.filter = LFilter(haar.decomp_lp._coeffs, haar.decomp_hp._coeffs, haar.decomp_lp.zero,haar.decomp_hp.zero, "filter")
        self.idwt2d = nodes.IDWT2D(shape, level=3)

    def call(self, input):
        t = self.idwt2d(input, wavelet=self)
        return t



class FullWavelet(tf.keras.layers.Layer):
    """Wavelet transform + activation Layer + Inverse Wavelet transform.
       Recon Wavelet coefficients depend on decomp Wavelet coefficients
    """
    def __init__(self, shape, level=1):
        super(FullWavelet, self).__init__()
        self.filter = LFilter(db3.decomp_lp._coeffs, db3.decomp_hp._coeffs, db3.decomp_hp.zero, db3.decomp_lp.zero, "filter")
        self.bias = tf.Variable( initial_value=tf.zeros(3*level+1),
            trainable=True,
            name="bias",
            dtype=tf.float32,)
        self.activation = tf.keras.layers.ReLU(max_value=1.0)
        self.idwt2d = nodes.IDWT2D(shape, level=level )
        self.dwt2 = nodes.DWT2D(shape, level=level)

    def call(self, inp, test=False):

        t = self.dwt2(inp, wavelet=self,)
        if not test:
            t = tf.math.subtract(t, self.bias)
            t = tf.keras.activations.relu(t)
        t = self.idwt2d(t, wavelet=self)

        return t