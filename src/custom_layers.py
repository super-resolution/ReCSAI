from tfwavelets.dwtcoeffs import haar, db3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from .utility import *
from . import custom_nodes as nodes

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


class CompressedSensing(tf.keras.layers.Layer):
    def __init__(self):
        super(CompressedSensing, self).__init__()

        #done: gaussian as initial value
        self.psf = get_psf(100, 128)  # todo: sigma px_size

        self.mat = tf.Variable(initial_value=self.psf_initializer(),trainable=False)
        self.mu = tf.constant(np.ones((1)), dtype=tf.float64)
        self.lam = tf.Variable(initial_value=np.ones((1)), dtype=tf.float64, name="lambda", trainable=True)*0.017#was0.005
        self.t = tf.Variable(np.array([1]),dtype=tf.float64)
        dense = lambda x: tf.sparse.to_dense(tf.SparseTensor(x[0], x[1], tf.shape(x[2], out_type=tf.int64)))
        self.sparse_dense = tf.keras.layers.Lambda(dense)

    def update_psf(self, psf):
        self.psf = psf
        self.mat = tf.Variable(initial_value=self.psf_initializer(),trainable=False)

    def psf_initializer(self):
        mat = create_psf_matrix(9, 8, self.psf)
        return tf.constant(mat)[:,:,tf.newaxis]

    def softthresh(self, input, lam):
        one = simulate_where_add(input, lam, tf.constant([np.inf],dtype=tf.float64), -lam, self.sparse_dense)
        two = simulate_where_add(input, tf.constant([-np.inf],dtype=tf.float64), -lam, lam, self.sparse_dense)
        return one+two

    #@tf.function
    def __call__(self, input):
        #done: fista here
        input = tf.cast(input, tf.float64)
        inp = tf.unstack(input, axis=-1)
        y = tf.constant(np.zeros((5329,3)))[tf.newaxis, :]
        y_n = tf.unstack(y, axis=-1)
        for i in range(len(inp)):
            x = inp[i]
            inp[i] = tf.reshape(x, (tf.shape(input)[0], input.shape[1]*input.shape[2]))
            y_new_last_it = tf.zeros_like(y_n[i])
            for j in range(100):
                re =tf.linalg.matvec(self.mat[:,:,0], inp[i] - tf.linalg.matvec(tf.transpose(self.mat[:,:,0]), y_n[i]))
                w = y_n[i]+1/self.mu*re
                y_new = self.softthresh(w, self.lam/self.mu)
                y_new = tf.cast(y_new, tf.float64)
                t_n = (1+tf.math.sqrt(1+4*self.t**2))/2
                y_n[i] = y_new+ (self.t-1)/t_n*(y_new-y_new_last_it)
                y_new_last_it = y_new
                self.t = t_n
            y_n[i] = tf.reshape(y_new, (tf.shape(input)[0], 8*input.shape[1]+1,8*input.shape[2]+1))
        y = tf.stack(y_n,axis=-1)
        #b = tf.cast(y_n[0], tf.float64)
        #i = tf.reduce_max(b)
        #x = tf.keras.activations.sigmoid((b/i-0.8)*50).numpy()
        return y#,tfa.image.connected_components(tf.cast(x+0.05, tf.int32))



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
        self.dwt2 = nodes.DWT2D(level=level)

    def call(self, inp):

        t = self.dwt2(inp, wavelet=self,)
        t = tf.math.subtract(t, self.bias)
        t = tf.keras.activations.relu(t)
        t = self.idwt2d(t, wavelet=self)

        return t