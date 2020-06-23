import custom_nodes as nodes
import tfwavelets
from tfwavelets.dwtcoeffs import haar, db3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


class LFilter(tf.keras.layers.Layer):
    """Filterbase for Wavelet Layers"""
    def __init__(self, initial_coeffs, zero, name, trainable=True):
        super(LFilter, self).__init__()
        self._coeffs = initial_coeffs.astype(np.float32)

        self.zero = zero

        self.coeffs = tf.Variable(
            initial_value=tfwavelets.utils.adapt_filter(initial_coeffs),
            trainable=trainable,
            name=name,
            dtype=tf.float32,
            constraint=tf.keras.constraints.max_norm(np.sqrt(2), [1, 2])
        )

        def my_initializer(shape, dtype=None):
            return tf.constant(initial_coeffs)[tf.newaxis,:,tf.newaxis,tf.newaxis ]
        self.filt = tf.keras.layers.Conv2D(1, (1,initial_coeffs.shape[0]),
                                           kernel_constraint=tf.keras.constraints.max_norm(np.sqrt(2), [1, 2]),
                                           dtype=tf.float32,
                                           name=name+"filter",
                                           kernel_initializer=my_initializer, padding="valid")

        # Erase stuff that will be invalid once the filter coeffs has changed
        self._coeffs = [None] * len(self._coeffs)
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

    def _edge_matrices(self):
        """Computes the submatrices needed at the ends for circular convolution.

        Returns:
            Tuple of 2d-arrays, (top-left, top-right, bottom-left, bottom-right).
        """
        if not isinstance(self._coeffs, np.ndarray):
            self._coeffs = np.array(self._coeffs)

        n, = self._coeffs.shape
        self._coeffs = self._coeffs[::-1]

        # Some padding is necesssary to keep the submatrices
        # from having having columns in common
        padding = max((self.zero, n - self.zero - 1))
        matrix_size = n + padding
        filter_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        negative = self._coeffs[
                   -(self.zero + 1):]  # negative indexed filter coeffs (and 0)
        positive = self._coeffs[
                   :-(self.zero + 1)]  # filter coeffs with strictly positive indeces

        # Insert first row
        filter_matrix[0, :len(negative)] = negative

        # Because -0 == 0, a length of 0 makes it impossible to broadcast
        # (nor is is necessary)
        if len(positive) > 0:
            filter_matrix[0, -len(positive):] = positive

        # Cycle previous row to compute the entire filter matrix
        for i in range(1, matrix_size):
            filter_matrix[i, :] = np.roll(filter_matrix[i - 1, :], 1)

        # TODO: Indexing not thoroughly tested
        num_pos = len(positive)
        num_neg = len(negative)
        top_left = filter_matrix[:num_pos, :(num_pos + num_neg - 1)]
        top_right = filter_matrix[:num_pos, -num_pos:]
        bottom_left = filter_matrix[-num_neg + 1:, :num_neg - 1]
        bottom_right = filter_matrix[-num_neg + 1:, -(num_pos + num_neg - 1):]

        # Indexing wrong when there are no negative indexed coefficients
        if num_neg == 1:
            bottom_left = np.zeros((0, 0), dtype=np.float32)
            bottom_right = np.zeros((0, 0), dtype=np.float32)

        return top_left, top_right, bottom_left, bottom_right


class DWT2(tf.keras.layers.Layer):
    """Wavelet transform"""
    def __init__(self):
        super(DWT2, self).__init__()
        self.decomp_lp = LFilter(haar.decomp_lp._coeffs, haar.decomp_lp.zero, "decomp_lp")
        self.decomp_hp = LFilter(haar.decomp_hp._coeffs, haar.decomp_hp.zero, "decomp_hp")
        self.recon_lp = LFilter(haar.recon_lp._coeffs, haar.recon_lp.zero, "recon_lp")
        self.recon_hp = LFilter(haar.recon_hp._coeffs, haar.recon_hp.zero, "recon_hp")

    def call(self, input):
        #print(input.shape)
        t = nodes.dwt2d(input, wavelet=self)
        #t = tf.keras.activations.sigmoid(input)
        return t


class IDWT2(tf.keras.layers.Layer):
    """Inverse Wavelet transform"""
    def __init__(self):
        super(IDWT2, self).__init__()
        self.decomp_lp = LFilter(haar.decomp_lp._coeffs, haar.decomp_lp.zero, "decomp_lp")
        self.decomp_hp = LFilter(haar.decomp_hp._coeffs, haar.decomp_hp.zero, "decomp_hp")
        self.recon_lp = LFilter(haar.recon_lp._coeffs, haar.recon_lp.zero, "recon_lp")
        self.recon_hp = LFilter(haar.recon_hp._coeffs, haar.recon_hp.zero, "recon_hp")
        self.idwt2d = nodes.IDWT2D()

    def call(self, input):
        #print(input.shape)
        t = self.idwt2d(input, wavelet=self)
        return t



class FullWavelet(tf.keras.layers.Layer):
    """Wavelet transform + activation Layer + Inverse Wavelet transform.
       Recon Wavelet coefficients depend on decomp Wavelet coefficients
    """
    def __init__(self):
        super(FullWavelet, self).__init__()
        self.decomp_lp = LFilter(db3.decomp_lp._coeffs, db3.decomp_lp.zero, "decomp_lp")
        self.decomp_hp = LFilter(db3.decomp_hp._coeffs, db3.decomp_hp.zero, "decomp_hp")
        self.recon_lp = LFilter(db3.recon_lp._coeffs, db3.recon_lp.zero, "recon_lp", trainable=False)
        # self.recon_lp.filt.kernel = tf.reverse(self.decomp_lp.filt.kernel, axis=1)
        # self.recon_lp.filt._trainable_weights = []
        # self.recon_lp.filt.trainable_weights.append(self.decomp_lp.filt.kernel)

        self.recon_hp = LFilter(db3.recon_hp._coeffs, db3.recon_hp.zero, "recon_hp", trainable=False)
        # self.recon_hp.filt.kernel = tf.reverse(self.decomp_hp.filt.kernel, axis=1)
        # self.recon_hp.filt._trainable_weights = []
        # self.recon_hp.filt.trainable_weights.append(self.decomp_hp.filt.kernel)
        #self.recon_lp.coeffs = tf.reverse(self.decomp_lp.coeffs, tf.constant([0], dtype=tf.int32))
        #self.recon_hp.coeffs = tf.reverse(self.decomp_hp.coeffs, tf.constant([0], dtype=tf.int32))
        self.bias = tf.Variable( initial_value=tf.zeros(1),
            trainable=True,
            name="bias",
            dtype=tf.float32,)
        self.activation = tf.keras.layers.ReLU(max_value=1.0)
        self.idwt2d = nodes.IDWT2D()

    def call(self, inp):
        t = nodes.dwt2d(inp, wavelet=self,)
        t = tf.math.subtract(t, self.bias)
        #t = self.split(t)


        t = tf.keras.activations.relu(t)
        #self.recon_lp.coeffs = tf.reverse(self.decomp_lp.coeffs, tf.constant([0], dtype=tf.int32))
        #self.recon_hp.coeffs = tf.reverse(self.decomp_hp.coeffs, tf.constant([0], dtype=tf.int32))
        t = self.idwt2d(t, wavelet=self)
        t = tf.keras.activations.sigmoid(t)
        return t