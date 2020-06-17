import tfwavelets
from tfwavelets.dwtcoeffs import haar
import numpy as np
import tensorflow as tf


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
        print(input.shape)
        t = tfwavelets.nodes.dwt2d(input[0], wavelet=self)
        return t[tf.newaxis, ...]


class IDWT2(tf.keras.layers.Layer):
    """Inverse Wavelet transform"""
    def __init__(self):
        super(IDWT2, self).__init__()
        self.wavelet = tfwavelets.dwtcoeffs.TrainableWavelet(tfwavelets.dwtcoeffs.haar)
        self.rlp = self.wavelet.recon_lp.coeffs
        self.rhp = self.wavelet.recon_hp.coeffs
        self.dlp = self.wavelet.decomp_hp.coeffs
        self.dhp = self.wavelet.decomp_lp.coeffs

    def call(self, input):
        print(input.shape)
        t = tfwavelets.nodes.idwt2d(input[0], wavelet=self.wavelet)
        return t[tf.newaxis, ...]


class FullWavelet(tf.keras.layers.Layer):
    """Wavelet transform + activation Layer + Inverse Wavelet transform.
       Recon Wavelet coefficients depend on decomp Wavelet coefficients
    """
    def __init__(self):
        super(FullWavelet, self).__init__()
        self.decomp_lp = LFilter(haar.decomp_lp._coeffs, haar.decomp_lp.zero, "decomp_lp")
        self.decomp_hp = LFilter(haar.decomp_hp._coeffs, haar.decomp_hp.zero, "decomp_hp")
        self.recon_lp = LFilter(haar.recon_lp._coeffs, haar.recon_lp.zero, "recon_lp", trainable=False)
        self.recon_hp = LFilter(haar.recon_hp._coeffs, haar.recon_hp.zero, "recon_hp", trainable=False)

        self.relu = tf.keras.layers.ReLU()

    def call(self, inp):
        t = tfwavelets.nodes.dwt2d(inp[0], wavelet=self)
        t = self.relu(t[tf.newaxis, ...])
        self.recon_lp.coeffs = tf.reverse(self.decomp_lp.coeffs, tf.constant([1], dtype=tf.int32))
        self.recon_hp.coeffs = tf.reverse(self.decomp_hp.coeffs, tf.constant([1], dtype=tf.int32))
        t = tfwavelets.nodes.idwt2d(t[0], wavelet=self)
        return t[tf.newaxis, ...]