"""
The 'nodes' module contains methods to construct TF subgraphs computing the 1D or 2D DWT
or IDWT. Intended to be used if you need a DWT in your own TF graph.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *


@tf.function
def concat(a, b, ax):
    u1 = tf.unstack(a, axis=ax)
    u2 = tf.unstack(b, axis=ax)
    u1 += u2
    return tf.stack(u1, axis=ax)

@tf.function
def split(a, ax):
    x = tf.unstack(a, axis=ax)
    return tf.stack(x[:a.shape[ax]//2], axis=ax), tf.stack(x[a.shape[ax]//2:],axis=ax)



def cyclic_conv1d(input_node, filter_):
    """
    Cyclic convolution

    Args:
        input_node:  Input signal (3-tensor [batch, width, in_channels])
        filter_:     Filter

    Returns:
        Tensor with the result of a periodic convolution
    """
    # Create shorthands for TF nodes
    kernel_node = filter_.coeffs
    tl_node, tr_node, bl_node, br_node = filter_.edge_matrices

    # Do inner convolution
    inner = tf.nn.conv1d(input_node, kernel_node[::-1], stride=1, padding='VALID')

    # Create shorthands for shapes
    input_shape = tf.shape(input_node)
    tl_shape = tf.shape(tl_node)
    tr_shape = tf.shape(tr_node)
    bl_shape = tf.shape(bl_node)
    br_shape = tf.shape(br_node)

    # Slices of the input signal corresponding to the corners
    tl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, tl_shape[2], -1])
    tr_slice = tf.slice(input_node,
                        [0, input_shape[1] - tr_shape[2], 0],
                        [-1, tr_shape[2], -1])
    bl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, bl_shape[2], -1])
    br_slice = tf.slice(input_node,
                        [0, input_shape[1] - br_shape[2], 0],
                        [-1, br_shape[2], -1])

    # TODO: It just werks (It's the magic of the algorithm). i.e. Why do we have to transpose?
    tl = tl_node @ tf.transpose(tl_slice, perm=[2, 1, 0])
    tr = tr_node @ tf.transpose(tr_slice, perm=[2, 1, 0])
    bl = bl_node @ tf.transpose(bl_slice, perm=[2, 1, 0])
    br = br_node @ tf.transpose(br_slice, perm=[2, 1, 0])

    head = tf.transpose(tl + tr, perm=[2, 1, 0])
    tail = tf.transpose(bl + br, perm=[2, 1, 0])

    return tf.concat((head, inner, tail), axis=1)

def cyclic_conv1d_alt(input_node, filter_):
        """
        Alternative cyclic convolution. Uses more memory than cyclic_conv1d.

        Args:
            input_node:         Input signal
            filter_ (Filter):   Filter object

        Returns:
            Tensor with the result of a periodic convolution.
        """
        kernel_node = filter_.coeffs

        N = int(input_node.shape[2])

        start = N - filter_.num_neg()
        end = filter_.num_pos() - 1
        # Perodically extend input signal
        input_new = tf.concat(
            (input_node[:, :, start:, :], input_node, input_node[:, :, 0:end, :]),
            axis=2
        )

        # Convolve with periodic extension
        result = filter_.filt(input_new)

        return result


class CYCLCONV(tf.keras.layers.Layer):
    def __init__(self):
        super(CYCLCONV, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=2)
        #self.input_new = tf.zeros((None,64,64,1))

    def __call__(self, input_node, filter_):
        """
        Alternative cyclic convolution. Uses more memory than cyclic_conv1d.

        Args:
            input_node:         Input signal
            filter_ (Filter):   Filter object

        Returns:
            Tensor with the result of a periodic convolution.
        """

        N = int(input_node.shape[2])
        #
        start = N - filter_.num_neg()
        end = filter_.num_pos() - 1
        # # Perodically extend input signal
        input_new = self.concat(
            [input_node[:,:, start:, :], input_node, input_node[:,:, 0:end, :]],
        )
        # Convolve with periodic extension
        input_node = filter_.filt(input_new)

        return input_node





def dwt1d(input_node, wavelet, levels=1):
    """
    Constructs a TF computational graph computing the 1D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [batch, signal, channels].
        wavelet:        Wavelet object
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * (levels + 1)

    last_level = input_node
    for level in range(levels):
        lp_res = cyclic_conv1d_alt(last_level, wavelet.decomp_lp)[:,:, ::2, :]
        hp_res = cyclic_conv1d_alt(last_level, wavelet.decomp_hp)[:,:, 1::2, :]

        last_level = lp_res
        coeffs[levels - level] = hp_res

    coeffs[0] = last_level
    return tf.concat(coeffs, axis=2)


def dwt2d(input_node, wavelet, levels=1):
    """
    Constructs a TF computational graph computing the 2D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [rows, cols, channels].
        wavelet:        Wavelet object.
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * levels

    last_level = input_node
    #print(input_node.shape[0], input_node.shape[1])
    m, n = int(input_node.shape[1]), int(input_node.shape[2])

    for level in range(levels):
        local_m, local_n = m // (2 ** level), n // (2 ** level)

        first_pass = dwt1d(last_level, wavelet, 1)
        x = tf.transpose(first_pass, perm=[0, 2, 1, 3])
        second_pass = tf.transpose(
            dwt1d(
                tf.transpose(first_pass, perm=[0, 2, 1, 3]),
                wavelet,
                1
            ),
            perm=[0, 2, 1, 3]
        )

        last_level = tf.slice(second_pass, [0,0, 0, 0], [-1,local_m // 2, local_n // 2, 1])
        coeffs[level] = [
            tf.slice(second_pass, [0,local_m // 2, 0, 0], [-1, local_m // 2, local_n // 2, 1]),
            tf.slice(second_pass, [0,0, local_n // 2, 0], [-1, local_m // 2, local_n // 2, 1]),
            tf.slice(second_pass, [0,local_m // 2, local_n // 2, 0],
                     [-1, local_m // 2, local_n // 2, 1])
        ]

    for level in range(levels - 1, -1, -1):
        upper_half = tf.concat([last_level, coeffs[level][0]], -1)
        lower_half = tf.concat([coeffs[level][1], coeffs[level][2]], -1)

        last_level = tf.concat([upper_half, lower_half], -1)

    return last_level

class IDWT1D(tf.keras.layers.Layer):
    def __init__(self):
        super(IDWT1D, self).__init__()
        self.cyclconv = CYCLCONV()
        self.test_concat = tf.keras.layers.Concatenate(axis=2)
        self.crop1 = crop(2, 0, 32)
        self.crop2 = crop(2, 32, 64)


    def upsample(self, input_node, odd=False):
        """Upsamples. Doubles the length of the input, filling with zeros

        Args:
            input_node: 3-tensor [batch, spatial dim, channels] to be upsampled
            odd:        Bool, optional. If True, content of input_node will be
                        placed on the odd indeces of the output. Otherwise, the
                        content will be places on the even indeces. This is the
                        default behaviour.

        Returns:
            The upsampled output Tensor.
        """

        columns = []
        for col in tf.unstack(input_node, axis=2):
            columns.extend([col, tf.zeros_like(col)])

        if odd:
            # https://stackoverflow.com/questions/30097512/how-to-perform-a-pairwise-swap-of-a-list
            # TODO: Understand
            # Rounds down to even number
            l = len(columns) & -2
            columns[1:l:2], columns[:l:2] = columns[:l:2], columns[1:l:2]

        # TODO: Should we actually expand the dimension?
        return tf.stack(columns, axis=2)

    def __call__(self, input_node, wavelet, levels=1):
        """
        Constructs a TF graph that computes the 1D inverse DWT for a given wavelet.

        Args:
            input_node (tf.placeholder):             Input signal. A 3D tensor with dimensions
                                                     as [batch, signal, channels]
            wavelet (tfwavelets.dwtcoeffs.Wavelet):  Wavelet object.
            levels (int):                            Number of levels.

        Returns:
            Output node of IDWT graph.
        """
        # m, n = int(input_node.shape[1]), int(input_node.shape[2])

        #first_n = n // (2 ** levels)
        #last_level = tf.slice(input_node, [0, 0, 0, 0], [-1, m, first_n, 1])

        for level in range(levels - 1, -1 , -1):
            #local_n = n // (2 ** level)

            last_level = self.crop1(input_node)#tf.slice(input_node, [0, 0, local_n//2, 0], [-1, m, local_n//2, 1])
            detail = self.crop2(input_node)
            print(detail.shape, last_level.shape)

            lowres_padded = self.upsample(last_level, odd=False)
            detail_padded = self.upsample(detail, odd=True)

            lowres_filtered = self.cyclconv(lowres_padded, wavelet.recon_lp)
            detail_filtered = self.cyclconv(detail_padded, wavelet.recon_hp)

            last_level = lowres_filtered + detail_filtered
        return last_level

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


class IDWT2D(tf.keras.layers.Layer):
    def __init__(self):
        super(IDWT2D, self).__init__()
        self.idwt1d = IDWT1D()
        self.concat1 = tf.keras.layers.Concatenate(axis=1)
        self.concat2 = tf.keras.layers.Concatenate(axis=2)
        # todo: implement multilevel crop
        self.crop1 = crop(3, 0, 1)
        self.crop2 = crop(3, 1, 2)
        self.crop3 = crop(3, 2, 3)
        self.crop4 = crop(3, 3, 4)

    def __call__(self, input_node, wavelet, levels=1):
        """
        Constructs a TF graph that computes the 2D inverse DWT for a given wavelet.

        Args:
            input_node (tf.placeholder):             Input signal. A 3D tensor with dimensions
                                                     as [rows, cols, channels]
            wavelet (tfwavelets.dwtcoeffs.Wavelet):  Wavelet object.
            levels (int):                            Number of levels.

        Returns:
            Output node of IDWT graph.
        """

        last_level = self.crop1(input_node)

        detail_tr = self.crop2(input_node)
        detail_bl = self.crop3(input_node)
        detail_br = self.crop4(input_node)

        upper_half = self.concat1([last_level, detail_tr])#tf.concat([last_level, detail_tr], 1)
        lower_half = self.concat1([detail_bl, detail_br])#tf.concat([detail_bl, detail_br], 1)

        this_level = self.concat2([upper_half, lower_half])#tf.concat([upper_half, lower_half], 2)
        first_pass = tf.transpose(
            self.idwt1d(
                tf.transpose(this_level, perm=[0,2, 1, 3]),
                wavelet,
                1
            ),
            perm=[0,2, 1, 3]
        )
        # # #Second pass, corresponding to first pass in dwt2d
        second_pass = self.idwt1d(first_pass, wavelet, 1)

        last_level = second_pass

        return last_level