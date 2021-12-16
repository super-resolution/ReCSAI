"""
The 'nodes' module contains methods to construct TF subgraphs computing the 1D or 2D DWT
or IDWT. Intended to be used if you need a DWT in your own TF graph.
"""

import tensorflow as tf
from tensorflow.keras.layers import *

def cyclic_conv1d_alt(input_node, filter_, mode):
        """
        Alternative cyclic convolution. Uses more memory than cyclic_conv1d.

        Args:
            input_node:         Input signal
            filter_ (Filter):   Filter object

        Returns:
            Tensor with the result of a periodic convolution.
        """
        N = int(input_node.shape[2])
        if mode == "lp":
            kernel = filter_.coeffs[0:1]

            start = N - filter_.zero_lp
            end = filter_.coeffs.shape[1] - filter_.zero_lp - 1
            # Perodically extend input signal
        if mode == "hp":
            kernel = filter_.coeffs[1:2]
            start = N - filter_.zero_hp
            end = filter_.coeffs.shape[1] - filter_.zero_hp - 1
        input_new = tf.concat(
            (input_node[:, :, start:, :], input_node, input_node[:, :, 0:end, :]),
            axis=2
        )

        # Convolve with periodic extension
        result = tf.nn.conv2d(input_new, kernel, strides=(1,1), padding="VALID")

        return result


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
        lp_res = cyclic_conv1d_alt(last_level, wavelet.filter, "lp")[:,:, ::2, :]
        hp_res = cyclic_conv1d_alt(last_level, wavelet.filter, "hp")[:,:, 1::2, :]

        last_level = lp_res
        coeffs[levels - level] = hp_res

    coeffs[0] = last_level
    return tf.concat(coeffs, axis=2)

class DWT2D(Layer):
    def __init__(self,shape, level ):
        super(DWT2D, self).__init__()
        self.padding = []
        amplifier = shape/128
        self.levels = level
        pad = 0
        for i in range(level):
            self.padding.append(ZeroPadding2D(padding=(int(pad), int(pad)), data_format="channels_last"))
            pad += int(amplifier*2**(4-i))#todo: this depends on image size!
            #x=0

    def __call__(self, input_node, wavelet):
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

        coeffs = [None] * self.levels

        last_level = input_node
        #print(input_node.shape[0], input_node.shape[1])
        m, n = int(input_node.shape[1]), int(input_node.shape[2])

        for level in range(self.levels):
            local_m, local_n = m // (2 ** level), n // (2 ** level)

            first_pass = dwt1d(last_level, wavelet, 1)
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
        last_level = self.padding[self.levels-1](last_level)
        for level in range(self.levels - 1, -1, -1):
            upper_half = tf.concat([(last_level),
                                    self.padding[level](coeffs[level][0])], -1)
            lower_half = tf.concat([self.padding[level](coeffs[level][1]),
                                    self.padding[level](coeffs[level][2])], -1)

            last_level = tf.concat([upper_half, lower_half], -1)

        return last_level

class IDWT1D(tf.keras.layers.Layer):
    def __init__(self, h_shape):
        super(IDWT1D, self).__init__()#todo: need shape here too
        #self.cyclconv = CYCLCONV()
        self.test_concat = tf.keras.layers.Concatenate(axis=2)
        no_crop = (0,-1)
        self.crop1 = crop(axis1=no_crop, axis2=(0,h_shape), axis3=no_crop)
        self.crop2 = crop(axis1=no_crop, axis2=(h_shape,2*h_shape), axis3=no_crop)
        self.concat = tf.keras.layers.Concatenate(axis=2)

    def cyclconv(self, input_node, filter_, mode):
        N = int(input_node.shape[2])
        if mode == "lp":
            start = N - filter_.zero_hp #swap hp lp zero
            end = len(filter_._coeffs) - filter_.zero_hp - 1
            kernel = filter_.coeffs[0:1,::-1]
        if mode == "hp":
            start = N - filter_.zero_lp #swap hp lp zero
            end = len(filter_._coeffs) - filter_.zero_lp - 1
            # # Perodically extend input signal
            kernel = filter_.coeffs[1:2,::-1]
        input_new = self.concat(
            [input_node[:,:, start:, :], input_node, input_node[:,:, 0:end, :]],
        )
        # Convolve with periodic extension
        input_new = tf.nn.conv2d(input_new, kernel, strides=(1,1), padding="VALID")
        return input_new


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

            last_level, detail = tf.split(input_node, 2, axis=2)#tf.slice(input_node, [0, 0, local_n//2, 0], [-1, m, local_n//2, 1])
            #detail = self.crop2(input_node)
            print(detail.shape, last_level.shape)

            lowres_padded = self.upsample(last_level, odd=False)
            detail_padded = self.upsample(detail, odd=True)

            lowres_filtered = self.cyclconv(lowres_padded, wavelet.filter, "lp")
            detail_filtered = self.cyclconv(detail_padded, wavelet.filter, "hp")

            last_level = lowres_filtered + detail_filtered
        return last_level

def crop(axis1,axis2,axis3):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        return x[:, axis1[0]:axis1[1], axis2[0]:axis2[1], axis3[0]:axis3[1]]
    return func


class IDWT2D(tf.keras.layers.Layer):
    def __init__(self,shape, level ):#todo: use tf shape here
        super(IDWT2D, self).__init__()
        h_shape = int(shape/2)
        self.levels = level
        self.idwt1d = IDWT1D(h_shape)
        self.concat1 = tf.keras.layers.Concatenate(axis=1)
        self.concat2 = tf.keras.layers.Concatenate(axis=2)
        x = y = [int((h_shape - h_shape * 0.5 ** (level-1))/2),h_shape-int((h_shape - h_shape * 0.5 ** (level-1))/2)]
        self.crops = [crop(x,y, [0,1])]
        j=0
        for i in range(level- 1, -1, -1):
            n = int((h_shape - h_shape * 0.5 ** (i))/2)
            x = y = [n,h_shape-n] # done image size here
            self.crops.append(crop(x, y, [1 + j * 3, 2 + j * 3]))
            self.crops.append(crop(x, y, [2 + j * 3, 3 + j * 3]))
            self.crops.append(crop(x, y, [3 + j * 3, 4 + j * 3]))
            j+=1
        self.crops.reverse()

    def __call__(self, input_node, wavelet, ):
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
        last_level = self.crops[self.levels*3](input_node)
        for level in range(self.levels - 1, -1, -1):


            detail_tr = self.crops[level*3+2](input_node)
            detail_bl = self.crops[level*3+1](input_node)
            detail_br = self.crops[level*3+0](input_node)

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