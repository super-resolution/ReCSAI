import tensorflow as tf
import math as m

import tensorflow_probability as tfp

@tf.function
def compute_loss_tfp(truth, predict, noiseless_gt,  data):
    #todo: use this to convert to distribution:
    tfp.layers.MultivariateNormalTriL(encoded_size, lambda s: s.sample(10))


    pass



@tf.function
def compute_loss_decode_ncs(truth, predict):
    return loc_loss(truth, predict) + count_loss(truth, predict)  # + b_loss

@tf.function
def compute_loss_decode(truth, predict, noiseless_gt,  data):
    l2 = tf.keras.losses.MeanSquaredError()
    b_loss = l2(data[:, :, :, 1] - noiseless_gt[:, :, :, 1], predict[:, :, :, 7])
    return loc_loss(truth,predict) + count_loss(truth, predict) + b_loss #+ loss


@tf.function
def compute_cs_loss(cs_out, noiseless_gt, mat):
    loss = tf.constant(0.0)
    for cs_slice in cs_out:
        inp_cs = tf.unstack(cs_slice, axis=-1)  #
        inp_ns = tf.unstack(noiseless_gt, axis=-1)
        for i, j in zip(inp_cs, inp_ns):
            res = tf.linalg.matvec(tf.transpose(mat), tf.keras.layers.Reshape((5184,), )(i), )
            val = tf.keras.layers.Reshape((9, 9), )(res / (0.001 + tf.reduce_max(tf.abs(res), axis=[1], keepdims=True)))
            ns = (j / (0.001 + tf.reduce_max(tf.abs(j), axis=[1], keepdims=True)))
            loss += 300 * tf.reduce_mean(tf.square(val - ns))
            loss += 1500 * tf.abs(tf.reduce_mean(i))  # sparsity constraint
    return loss

@tf.function
def count_loss(truth, predict):
    count = tf.zeros(tf.shape(truth)[0])
    for j in range(10):
        count += truth[:, j, 2]
    sigma_c = tf.keras.backend.sum((predict[:, :, :, 2]+0.0001) * (1 - predict[:, :, :, 2]), axis=[-1, -2])
    t = tf.keras.backend.sum(predict[:, :, :, 2], axis=[-1, -2])
    ten =  1 / 2 * tf.square(t - count)/ sigma_c - tf.math.log(tf.sqrt(2 * m.pi * sigma_c))
    c_loss = tf.reduce_sum(ten)

    return c_loss

@tf.function
def loc_loss(truth, predict):
    x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    X, Y = tf.meshgrid(x, x)
    L2 = tf.constant(0.0)
    count = tf.zeros(tf.shape(truth)[0])
    for j in range(10):
        count += truth[:, j, 2]
    for i in range(10):
        ten = truth[:, i, 2]/ (count+0.001) * (-tf.math.log(tf.keras.backend.sum(
            predict[:, :, :, 2] / (tf.keras.backend.sum(predict[:, :, :, 2], axis=[-1,-2], keepdims=True)
                                   *
                                   tf.math.sqrt(predict[:, :, :, 3] * predict[:, :, :, 4] * predict[:, :, :, 6]
                                                * (2 * tf.constant(m.pi)) ** 3)
                                   )

            * tf.math.exp(-1 / 2 * (
                    tf.square(
                        predict[:, :, :, 0] - (
                                    truth[:, i:i + 1, 0:1] - Y))
                    / (predict[:, :, :, 3])
                    + tf.square(predict[:, :, :, 1] - (truth[:, i:i + 1, 1:2] - X))
                    / (predict[:, :, :, 4])
                    + tf.square(predict[:, :, :, 5] - truth[:, i:i + 1, 3:4]) / predict[:, :, :, 6]
            ))
            , axis=[-1, -2])))

        L2 += tf.reduce_sum(ten)
    return L2
