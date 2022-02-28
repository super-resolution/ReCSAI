import tensorflow as tf
import math as m



@tf.function
def compute_loss_decode_ncs(truth, predict):
    return loc_loss(truth, predict) + count_loss(truth, predict)  # + b_loss

@tf.function
def compute_loss_decode(truth, predict, noiseless_gt,  data):
    l2 = tf.keras.losses.MeanSquaredError()  # todo: switched to l1

    # todo: loss for cs part...
    # loss = tf.constant(0.0)
    # for cs_slice in cs_out:
    #     inp_cs = tf.unstack(cs_slice, axis=-1)  # todo: this needs to be an input
    #     inp_ns = tf.unstack(noiseless_gt, axis=-1)  # todo: this needs to be an input
    #     for i, j in zip(inp_cs, inp_ns):
    #         res = tf.linalg.matvec(tf.transpose(mat), tf.keras.layers.Reshape((5184,), )(i), )
    #         val = tf.keras.layers.Reshape((9, 9), )(res / (0.001 + tf.reduce_max(tf.abs(res), axis=[1], keepdims=True)))
    #         ns = (j / (0.001 + tf.reduce_max(tf.abs(j), axis=[1], keepdims=True)))
    #         loss += 300 * tf.reduce_mean(tf.square(val - ns))
    #         loss += 1500 * tf.abs(tf.reduce_mean(i))  # sparsity constraint


    b_loss = l2(data[:, :, :, 1] - noiseless_gt[:, :, :, 1], predict[:, :, :, 7])
    return  loc_loss(truth,predict) + count_loss(truth, predict) +  b_loss #+ loss



@tf.function
def count_loss(truth, predict):
    count = tf.zeros(tf.shape(truth)[0])
    for j in range(10):
        count += truth[:, j, 2]
    sigma_c = tf.keras.backend.sum((predict[:, :, :, 2]+0.01) * (1 - predict[:, :, :, 2]), axis=[-1, -2])
    # i=0#todo: learn background and photon count
    t = tf.keras.backend.sum(predict[:, :, :, 2], axis=[-1, -2])
    L2 = 1 / 2 * tf.square(t - count)
    ten =  1 / 2 * tf.square(t - count)/ sigma_c - tf.math.log(tf.sqrt(2 * m.pi * sigma_c))
    c_loss = tf.reduce_sum(ten)
    #c_loss = tf.reduce_sum(tf.gather(ten, tf.where(count>0.9)))
    #c_loss += tf.reduce_sum(tf.gather(L2, tf.where(count<0.1)))

    return c_loss

@tf.function
def loc_loss(truth, predict):

    x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
    X, Y = tf.meshgrid(x, x)
    L2 = tf.constant(0.0)
    count = tf.zeros(tf.shape(truth)[0])
    for j in range(10):
        count += truth[:, j, 2]
    for i in range(10):  # up to 10 localisations
        ten = truth[:, i, 2]/ (count+0.001) * (-tf.math.log(tf.keras.backend.sum(
            predict[:, :, :, 2] / (tf.keras.backend.sum(predict[:, :, :, 2], axis=[-1,-2], keepdims=True)
                                   *
                                   tf.math.sqrt(predict[:, :, :, 3]**2 * predict[:, :, :, 4]**2 * predict[:, :, :, 6]**2
                                                * (2 * tf.constant(m.pi)) ** 3)
                                   )

            * tf.math.exp(-1 / 2 * (
                    tf.square(
                        predict[:, :, :, 0] - (
                                    truth[:, i:i + 1, 0:1] - Y))  # todo: test that this gives expected values
                    / (predict[:, :, :, 3]**2)
                    + tf.square(predict[:, :, :, 1] - (truth[:, i:i + 1, 1:2] - X))
                    / (predict[:, :, :, 4]**2)
                    + tf.square(predict[:, :, :, 5] - truth[:, i:i + 1, 3:4]) / predict[:, :, :, 6]**2
            ))
            , axis=[-1, -2])))  # todo: activation >= 0

        L2 += tf.reduce_sum(ten)
        # L2 += tf.reduce_sum(truth[:, i, 2] / (count + 0.001) * (-tf.math.log(tf.keras.backend.sum(
        #     predict[:, :, :, 2] / (tf.keras.backend.sum(predict[:, :, :, 2], axis=[-1,-2], keepdims=True)
        #                            *
        #                            tf.math.sqrt(predict[:, :, :, 3]**2 * predict[:, :, :, 4]**2 * predict[:, :, :, 6]**2
        #                                         * (2 * tf.constant(m.pi)) ** 3)
        #                            )
        #
        #     * tf.math.exp(-1 / 2 * (
        #             tf.square(
        #                 predict[:, :, :, 0] - (
        #                             truth[:, i:i + 1, 0:1] - Y))  # todo: test that this gives expected values
        #             / (predict[:, :, :, 3]**2)
        #             + tf.square(predict[:, :, :, 1] - (truth[:, i:i + 1, 1:2] - X))
        #             / (predict[:, :, :, 4]**2)
        #             + tf.square(predict[:, :, :, 5] - truth[:, i:i + 1, 3:4]) / predict[:, :, :, 6]**2
        #     ))
        #     , axis=[-1, -2])))  # todo: activation >= 0
        #                     )
    return L2
