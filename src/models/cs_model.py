from src.custom_layers.cs_layers import CompressedSensing, CompressedSensingInception
#from tensorflow.keras.layers import *
import tensorflow as tf
from src.custom_layers.utility_layer import downsample,upsample
#import tensorflow_probability as tfp
import math as m

#todo: imitate yolo architecture and use 9x9 output grid
class CompressedSensingInceptionNet(tf.keras.Model):
    TYPE = 0
    def __init__(self):
        super(CompressedSensingInceptionNet, self).__init__()
        self.inception1 = CompressedSensingInception(iterations=10)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.inception2 = CompressedSensingInception(iterations=100)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        #todo: extend depth?
        self.ReduceSum = tf.keras.layers.Lambda(lambda z: tf.keras.backend.sum(z, axis=[-1,-2]))
        self.ReduceSumKD = tf.keras.layers.Lambda(lambda z: tf.keras.backend.sum(z, axis=[-1,-2], keepdims=True))


        self.horizontal_path = [
            tf.keras.layers.Conv2D(256, (1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding="same"),
            # tf.keras.layers.Conv2D(128, (7, 1), activation=None, padding="same"),
            # tf.keras.layers.Conv2D(64, (1, 7), activation=None, padding="same"),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),#todo dont do max pooling...
            tf.keras.layers.BatchNormalization(),#todo: only use activation in horizontal layer?
            tf.keras.layers.Conv2D(32,(1,1), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding="same"),
            tf.keras.layers.Conv2D(8, (3, 3), activation=None, padding="same"),
        ]#x,y,sigmax,sigmay,classifier

        #todo: only need one actvation
        def sigmoid_activation(inputs):#softplus activation
            inputs_list = tf.unstack(inputs, axis=-1)
            inputs_list[0] = tf.keras.activations.tanh(inputs_list[0])
            inputs_list[1] = tf.keras.activations.tanh(inputs_list[1])
            inputs_list[2] = tf.keras.activations.sigmoid(inputs_list[2])  # last is classifier

            inputs_list[3] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[3])
            inputs_list[4] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[4])
            #todo: add input vec for intensity
            inputs_list[5] = 0.0001+tf.keras.activations.sigmoid(inputs_list[5])
            inputs_list[6] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[6])

            return tf.stack(inputs_list, axis=-1)



        self.activation = tf.keras.layers.Lambda(sigmoid_activation)

    def __call__(self, inputs, training=False):
        x = inputs
        x = x / (tf.reduce_max(tf.abs(x)))

        #todo: normalize input on max stack intensity
        x, cs1 = self.inception1(x, training=training)
        x = self.batch_norm(x)
        x, cs2 = self.inception2(x, training=training)
        x = self.batch_norm2(x)

        for layer in self.horizontal_path:
            x = layer(x)
        #todo: transpose

        x = self.activation(x)
        if training:
            return x,[cs1]#todo: bacck to cs_out2
        else:
            return x
    @property
    def mat(self):
        return self.inception1.cs.mat

    @property
    def sigma(self):
        if self.inception1.cs.sigma != self.inception2.cs.sigma:
           raise ValueError("sigma has to be identical in both layers")
        return self.inception1.cs.sigma

    @sigma.setter
    def sigma(self, value):
        self.inception1.cs.sigma = value
        self.inception2.cs.sigma = value



    def compute_loss_decode_ncs(self, truth,predict):

        l2 = tf.keras.losses.MeanSquaredError()#todo: switched to l1
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1


        #todo: loss for cs part...
        loss = 0


        x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
        X, Y = tf.meshgrid(x, x)
        L2 = tf.constant(0.0)
        count = tf.zeros(tf.shape(truth)[0])
        for j in range(10):
            count += truth[:,j,2]
        for i in range(10):#up to 10 localisations

            L2 += tf.reduce_sum(truth[:,i,2]/(count+0.001)*(-tf.math.log(self.ReduceSum(
                (predict[:, :, :, 2]+0.0001) /(self.ReduceSumKD(predict[:, :, :, 2]+0.0001)
                                     *
                                  tf.math.sqrt(predict[:,:, :, 3]*predict[:,:, :, 4]*predict[:,:, :, 6]
                                               *(2 * tf.constant(m.pi))**2)
                                               )
            *tf.math.exp(-1/2*(
                                tf.square(
                                        predict[:,:, :, 0] - (truth[:,i:i+1,0:1]-Y))  # todo: test that this gives expected values
                                         / (predict[:,:, :, 3])#Ydirection
                                         + tf.square(predict[:,:, :, 1] - (truth[:,i:i+1,1:2]-X))
                                         / (predict[:, :, :, 4])#Xdirection
                                            +tf.square(predict[:,:, :, 5] - truth[:,i:i+1,3:4])/predict[:,:, :, 6]
                                        #Photon count
                                             ))
            ))) # todo: activation >= 0
                                        ,
                            )
        #L2+= 1000*ce(predict[:, :, :, 2],truth_i[:,:,:,2])
        sigma_c = self.ReduceSum((predict[:,:, :, 2]+0.0001) * (1 - predict[:, :,:, 2]))
        #i=0#todo: learn background and photon count
        t = self.ReduceSum(predict[:, :, :, 2])
        c_loss = tf.reduce_sum(1/2*tf.square(t-count)/sigma_c-tf.math.log(tf.sqrt(2*m.pi*sigma_c)))
        #b_loss = l2(data[:,:,:,1]-noiseless_gt[:,:,:,1],predict[:,:,:,5])
        return  L2+c_loss#+ b_loss

    def compute_loss_decode(self, truth,predict, noiseless_gt, cs_out, data):

        l2 = tf.keras.losses.MeanSquaredError()#todo: switched to l1
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1


        #todo: loss for cs part...
        loss = tf.constant(0.0)
        for cs_slice in cs_out:
            inp_cs = tf.unstack(cs_slice, axis=-1)#todo: this needs to be an input
            inp_ns = tf.unstack(noiseless_gt, axis=-1)#todo: this needs to be an input
            for i,j in zip(inp_cs, inp_ns):
                res = tf.linalg.matvec(tf.transpose(self.inception1.cs.mat), tf.keras.layers.Reshape((5184,), )(i), )
                val = tf.keras.layers.Reshape((9,9), )(res/(0.001+tf.reduce_max(tf.abs(res),axis=[1],keepdims=True)))
                ns =(j/(0.001+tf.reduce_max(tf.abs(j),axis=[1],keepdims=True)))
                loss += 300*tf.reduce_mean(tf.square(val-ns))
                loss += 1500*tf.abs(tf.reduce_mean(i))#sparsity constraint

        x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
        X, Y = tf.meshgrid(x, x)
        L2 = tf.constant(0.0)
        count = tf.zeros(tf.shape(truth)[0])
        for j in range(10):
            count += truth[:,j,2]
        for i in range(10):#up to 10 localisations

            L2 += tf.reduce_sum(truth[:,i,2]/(count+0.001)*(-tf.math.log(self.ReduceSum(
                predict[:, :, :, 2] /(self.ReduceSumKD(predict[:, :, :, 2])
                                     *
                                      tf.math.sqrt(predict[:, :, :, 3] * predict[:, :, :, 4] * predict[:, :, :, 6]
                                                   * (2 * tf.constant(m.pi)) ** 3)
                                      )

            *tf.math.exp(-1/2*(
                                tf.square(
                                        predict[:,:, :, 0] - (truth[:,i:i+1,0:1]-Y))  # todo: test that this gives expected values
                                         / (predict[:,:, :, 3])
                                         + tf.square(predict[:,:, :, 1] - (truth[:,i:i+1,1:2]-X))
                                         / (predict[:, :, :, 4])
                                        + tf.square(predict[:, :, :, 5] - truth[:, i:i + 1, 3:4]) / predict[:, :, :, 6]
                                             ))
            ))) # todo: activation >= 0
                                        ,
                            )
        #L2+= 1000*ce(predict[:, :, :, 2],truth_i[:,:,:,2])
        sigma_c = self.ReduceSum(predict[:,:,:,2] * (1 - predict[:,:,:,2]))
        #print(self.ReduceSum(predict[:, :, :, 2]))
        #i=0#todo: learn background and photon count
        c_loss = tf.reduce_sum(1/2*tf.square(self.ReduceSum(predict[:, :, :, 2])-count)/sigma_c-tf.math.log(tf.sqrt(2*m.pi*sigma_c)))
        b_loss = l2(data[:,:,:,1]-noiseless_gt[:,:,:,1], predict[:,:,:,7])
        return  L2+c_loss+ loss + b_loss


class CompressedSensingCVNet(tf.keras.Model):
    TYPE = 0

    def __init__(self):
        super(CompressedSensingCVNet, self).__init__()
        self.cs_layer = CompressedSensing(iterations=100)
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), input_shape=(72*72,3) )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        #todo: keep part of the down path
        self.down_path = [downsample(64,3,apply_batchnorm=True),
        downsample(128,3),
        downsample(256,3,apply_batchnorm=True),
                          ]
        #todo: concatnate here

        #todo: horizontal path

        self.horizontal_path = [
            tf.keras.layers.Conv2D(256, (1, 1), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(128, (7, 1), activation=None, padding="same"),
            tf.keras.layers.Conv2D(64, (1, 7), activation=None, padding="same"),
            #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3,3), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(8, (3, 3), activation=None, padding="same"),

        ]

        def sigmoid_activation(inputs):#softplus activation
            inputs_list = tf.unstack(inputs, axis=-1)
            inputs_list[0] = tf.keras.activations.tanh(inputs_list[0])
            inputs_list[1] = tf.keras.activations.tanh(inputs_list[1])
            inputs_list[2] = tf.keras.activations.sigmoid(inputs_list[2])  # last is classifier

            inputs_list[3] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[3])
            inputs_list[4] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[4])
            #todo: add input vec for intensity
            inputs_list[5] = 0.0001+tf.keras.activations.sigmoid(inputs_list[5])
            inputs_list[6] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[6])

            return tf.stack(inputs_list, axis=-1)

        self.activation = tf.keras.layers.Lambda(sigmoid_activation)
        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, inputs, training=False):
        x = inputs/(0.001+tf.reduce_max(tf.abs(inputs),axis=[1,2],keepdims=True))
        cs = self.cs_layer(x, 0.03)
        x = self.reshape(cs)#72x72
        x = self.batch_norm(x)
        for layer in self.down_path:
            x = layer(x)



        x = self.concat([x, inputs])
        for layer in self.horizontal_path:
            x = layer(x)

        x = self.activation(x)
        if training:
            return x,[cs]#todo: bacck to cs_out2
        else:
            return x

    @property
    def mat(self):
        return self.cs_layer.mat

    @property
    def sigma(self):
        return self.cs_layer.sigma

    @sigma.setter
    def sigma(self, value):
        self.cs_layer.sigma = value

