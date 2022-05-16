from src.custom_layers.cs_layers import CompressedSensing, CompressedSensingInception, CompressedSensingConvolutional, UNetLayer
#from tensorflow.keras.layers import *
import tensorflow as tf
from src.custom_layers.utility_layer import downsample,upsample
#import tensorflow_probability as tfp
import math as m
from src.models.loss_functions import compute_loss_decode

class BaseModel(tf.keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        def sigmoid_activation(inputs):#softplus activation
            inputs_list = tf.unstack(inputs, axis=-1)
            inputs_list[0] = tf.keras.activations.tanh(inputs_list[0])
            inputs_list[1] = tf.keras.activations.tanh(inputs_list[1])
            inputs_list[2] = tf.keras.activations.sigmoid(inputs_list[2])  # last is classifier

            inputs_list[3] = 0.01+3*tf.keras.activations.sigmoid(inputs_list[3])
            inputs_list[4] = 0.01+3*tf.keras.activations.sigmoid(inputs_list[4])
            #todo: add input vec for intensity
            inputs_list[5] = 0.0001+tf.keras.activations.sigmoid(inputs_list[5])
            inputs_list[6] = 0.001+3*tf.keras.activations.sigmoid(inputs_list[6])

            return tf.stack(inputs_list, axis=-1)
        self.activation = tf.keras.layers.Lambda(sigmoid_activation)


class CompressedSensingInceptionNet(BaseModel):
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



    def __call__(self, inputs, training=False):
        x = inputs
        x = x / (tf.reduce_max(tf.abs(x)))

        x, cs1 = self.inception1(x, training=training)
        x = self.batch_norm(x)
        x, cs2 = self.inception2(x, training=training)
        x = self.batch_norm2(x)

        for layer in self.horizontal_path:
            x = layer(x)
        #todo: transpose

        x = self.activation(x)
        if training:
            return x#,[cs1]
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


class CompressedSensingCVNet(BaseModel):
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
            return x#,[cs]#todo: bacck to cs_out2
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


class CompressedSensingUNet(BaseModel):
    TYPE = 0
    OUTPUT_CHANNELS = 8
    def __init__(self):
        super(CompressedSensingUNet, self).__init__()
        self.cs_layer = CompressedSensing(iterations=1)
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), input_shape=(72*72,3) )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        initializer = tf.random_normal_initializer(0., 0.02)
        #todo: keep part of the down path
        self.down_path = [downsample(2,3,apply_batchnorm=False),#36
        downsample(4,3),#18
        downsample(8,3),#9
        downsample(64,5, strides=3),#3
        downsample(128,5,apply_batchnorm=False,strides=3),#1
                          ]
        self.up_path = [upsample(64, 4, apply_dropout=False, strides=3),
                        tf.keras.layers.Conv2DTranspose(8, 4,
                                                        strides=3,
                                                        padding='same',
                                                        kernel_initializer=initializer,)
]

        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, inputs, training=False):
        x = inputs/(0.001+tf.reduce_max(tf.abs(inputs),axis=[1,2],keepdims=True))
        cs =self.cs_layer(x, 0.03)
        x = self.reshape(cs)#72x72
        skip = []
        x = self.down_path[0](x)
        skip.append(x)
        x = self.down_path[1](x)
        skip.append(x)
        x = self.down_path[2](x)
        x = tf.keras.layers.Concatenate()([x, inputs])
        skip.append(x)

        #todo: concat input here
        x = self.down_path[3](x)
        skip.append(x)
        x = self.down_path[4](x)

        x = self.up_path[0](x)
        x = tf.keras.layers.Concatenate()([x, skip[-1]])
        x = self.up_path[1](x)

        x = self.activation(x)
        if training:
            return x#,[cs]#todo: bacck to cs_out2
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

class CompressedSensingResUNet(BaseModel):
    TYPE = 0
    OUTPUT_CHANNELS = 8
    def __init__(self):
        super(CompressedSensingResUNet, self).__init__()
        self.initial_layer = UNetLayer(8)

        self.decoder = UNetLayer(1)
        self.update_encoder = UNetLayer(8)



    def __call__(self, inputs, training=False):
        x = self.initial_layer(inputs)
        x_zero = self.activation(x)
        x = x_zero
        reconstruction_delta_list = []
        for i in range(4):
            y = self.decoder(x)
            y += x[:,:,:,7:8]
            y -= inputs[:,:,:,1:2]
            reconstruction_delta_list.append(y)
            update = self.update_encoder(y)
            update = tf.keras.activations.tanh(update)
            x += update
        x = self.activation(x) #todo not compute this every time!
        if training:
            return x#,reconstruction_delta_list #,[cs]#todo: bacck to cs_out2
        else:
            return x

    def compute_loss(self,feature_map,reconstruction_delta_list, truth, noiseless_gt, data):
        loss = compute_loss_decode(truth, feature_map, noiseless_gt, data)
        # for feature_map in reconstruction_delta_list:
        #     loss += tf.reduce_sum(tf.square(feature_map))
        return loss


    @property
    def mat(self):
        return self._mat

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value


class StandardUNet(BaseModel):
    TYPE = 0
    OUTPUT_CHANNELS = 8
    def __init__(self):
        super(StandardUNet, self).__init__()
        self.initial_layer = UNetLayer(8)
        self.layer2 = UNetLayer(8)


    def __call__(self, inputs, training=False):
        x = self.initial_layer(inputs)
        x = self.layer2(x)
        x = self.activation(x)

        if training:
            return x#,[cs]#todo: bacck to cs_out2
        else:
            return x

    @property
    def mat(self):
        return self._mat

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

class CompressedSensingConvNet(BaseModel):
    TYPE = 0
    OUTPUT_CHANNELS = 8
    def __init__(self):
        super(CompressedSensingConvNet, self).__init__()
        self.cs_layer = CompressedSensingConvolutional(iterations=5)


    def __call__(self, inputs, training=False):
        x = inputs#/(0.001+tf.reduce_max(tf.abs(inputs),axis=[1,2],keepdims=True))
        cs = tf.zeros((tf.shape(inputs)[0], 72*72, 3))
        x =self.cs_layer(x, 0.03)

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
