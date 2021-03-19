from src.custom_layers.cs_layers import CompressedSensing
#from tensorflow.keras.layers import *
import tensorflow as tf
import tensorflow_addons as tfa
from src.custom_layers.utility_layer import downsample,upsample

#todo: imitate yolo architecture and use 9x9 output grid
class CompressedSensingCVNet(tf.keras.Model):
    def __init__(self):
        super(CompressedSensingCVNet, self).__init__()
        self.cs_layer = CompressedSensing()
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), )
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
            tf.keras.layers.Conv2D(128, (1, 1), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(64, (7, 7), activation=None, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3,3), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(3, (3, 3), activation=None, padding="same"),

        ]


        #todo: activation might not be that bad
        def activation(inputs):
            inputs_list = tf.unstack(inputs,axis=-1)
            inputs_list[2] = tf.keras.activations.softmax(inputs_list[2], axis=[-2,-1])#last is classifier
            return tf.stack(inputs_list, axis=-1)
        self.activation = tf.keras.layers.Lambda(activation)
        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, inputs):
        x = self.cs_layer(inputs)
        x = self.reshape(x)#72x72
        x = self.batch_norm(x)
        for layer in self.down_path:
            x = layer(x)



        x = tf.keras.layers.Concatenate()([x, inputs])
        for layer in self.horizontal_path:
            x = layer(x)
        #todo: transpose

        x = self.activation(x)
        return x
        #todo: output 3 x,y,classifier

    def update(self, sigma, px_size):
        self.cs_layer.sigma = sigma

    def compute_loss(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        ce = tf.keras.losses.CategoricalCrossentropy()
        mask = tf.keras.activations.sigmoid(truth[:,:,:,2:3])
        L2 = l2(predict[:,:,:,0:2]*mask, truth[:,:,:,0:2])
        BCE = ce( truth[:,:,:,2], predict[:,:,:,2],)
        return BCE + 3*L2
        #todo: build truth image and compute loss
        #todo: select loc compute coordinates
        #todo: compute loss...



class CompressedSensingNet(tf.keras.Model):
    def __init__(self):
        super(CompressedSensingNet, self).__init__()
        self.cs_layer = CompressedSensing()
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), )

        # self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same")
        # self.pooling_3 = tf.keras.layers.MaxPooling2D((3, 3))
        # self.pooling_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # self.batch_norm2 = tf.keras.layers.BatchNormalization()
        # self.batch_norm3 = tf.keras.layers.BatchNormalization()
        #
        #
        # self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")
        # self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")
        # self.conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same")

        self.conv1 = downsample(32,3,apply_batchnorm=True)
        self.conv2 = downsample(64,3)
        self.conv3 = downsample(128,3,apply_batchnorm=True)
        self.conv4 = downsample(256,3,apply_batchnorm=True)


        self.dense1 = tf.keras.layers.Dense(64, activation="relu")#todo: add l1 regularization
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")#todo: add l1 regularization
        self.dense3 = tf.keras.layers.Dense(16, activation="relu")#todo: add l1 regularization
        self.dense4 = tf.keras.layers.Dense(9)

        def activation(tensor):
            x = tf.unstack(tensor, axis=-1)
            for i in range(3):
                x[6+i] = tf.keras.activations.sigmoid(x[6+i])
            return tf.stack(x, axis=-1)
        self.activation = tf.keras.layers.Lambda(activation)

    def update(self, sigma, px_size):
        self.cs_layer.sigma = sigma
        #self.cs_layer.px_size = px_size todo: not yet needed

    def __call__(self, inputs):

        x = self.cs_layer(inputs)
        x = self.reshape(x)#72x72
        x = self.batch_norm(x)
        x = self.conv1(x)
        #x = self.pooling_2(x)#36x36
        x = self.conv2(x)
        #x = self.pooling_2(x)#18x18
        #x = self.batch_norm2(x)
        x = self.conv3(x)
        #x = self.pooling_2(x)#9x9
        x = tf.keras.layers.concatenate([x,inputs],axis=-1)
        #x = self.dropout(x)
        x = self.conv4(x)
        #x = self.pooling_3(x)
        #x = self.batch_norm3(x)
        x = tf.keras.layers.Flatten()(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        #todo: sigmoid activation for dense(6:)
        x = self.activation(x)


        return x

    def compute_loss(self, truth_p, predict_p, truth_c, predict_c):
        l2 = tf.keras.losses.MeanSquaredError()
        ce = tfa.losses.SigmoidFocalCrossEntropy()
        RMSE = l2(truth_p, predict_p)#*tf.repeat(truth_c, repeats=[2,2,2],axis=1))
        BCE = tf.reduce_sum(ce(truth_c, predict_c,))
        return 3*RMSE+BCE

    def sort(self, tensor):
        x = tf.unstack(tensor, axis=-1)
        squ = []
        for i in range(len(x)//2):
            i*=2
            squ.append(x[i]**2+x[i+1]**2)
        new = tf.stack(squ, axis=-1)
        return tf.argsort(new, axis=-1, direction='ASCENDING', stable=False, name=None)

    def permute_tensor_structure(self, tensor, indices):
        c = indices+6
        x = indices*2
        y = indices*2+1
        v = tf.stack([x,y],axis=-1)
        v = tf.reshape(v, [indices.shape[0],-1])
        perm = tf.concat([v,c],axis=-1)
        return tf.gather(tensor, perm, batch_dims=1, name=None, axis=-1)

    def compute_permute_loss(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        ce = tfa.losses.SigmoidFocalCrossEntropy()

        indices = self.sort(predict[:,0:6])
        indices2 = self.sort(truth[:,0:6])
        predict = self.permute_tensor_structure(predict, indices)
        truth = self.permute_tensor_structure(truth, indices2)
        mask = tf.repeat(truth[:,6:], repeats=[2,2,2],axis=1)
        L2 = l2(predict[:,0:6]*mask, truth[:,0:6])
        BCE = tf.reduce_sum(ce(truth[:,6:], predict[:,6:],))
        return 3*L2+BCE
        #todo: test this

    def compute_alternative_permute_loss(self, truth, predict):
        #todo: compute a per tensor and classifier
        for i in range(3):
            tensor = tf.gather(predict, [2*i,2*i+1,i+6])
            for j in range(3):
                truth = tf.gather(truth, [2*i,2*i+1,i+6])
        #todo: compute for least loss
