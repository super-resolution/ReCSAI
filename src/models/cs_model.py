from ..custom_layers import *
from tensorflow.keras.layers import *

class CompressedSensingNet(tf.keras.Model):
    def __init__(self):
        super(CompressedSensingNet, self).__init__()
        self.cs_layer = CompressedSensing()
        self.reshape = tf.keras.layers.Reshape((73, 73, 3), )

        self.conv1 = Conv2D(32, (3, 3), activation='relu', padding="same")
        self.pooling_3 = tf.keras.layers.MaxPooling2D((3, 3))
        self.pooling_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()


        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")
        self.conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same")


        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.dense3 = tf.keras.layers.Dense(16, activation="relu")
        self.dense4 = tf.keras.layers.Dense(9)
        self.preparation = tf.keras.layers.Lambda(lambda x: (x-tf.keras.backend.min(x))/tf.keras.backend.max(x-tf.keras.backend.min(x)))


    def update(self, sigma, px_size):
        self.cs_layer.update_psf(sigma, px_size)

    def __call__(self, inputs):

        x = self.cs_layer(inputs)
        x = self.reshape(x)#72x72
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.pooling_2(x)#36x36
        x = self.conv2(x)
        x = self.pooling_2(x)#18x18
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.pooling_2(x)#9x9
        x = tf.keras.layers.concatenate([x,inputs],axis=-1)
        x = self.conv4(x)
        x = self.pooling_3(x)
        x = self.batch_norm3(x)
        x = tf.keras.layers.Flatten()(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)


        return x

def compute_cs_loss(truth_p, predict_p, truth_c, predict_c):
    l2 = tf.keras.losses.MeanSquaredError()
    ce = tf.keras.losses.CategoricalCrossentropy()
    RMSE = l2(truth_p, predict_p)#*tf.repeat(truth_c, repeats=[2,2,2],axis=1)
    BCE = ce(truth_c, predict_c,)
    return 5*RMSE+BCE