import tensorflow as tf
from src.factory import Factory
from src.custom_layers.cs_layers import CompressedSensing, CompressedSensingInception
from src.utility import get_root_path
from astropy.convolution import Gaussian2DKernel
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet,CompressedSensingInceptionNet



# #todo: functional_test
def create(im_shape):
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)
    ph = np.random.randint(5000, 30000)
    points = factory.create_crop_point_set(photons=ph)
    sigma_x = 150
    sigma_y = 150
    factory.kernel = (sigma_x, sigma_y)


    ind = np.random.randint(0, points.shape[0])
    n = 2  # np.random.poisson(1.7)
    image = factory.create_image()
    image = factory.create_points_add_photons(image, points[ind:ind + n], points[ind:ind + n, 2])
    image = factory.reduce_size(image)
    image = factory.accurate_noise_simulations_camera(image)
    return image
#
# def test_output():
#     #todo: useful for debugging might be outsourced to display?
#     #done input bigger tensor
#     crop = create(9)
#     crop_new = np.zeros((crop.shape[0], crop.shape[1],3))
#     for i in range(3):
#         crop_new[:,:,i] = crop
#     crop = crop_new.astype(np.float32)
#     crop/=crop.max()
#
#     #crop = np.load(os.getcwd() + r"\crop.npy")
#     layer = CompressedSensing()
#     crop_tensor = tf.constant((crop),dtype=tf.float64)
#     im = tf.stack([crop_tensor, crop_tensor])
#     y = layer(im)
#     x = layer(im)
#     fig,axs = plt.subplots(3)
#     y = tf.reshape(y, (-1, 73,73,3))
#     x = tf.reshape(x, (-1, 73,73,3))
#
#     c_spline = interpolate.interp2d(np.arange(0,9,1), np.arange(0,9,1), crop[:,:,1], kind='cubic')
#
#     new = c_spline(np.arange(0,9,0.125),np.arange(0,9,0.125))
#
#     axs[0].imshow(x[0,:,:,1])
#     axs[1].imshow(y[0,:,:,1])
#     axs[2].imshow(new)
#     plt.show()
#     x=0
    #done: load file
    #done: run layer
class ViewLayerOutputs():
    def __init__(self):
        path = get_root_path() + r"\trainings\cs_inception\_new_EST_lammu"
        self.network = CompressedSensingInceptionNet()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.network.sigma = 150
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer , net=self.network)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=6)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def cs_inception_path_output(self):
        crop = create(9)
        crop_new = np.zeros((crop.shape[0], crop.shape[1], 3))
        for i in range(3):
            crop_new[:, :, i] = crop
        crop = crop_new.astype(np.float32)
        crop /= crop.max()
        crop_tensor = tf.constant((crop), dtype=tf.float64)
        im = tf.stack([crop_tensor, crop_tensor])
        out,cs_out = self.network.inception1(im)
#        print(self.network.inception1.cs.lam)
        test = tf.reshape(tf.linalg.matvec(tf.transpose(self.network.inception1.cs.mat),tf.reshape(cs_out[0,:,:,1],5184), ),(9,9))
        plt.imshow(test/tf.reduce_max(test,axis=[0,1], keepdims=True))
        plt.show()

        fig, axs = plt.subplots(3,4)

        axs[0][0].imshow(cs_out[0,:,:,1])
        axs[0][1].imshow(out[0,:,:,1])
        axs[0][2].imshow(out[0,:,:,2])
        axs[0][3].imshow(out[0,:,:,3])

        axs[1][0].imshow(out[0,:,:,4])
        axs[1][1].imshow(out[0,:,:,5])
        axs[1][2].imshow(out[0,:,:,6])
        # axs[1][3].imshow(out[0,:,:,7])
        #
        # axs[2][0].imshow(out[0,:,:,8])
        # axs[2][1].imshow(out[0,:,:,9])
        axs[2][2].imshow(tf.reshape(test,(9,9)))

        axs[2][3].imshow(im[0,:,:,1])
        plt.show()


if __name__ == '__main__':
    V = ViewLayerOutputs()
    V.cs_inception_path_output()