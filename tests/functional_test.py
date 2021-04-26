import tensorflow as tf
from src.factory import Factory
from src.custom_layers.cs_layers import CompressedSensing

from astropy.convolution import Gaussian2DKernel
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate

#todo: functional_test
def create(im_shape):
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)
    ph = np.random.randint(5000, 30000)
    points = factory.create_crop_point_set(photons=ph)
    sigma_x = 150
    sigma_y = 150
    factory.kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)


    ind = np.random.randint(0, points.shape[0])
    n = 2  # np.random.poisson(1.7)
    image = factory.create_image()
    image = factory.create_points_add_photons(image, points[ind:ind + n], points[ind:ind + n, 2])
    image = factory.reduce_size(image)
    image = factory.accurate_noise_simulations_camera(image)
    return image

def test_output():
    #todo: useful for debugging might be outsourced to display?
    #done input bigger tensor
    crop = create(9)
    crop_new = np.zeros((crop.shape[0], crop.shape[1],3))
    for i in range(3):
        crop_new[:,:,i] = crop
    crop = crop_new.astype(np.float32)
    crop/=crop.max()

    #crop = np.load(os.getcwd() + r"\crop.npy")
    layer = CompressedSensing()
    crop_tensor = tf.constant((crop),dtype=tf.float64)
    im = tf.stack([crop_tensor, crop_tensor])
    y = layer(im)
    x = layer(im)
    fig,axs = plt.subplots(3)
    y = tf.reshape(y, (-1, 73,73,3))
    x = tf.reshape(x, (-1, 73,73,3))

    c_spline = interpolate.interp2d(np.arange(0,9,1), np.arange(0,9,1), crop[:,:,1], kind='cubic')

    new = c_spline(np.arange(0,9,0.125),np.arange(0,9,0.125))

    axs[0].imshow(x[0,:,:,1])
    axs[1].imshow(y[0,:,:,1])
    axs[2].imshow(new)
    plt.show()
    x=0
    #done: load file
    #done: run layer