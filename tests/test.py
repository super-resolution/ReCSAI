import tensorflow as tf
from src.factory import Factory
from astropy.convolution import Gaussian2DKernel
import numpy as np

class BaseTest(tf.test.TestCase):
    def create_noiseless_random_data_crop(self, im_shape, sigma, px_size, batch_size=3):
        factory = Factory()
        factory.shape = (im_shape * px_size, im_shape * px_size)
        factory.image_shape = (im_shape, im_shape)
        ph = np.random.randint(5000, 30000)
        points = factory.create_crop_point_set(photons=ph)
        sigma_x = sigma
        sigma_y = sigma
        factory.kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)
        ind = np.random.randint(0, points.shape[0])
        n = 2  # np.random.poisson(1.7)
        image = factory.create_image()
        image = factory.create_points_add_photons(image, points[ind:ind + n], points[ind:ind + n, 2])
        image = factory.reduce_size(image)
        tf_image = tf.constant(factory.accurate_noise_simulations_camera(image), dtype=tf.float64)
        tf_stack = tf.expand_dims(tf.stack([tf_image,]*batch_size, axis=-1),axis=0)#list multiplied should repeat
        tf_stack /= tf.keras.backend.max(tf_stack)
        return tf_stack
