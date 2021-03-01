from src.custom_layers import *
from src.utility import *
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from src.factory import Factory
import tensorflow as tf
from src.custom_layers.cs_layers import CompressedSensingInception, CompressedSensing
from unittest import skip

#done: unit test should extend tf test case
class TestCompressedSensingLayer(tf.test.TestCase):
    def setUp(self):
        self.layer = CompressedSensing()

        #create random test_data
    def create_random_data_crop(self, im_shape, sigma, px_size, batch_size=1):
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
        tf_stack = tf.expand_dims(tf.stack([tf_image, tf_image, tf_image], axis=-1),axis=0)
        tf_stack /= tf.keras.backend.max(tf_stack)
        return tf_stack#todo return tensor


    def test_output_is_sparse(self):
        #cropsize = 9; sigma= 150; px_size= 100
        data = self.create_random_data_crop(9, sigma=150, px_size=100)

        self.layer.update_psf(sigma=150, px_size=100)
        output = self.layer(data)
        output = tf.reshape(output, (-1, 73, 73, 3))
        #not all pixels are 0
        self.assertNotAllEqual(output, tf.zeros_like(output), msg="All entries of output equal zero")
        #numbers of pixels ==0 > numbers of pixels !=0
        self.assertLess(tf.where(output>=0.01).shape[0],tf.where(output<0.01).shape[0], msg="Result is not sparse" )

    @skip
    def test_psf_matrix(self):
        #todo: how do I test the psf matrix??
        self.fail()

    @skip
    def test_lambda(self):
        #todo: properties for cs layer...
        self.fail()

    def test_different_iterations_have_different_outputs(self):
        data = self.create_random_data_crop(9, sigma=150, px_size=100)
        self.layer.set_iteration_count(5)
        output1 = self.layer(data)
        self.layer.set_iteration_count(100)
        output2 = self.layer(data)
        self.assertNotAllClose(output1, output2, msg="Different iterations yiel the same output...")

class TestCompressedSensingInceptionLayer(tf.test.TestCase):
    def setUp(self):
        self.layer = CompressedSensingInception()

    def test_cs_layer_properties(self):
        self.fail()

    def test_all_branches_produce_nonzero_output(self):
        self.fail()

    def test_shapes_after_each_layer(self):
        self.fail()

    def test_one_one_convolution(self):
        self.fail()

    def test_pooling_layers(self):
        self.fail()

class TestLossFunction(tf.test.TestCase):
    def setUp(self):
        pass

    def test_permutation_is_ok(self):
        self.fail()

    def test_if_coords_permute_classifier_permuts(self):
        self.fail()


def test_matrix():
    mat = create_psf_matrix(9, 8)


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

def test_layer():
    layer = CompressedSensing()
    im = tf.constant(np.zeros((9,9)),dtype=tf.float64)
    x = layer(im)
    y=0

if __name__ == '__main__':
    test_output()