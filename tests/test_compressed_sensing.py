import tensorflow as tf
from src.custom_layers.cs_layers import CompressedSensingInception, CompressedSensing
from unittest import skip
from tests.test import BaseTest
import matplotlib.pyplot as plt



#done: unit test should extend tf test case
class TestCompressedSensingLayer(BaseTest):

    def setUp(self):
        self.layer = CompressedSensing()

        #create random test_data



    def test_output_is_sparse(self):
        #cropsize = 9; sigma= 150; px_size= 100
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)

        self.layer.sigma =150
        output = self.layer(data)
        output = tf.reshape(output, (-1, 73, 73, 3))
        #not all pixels are 0
        self.assertNotAllEqual(output, tf.zeros_like(output), msg="All entries of output equal zero")
        #numbers of pixels ==0 > numbers of pixels !=0
        self.assertLess(tf.where(output>=0.01).shape[0],tf.where(output<0.01).shape[0], msg="Result is not sparse" )

    def test_different_iterations_have_different_outputs(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        self.layer.iterations = 5
        output1 = self.layer(data)
        self.layer.iterations = 100
        output2 = self.layer(data)
        self.assertNotAllClose(output1, output2, msg="Different iterations yiel the same output...")

    def test_perfect_reconstruction_from_noiseless_data(self):
        #todo: simulate perfect psf use cs high iteration receive sparse result
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        self.layer.iterations = 10000#todo: 4000 is close to sparse
        output = self.layer(data)
        output = tf.reshape(output, (1,72,72,3))
        plt.imshow(output[0,:,:,1])
        plt.show()
        self.assertGreater(tf.where(output>0).shape[0], 50)

    @skip
    def test_different_sigma_have_different_outputs(self):
        #todo: how do I test the psf matrix??
        #todo: rewrite psf matrix creater to output the same as old...
        self.fail()

    @skip
    def test_lambda(self):
        #todo: properties for cs layer...
        self.fail()





class TestCompressedSensingInceptionLayer(BaseTest):
    def setUp(self):
        self.layer = CompressedSensingInception()

    def test_w_output_has_expected_shape_(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        output = self.layer.convolution1x1_w1(data)
        self.assertListEqual(list(output.shape), [data.shape[0],data.shape[1], data.shape[2], 1], msg="path w has unexpected output shape")

    def test_x_path_has_expected_shape(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        layers = self.layer.x_path
        x = data
        for layer in layers:
            x = layer(x)
        self.assertListEqual(list(x.shape), [data.shape[0],data.shape[1], data.shape[2], 1], msg="path x has unexpected output shape")

    def test_y_path_has_expected_shape(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        layers = self.layer.y_path
        y = data
        for layer in layers:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],data.shape[1], data.shape[2], 1], msg="path y has unexpected output shape")

    def test_z_path_has_expected_shape(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        z = data
        skips = []
        for layer in self.layer.z_path_down:
            z = layer(z)
            skips.append(z)

        skip = skips[0]

        z = self.layer.up1(z)
        z = self.layer.concat([z,skip])
        z = self.layer.up2(z)

        self.assertListEqual(list(z.shape), [data.shape[0],data.shape[1], data.shape[2], 1], msg="path z has unexpected output shape")


    @skip
    def test_cs_layer_properties(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        output = self.layer(data)
        for i in range(output.shape[0]):
            self.assertAllClose(output[i])
        #todo: can CompressedSensing Properties be updated?
        #todo: i.e. iteration count, psf, lambda, mu
        self.fail()

    def test_all_branches_produce_nonzero_output(self):
        data = self.create_noiseless_random_data_crop(9, sigma=150, px_size=100)
        output = self.layer(data)
        # for i in range(output.shape[-1]):
        #     self.assertAllClose(output[:,:,:,i], )
        self.assertListEqual(list(output.shape), [data.shape[0],data.shape[1], data.shape[2], 4], msg="layer has unexpected output shape")



    def test_u_net_has_skip_layers(self):
        self.fail()

    def test_shapes_after_layer(self):
        #todo: acess sublayers and check shape is as expected
        self.fail()


class TestLossFunction(tf.test.TestCase):
    def setUp(self):
        pass









