import tensorflow as tf
from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet, CompressedSensingInceptionNet
from unittest import skip
from tests.test import BaseTest
import copy

class TestCompressedSensingNet(BaseTest):
    def setUp(self):
        self.network = CompressedSensingNet()

    def permute_tensor_structure(self, tensor, indices):
        c = indices+6
        x = indices*2
        y = indices*2+1
        v = tf.transpose([x,y])
        v = tf.reshape(v, [indices.shape[0],-1])
        perm = tf.concat([v,c],axis=-1)
        return tf.gather(tensor, perm, batch_dims=1, name=None, axis=-1)


    def test_update_psf(self):
        self.network.cs_layer.sigma = 150
        initial_psf = copy.deepcopy(self.network.cs_layer.mat.numpy())
        self.network.update(180, 100)
        new_psf = self.network.cs_layer.mat.numpy()
        self.assertNotAllClose(initial_psf, new_psf, msg="PSF update doesnt work")

    def test_network_learns_loss_function(self):
        #todo: apply l1 regularizer on cs path?
        #todo: loss function for cs path
        #todo: penaltize zeros and apply dropout in this layer?
        self.fail()

    def test_cs_layer_output_not_zero(self):
        self.fail()

    def test_loss_is_legit(self):
        self.fail()

    def test_permutation_loss(self):
        #todo: test multiple permutations
        tensor1 = tf.constant([[30,20,15,7,0,0,1,1,0],[13,5,0,0,15,10,1,0,1]],dtype=tf.float32)#todo: permute tensor and receive loss of zero?
        tensor2 = self.network.permute_tensor_structure(tensor1, tf.constant([[2,1,0],[0,1,2]]))
        loss = self.network.compute_permute_loss(tensor1, tensor2)
        self.assertEqual(loss,0, msg="loss for only permuted tensors is not zero")
        tensor3 = tf.constant([[29, 22, 15, 7, 0, 0, 1, 1, 0],[13,5,0,0,15,10,1,0,1] ], dtype=tf.float32)
        loss = self.network.compute_permute_loss(tensor3, tensor2)
        self.assertAllClose(loss, (1.0+4.0)/2.0, msg="loss for values is not rmse")

        #todo: permute classifier and tensor together

        #todo: it should be neglectable wether the position is reconstructed in vector 1 2 or 3


class TestCsInceptionNet(BaseTest):
    def setUp(self):
        self.network = CompressedSensingInceptionNet()

    def test_sigma_updates_in_both_cs_layers(self):
        self.network.sigma = 150
        initial_psf = copy.deepcopy(self.network.inception1.cs.mat.numpy())
        self.network.sigma = 180
        new_psf = self.network.inception1.cs.mat.numpy()
        self.assertNotAllClose(initial_psf, new_psf, msg="PSF update doesnt work")
        new_psf2 = self.network.inception2.cs.mat.numpy()
        self.assertAllClose(new_psf, new_psf2)



class TestCsUNet(BaseTest):
    def setUp(self):
        self.network = CompressedSensingCVNet()

    def test_shapes(self):
        data = self.create_noiseless_random_data_crop(9, 150, 100)
        y = data
        for layer in self.network.down_path:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],1, 1, 128], msg="down path y has unexpected output shape")
        for layer in self.network.up_path1:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],9, 9, 64], msg="up path 1 y has unexpected output shape")
        y = self.network.concat([y,data])
        for layer in self.network.up_path2:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],36, 36, 16], msg="up path 2 y has unexpected output shape")

        y = self.network.last(y)
        y = self.network.activation(y)
        self.assertListEqual(list(y.shape), [data.shape[0],72, 72, 3], msg="last y has unexpected output shape")


class TestDriftCorrectNet(tf.test.TestCase):
    def setUp(self):
        #todo: create test dataset with random on and off time and simulate a changing drift
        pass

    def test_layer_shape(self):
        pass

    def test_output(self):
        pass
