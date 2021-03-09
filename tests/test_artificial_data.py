import tensorflow as tf
from src.factory import Factory
import numpy as np
from astropy.convolution import Gaussian2DKernel
from unittest import skip
from src.data import crop_generator


class TestArtificialDataCreation(tf.test.TestCase):

    def setUp(self):
        im_shape = 100
        px_size=100
        sigma = 150
        self.factory = Factory()
        self.factory.shape = (im_shape * px_size, im_shape * px_size)
        self.factory.image_shape = (im_shape, im_shape)
        ph = np.random.randint(5000, 30000)
        #create random point set
        self.points = self.factory.create_crop_point_set(photons=ph)
        sigma_x = sigma
        sigma_y = sigma
        self.factory.kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)

    def test_flip_equals_data_created_for_those_coords(self):
        i = np.random.randint(0,100)
        point = self.points[i:i+1]
        image = self.factory.create_image()
        image = self.factory.create_points_add_photons(image, point, point[:, 2])
        image = self.factory.reduce_size(image)
        image = np.fliplr(image)
        #create flipped ground truth
        point[0,1] = (self.factory.shape[1]-1)-point[0,1]#todo: substract 1 because we start at zero write fitting function for that
        image2 = self.factory.create_image()
        image2 = self.factory.create_points_add_photons(image2, point, point[:, 2])
        image2 = self.factory.reduce_size(image2)
        self.assertAllClose(image,image2)

    def test_only_painted_localisations_are_in_list(self):
        init_indices = np.random.choice(self.points.shape[0], 10)
        on_points = self.points[init_indices]
        switching_rate = 0.2
        #todo: test this in flimbi mode
        image, truth, on_points = self.factory.simulate_accurate_flimbi(self.points, on_points,
                                                                   switching_rate=switching_rate)  # todo: simulate off
        #todo: on_points contain only points painted in image
        for point in on_points:
            point = point.astype(np.int16)
            sub = image[point[0]-450:point[0]+450,point[1]]
            self.assertNotAllClose(sub, np.zeros_like(sub), msg="Flimbi convolve adds points to list which are not painted")

        self.assertGreater(tf.where((truth-image)<0.001).shape[0],0.9*image.flatten().shape[0], msg="image doesn't resemble truth")
        #image = self.factory.reduce_size(image)#do this later?
        #truth = self.factory.reduce_size(truth)
        #todo: truth and image should be close
        self.assertLess(tf.where(truth==0).shape[0], tf.where(image==0).shape[0], msg="PSF isn't disrupted")
        #todo: zero entries in image > zero entries in truth
        del image,truth

    @skip
    def test_noise_is_simulated_as_expected(self):
        self.fail()

    def test_ground_truth_is_sparse(self):
        #test can fail for high density samples
        image = self.factory.create_image()
        image = self.factory.create_points_add_photons(image, self.points[0:10], self.points[0:10, 2])
        self.assertNotAllClose(image, 0, msg="image is zero")
        self.assertLess(tf.where(image>0.01).shape[0], tf.where(image<0.01).shape[0], msg="image not sparse")

    @skip
    def test_on_time_for_flim_data(self):
        self.fail()

class TestCropGenerator(tf.test.TestCase):
    def setUp(self):
        self.generator = crop_generator(9)

    def test_noise_lvl_is_randomized(self):
        data,_ = self.generator().__next__()
        images = []
        set_ = data[0:4]
        images.append(set_[0])
        images.append(tf.image.flip_left_right(set_[1]))
        images.append(tf.image.flip_up_down(set_[2]))
        images.append(tf.image.flip_up_down(tf.image.flip_left_right(set_[3])))
        for i in range(4):
            for j in range(4):
                if i!=j:
                    self.assertNotAllClose(images[j], images[i])
        for i in range(4):
            for j in range(4):
                if i!=j:
                    self.assertAllClose(images[j], images[i], atol=0.1, rtol=100.0)
        #todo: each of 4 consecutive images has different noise
        #todo: undo flip and comapare

    def test_none_of_three_channels_is_empty(self):
        data,_ = self.generator().__next__()
        for i in range(data.shape[0]):
            for j in range(3):
                self.assertAllClose(data[i,:,:,j])

        #todo: check for zeros

    def test_flip_is_working_as_expected(self):
        #todo: images are not equal on noise lvl comparison
        self.fail()