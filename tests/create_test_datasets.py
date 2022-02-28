import tensorflow as tf
from src.factory import Factory
import numpy as np

class TestDatasets():
    def __init__(self, im_shape):
        self.factory = Factory()
        self.factory.shape= (im_shape*100,im_shape*100)
        self.factory.image_shape = (im_shape,im_shape)
        sigma_x = 150
        sigma_y = 150
        self.factory.kernel = (sigma_x, sigma_y)


    def create_random(self):
        ph = np.random.randint(5000, 30000)

        points = self.factory.create_crop_point_set(photons=ph)
        ind = np.random.randint(0, points.shape[0])
        n = 2  # np.random.poisson(1.7)
        image = self.factory.create_image()
        image = self.factory.create_points_add_photons(image, points[ind:ind + n], points[ind:ind + n, 2])
        print(points[ind:ind + n]*73/(10*100))
        image = self.factory.reduce_size(image)
        image = self.factory.accurate_noise_simulations_camera(image)
        #repeat image 3 times
        crop_new = np.zeros((image.shape[0], image.shape[1], 3))
        for i in range(3):
            crop_new[:, :, i] = image
        crop = crop_new.astype(np.float32)
        crop /= crop.max()
        crop_tensor = tf.constant((crop), dtype=tf.float64)
        im = tf.stack([crop_tensor, crop_tensor])
        return im, points[ind:ind + n]

    def create_decreasing_distance(self):
        frames = []
        p = []
        y = self.factory.shape[0]/2
        for i in range(5):
            delta = 200-i*40
            x1 = self.factory.shape[0]/2-delta
            x2 = self.factory.shape[0]/2+delta
            points = np.array([[y,x1],[y,x2]])
            image = self.factory.create_image()
            image = self.factory.create_points_add_photons(image, points, np.array([5000,5000]))
            image = self.factory.reduce_size(image)
            image = self.factory.accurate_noise_simulations_camera(image)

            # repeat image 3 times
            crop_new = np.zeros((image.shape[0], image.shape[1], 3))
            for i in range(3):
                crop_new[:, :, i] = image
            frames.append(crop_new)
            p.append(points)
        im = tf.stack(frames)
        return im,p

    def create_increasing_noise(self):
        frames = []
        p = []
        y = self.factory.shape[0]/2
        for i in range(5):
            delta = 150
            x1 = self.factory.shape[0]/2-delta
            x2 = self.factory.shape[0]/2+delta
            points = np.array([[y,x1],[y,x2]])
            image = self.factory.create_image()
            image = self.factory.create_points_add_photons(image, points, np.array([3000-500*i,3000-500*i]))
            image = self.factory.reduce_size(image)
            image = self.factory.accurate_noise_simulations_camera(image)
            # repeat image 3 times
            crop_new = np.zeros((image.shape[0], image.shape[1], 3))
            for i in range(3):
                crop_new[:, :, i] = image
            frames.append(crop_new)
            p.append(points)
        im = tf.stack(frames)
        return im,p

    def create_decreasing_lifetime(self):
        frames = []
        p = []
        y = self.factory.shape[0]/2
        for i in range(5):
            delta = 200
            lt = 40-6*i
            x1 = self.factory.shape[0]/2-delta
            x2 = self.factory.shape[0]/2+delta
            points = np.array([[y,x1, 5000, lt],[y,x2,5000, lt]])
            image,_,_ = self.factory.simulate_accurate_flimbi([], points, switching_rate=0, inverted=[3,3])
            image = self.factory.reduce_size(image)
            image = self.factory.accurate_noise_simulations_camera(image)
            # repeat image 3 times
            crop_new = np.zeros((image.shape[0], image.shape[1], 3))
            for i in range(3):
                crop_new[:, :, i] = image
            frames.append(crop_new)
            p.append(points)
        im = tf.stack(frames)
        return im,p