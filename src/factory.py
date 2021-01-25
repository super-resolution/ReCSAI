import numpy as np
from astropy.convolution import Gaussian2DKernel
import cv2
import numba


class Factory():
    def __init__(self):
        self.kernel = Gaussian2DKernel(x_stddev=150, y_stddev=150)
        self.shape = (8192, 8192)
        self.image_shape = (64, 64)
        seed = 42  # switch seed
        self.rs = np.random.RandomState(seed)

    def create_image(self):
        image = np.zeros(self.shape).astype(np.float32)
        return image

    def reduce_size(self, image):
        return cv2.resize(image, self.image_shape, interpolation=cv2.INTER_AREA)*(self.shape[0]/self.image_shape[0])**2

    def create_crop_point_set(self, photons=1200):
        n_points = 100000
        points = np.zeros((n_points, 5)).astype(np.float32)
        distribution = np.random.poisson(photons, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
        for i in range(n_points):
            p = np.random.randint(150, self.shape[0]-150, size=2)
            points[i, 0:2] = p
            points[i, 2] = distribution[i]  # todo: increase lifetime?
            points[i, 3] = np.random.randint(0, int(points[i, 2]))
            points[i, 4] = np.random.randint(-25, 25)
        return points

    def create_point_set(self, on_time=600):
        n_points = 100000
        points = np.zeros((n_points, 5)).astype(np.float32)
        distribution = np.random.poisson(on_time, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
        for i in range(n_points):
            p = np.random.randint(0, self.shape[0], size=2)
            points[i, 0:2] = p
            points[i, 2] = distribution[i]  # todo: increase lifetime?
            points[i, 3] = np.random.randint(0, int(points[i, 2]))
            points[i, 4] = np.random.randint(-25, 25)
        return points

    def accurate_noise_simulations_camera(self, image, quantum_efficiency=0.70, dark_noise=64 / 10,
                                              sensitivity=1, bitdepth=12):
        image = image.astype(np.uint16)

        shot_noise = self.rs.poisson(image, image.shape)

        # Round the result to ensure that we have a discrete number of electrons
        electrons = np.round(quantum_efficiency * shot_noise)

        # electrons dark noise 34 counts per second
        electrons_out = np.round(self.rs.normal(scale=dark_noise, size=electrons.shape) + electrons)

        # ADU/e-
        max_adu = np.int(2 ** bitdepth - 1)
        adu = (electrons_out * sensitivity).astype(np.int)
        adu[adu > max_adu] = max_adu  # models pixel saturation
        adu[adu < 0] = 0  # models pixel saturation
        return adu

    def simulate_accurate_flimbi(self, points, on_points, switching_rate=8.0):
        """input random distributed convolved and resized image"""
        #todo: simulate world in 0.1 ms

        image = self.create_image()
        ground_truth = self.create_image()
        image, ground_truth, on_points = self.concolve_flimbi_style(image, ground_truth, points, on_points, switching_rate=switching_rate)
        return image, ground_truth, on_points

    def create_points_add_photons(self, image, points, photons):
        for i in range(points.shape[0]):
            image[int(points[i,0]),int(points[i,1])] = photons[i]
        data = cv2.filter2D(image, -1, self.kernel.array, borderType=cv2.BORDER_CONSTANT)
        return data

    def concolve_flimbi_style(self, image, ground_truth, points, on_points, switching_rate):
        line = self.shape[0]/self.image_shape[0]
        on_points = np.delete(on_points, np.where(on_points[...,2] < 0), axis=0)


        exclude = []

        for i in range(self.image_shape[0]):
            n_on = self.rs.poisson(switching_rate)
            if n_on > 0:
                ind = self.rs.choice(points.shape[0], n_on)
                invalid_ind = []
                for k in range(ind.shape[0]):
                    if points[ind[k],0]+600 < i*line:
                        invalid_ind.append(k)
                ind = np.delete(ind, invalid_ind)
                # indices = line_point_indices[ind]
                on_points = np.concatenate((on_points, points[ind]), axis=0)


            image,ex = Factory.flimbi_paint_locs(image, on_points[1:], self.kernel.array, line, i)
            exclude += ex
        exclude = list(set(exclude))
        exclude = [i+1 for i in exclude]
        print("localization " + str(exclude) + " was not painted")
        on_points = np.delete(on_points, exclude, axis=0)
        ground_truth = self.create_points_add_photons(ground_truth, on_points[1:], np.ones(on_points.shape[0])*800)#todo: ground truth as points
        return image, ground_truth, on_points[1:]

    @staticmethod
    @numba.jit(nopython=True)
    def flimbi_paint_locs(image, on_points, kernel_array, line, j, flimbi_mode=True):
        #always decrease lifetime
        exclude = []
        off = []
        for i in range(on_points.shape[0]):
            y,x = int(on_points[i,0]),int(on_points[i,1])
            on_points[i,2] -= 6
            if flimbi_mode:
                b = False
                #for j in off:
                #    if j==i:
                #        b=True
                #if b:
                #    continue
                #caution localizations might not be painted at all
                if on_points[i,2] < 0:#todo kick point if it wasnt painted
                    off.append(i)

                    if j * line < y - 600:
                        exclude.append(i)
                    continue
            #if line %2 ==0:
            #   x -= int(on_points[i,4])
            vertical_i_range = np.array([min(max(j * line, y - 600), (j + 1) * line),
                                         max(min((j + 1) * line - 1, y + 600), line * j)]).astype(np.int32)
            vertical_k_range = np.array([min(max(vertical_i_range[0] - (y - 600), 0), 1200),
                                         max(min(vertical_i_range[1] - (y - 600), 1200), 0)]).astype(np.int32)

            horizontal_i_range = np.array([max(x - 600, 0), min(x + 600, image.shape[1] - 1)]).astype(
                np.int32)
            horizontal_k_range = np.array(
                [horizontal_i_range[0] - (x - 600), horizontal_i_range[1] - (x - 600)]).astype(np.int32)
            #print(vertical_i_range, vertical_k_range, horizontal_k_range, horizontal_i_range)
            image[vertical_i_range[0]:vertical_i_range[1],
            horizontal_i_range[0]:horizontal_i_range[1]] += kernel_array[vertical_k_range[0]:vertical_k_range[1],
                                                            horizontal_k_range[0]:horizontal_k_range[1]].astype(np.float32) * 800
        return image,exclude


