import numpy as np
from astropy.convolution import Gaussian2DKernel
import cv2
import numba
import scipy
import copy
from collections import namedtuple

class Factory():
    def __init__(self):
        self._kernel = Gaussian2DKernel(x_stddev=150, y_stddev=150)
        self.shape = (8192, 8192)
        self.image_shape = (64, 64)
        seed = 42  # switch seed
        self.rs = np.random.RandomState(seed)

    @property
    def kernel(self):
        return self._kernel._array

    @kernel.setter
    def kernel(self, value):
        self._kernel = Gaussian2DKernel(x_stddev=value[0], y_stddev=value[1])
        print(self._kernel.shape)

    def create_image(self):
        image = np.zeros(self.shape).astype(np.float32)
        return image

    def reduce_size(self, image):
        return cv2.resize(image, self.image_shape, interpolation=cv2.INTER_AREA)*(self.shape[0]/self.image_shape[0])**2

    def create_crop_point_set(self, photons=1200, on_time=400):
        n_points = 100000
        points = np.zeros((n_points, 5)).astype(np.float32)
        distribution = np.random.poisson(photons, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
        on_time = np.random.poisson(on_time, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)

        for i in range(n_points):
            p = np.random.randint(150, self.shape[0]-150, size=2)
            points[i, 0:2] = p
            points[i, 2] = distribution[i]  # todo: increase lifetime?
            points[i, 3] = on_time[i]
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

    def create_points_add_photons(self, image, points, photons):
        for i in range(points.shape[0]):
            image[int(points[i,0]),int(points[i,1])] = photons[i]
        data = cv2.filter2D(image, -1, self.kernel, borderType=cv2.BORDER_CONSTANT)
        return data

    def simulate_accurate_flimbi(self, points, on_points, switching_rate=8.0, inverted=False):
        """input random distributed convolved and resized image"""
        #todo: simulate world in 0.1 ms

        image = self.create_image()
        ground_truth = self.create_image()
        image, ground_truth, on_points = self.concolve_flimbi_style(image, ground_truth, points, on_points, switching_rate=switching_rate, inverted=inverted)
        return image, ground_truth, on_points

    def concolve_flimbi_style(self, image, ground_truth, points, on_points, switching_rate, inverted=False):
        line = self.shape[0]/self.image_shape[0]
        on_points = np.delete(on_points, np.where(on_points[...,3] < 0), axis=0)
        exclude = []

        for i in range(self.image_shape[0]):#todo: restructure this
            n_on = self.rs.poisson(switching_rate)
            if n_on > 0:
                ind = self.rs.choice(points.shape[0], n_on)
                invalid_ind = []
                for k in range(ind.shape[0]):
                    if points[ind[k],0]+450 < i*line:
                        invalid_ind.append(k)
                ind = np.delete(ind, invalid_ind)
                # indices = line_point_indices[ind]
                on_points = np.concatenate((on_points, points[ind]), axis=0)


            image,ex = Factory.flimbi_paint_locs(image, on_points, self.kernel, line, i, inverted=inverted)
            exclude += ex
        exclude = list(set(exclude))
        exclude = [i+1 for i in exclude]
        #print("localization " + str(exclude) + " was not painted")
        # if len(exclude)>0:
        #     on_points = np.delete(on_points, exclude, axis=0)
        ground_truth = self.create_points_add_photons(ground_truth, on_points, np.ones(on_points.shape[0])*800)#todo: ground truth as points
        return image, ground_truth, on_points

    @staticmethod
    #@numba.jit(nopython=True)
    def flimbi_paint_locs(image, on_points, kernel_array, line, j, flimbi_mode=True, inverted=False):
        #always decrease lifetime
        exclude = []
        off = []
        for i in range(on_points.shape[0]):
            Localization = namedtuple("Localization", ("x","y","x_start","x_end","y_start","y_end","intensity"))
            Localization.x = int(on_points[i,1])
            Localization.y = int(on_points[i,0])
            on_points[i,3] -= 6
            Localization.intensity = on_points[i, 2]
            if flimbi_mode:
                b = False
                #for j in off:
                #    if j==i:
                #        b=True
                #if b:
                #    continue
                #caution localizations might not be painted at all
                if inverted:
                    if on_points[i, 3] > 0:  # todo kick point if it wasnt painted
                        off.append(i)

                        if j * line < Localization.y - 450:
                            exclude.append(i)
                        continue
                else:
                    if on_points[i,3] < 0:#todo kick point if it wasnt painted
                        off.append(i)

                        if j * line < Localization.y - 450:
                            exclude.append(i)
                        continue
            kernel_range_y = int(kernel_array.shape[0]/2)
            kernel_range_x = int(kernel_array.shape[1]/2)

            Localization.y_start = Localization.y-kernel_range_y
            Localization.y_end = Localization.y+kernel_range_y+1
            #constrain localization to be in current raster line
            vertical_i_range = np.array([min(max(j * line, Localization.y_start), (j + 1) * line),
                                         max(min((j + 1) * line, Localization.y_end), line * j)]).astype(np.int32)
            vertical_k_range = np.array([vertical_i_range[0] - Localization.y_start,
                                         vertical_i_range[1] - Localization.y_start]).astype(np.int32)

            Localization.x_start = Localization.x-kernel_range_x
            Localization.x_end = Localization.x+kernel_range_x+1
            horizontal_i_range = np.array([max(Localization.x_start, 0),
                                           min(Localization.x_end, image.shape[1])]).astype(np.int32)
            horizontal_k_range = np.array(
                [horizontal_i_range[0] - Localization.x_start,
                 horizontal_i_range[1] - Localization.x_start])
            #print(vertical_i_range, vertical_k_range, horizontal_k_range, horizontal_i_range)
            #if vertical_k_range[0]<0 or vertical_k_range[1]>2*kernel_range_y:
                #print(vertical_i_range, vertical_k_range)
                #continue
            #if horizontal_k_range[0]<0 or horizontal_k_range[1]>2*kernel_range_x:
                #print(horizontal_k_range, horizontal_i_range)
                #continue
            image[vertical_i_range[0]:vertical_i_range[1],
            horizontal_i_range[0]:horizontal_i_range[1]] += kernel_array[vertical_k_range[0]:vertical_k_range[1],
                                                            horizontal_k_range[0]:horizontal_k_range[1]].astype(np.float32) * Localization.intensity
        return image,exclude

    def create_microtuboli_point_set(self):
        # import matplotlib.pyplot as plt
        # image = self.create_image()
        all_indices = []
        for k in range(10):
            starting_point = np.random.randint(int(self.shape[0]/4),int(3*self.shape[1]/4), 2).astype(np.float32)#start in the middle of the image
            direction = np.random.rand(1)*np.pi
            for i in range(5000):
                if i%50==0:
                    direction += (np.random.rand(1)-0.5)*0.3
                ratio = np.abs(np.tan(float(direction)%np.pi))
                x = ratio/(ratio+1)
                y = 1/(ratio+1)
                starting_point += np.array([x, y])
                indices=[]
                for j in range(30):
                    j -= 15
                    prob = 0.1*np.exp(-(np.abs(j)-15)**2/(2*15**2))
                    new_point = starting_point + np.array([-j*y,j*x])
                    if new_point[0]<self.shape[0] and new_point[1]<self.shape[1] and new_point[0]>0 and new_point[1]>0 and np.random.rand()<prob:
                        indices.append(new_point)
                indices = np.array(indices).astype(np.int32)
                if indices.any():
                    all_indices.append(indices)

        all_indices = np.unique(np.concatenate(all_indices, axis=0), axis=0)
        return all_indices
        # image[all_indices[:,0], all_indices[:,1]] +=1
        #
        # plt.imshow(image)
        # plt.show()

    def point_set_to_storm_point_set(self, points, frames, on_time=600):
        acquisition_time= 300
        switching_rate = 1
        distribution = np.random.poisson(on_time, points.shape[0])
        points_new = np.zeros((points.shape[0],3))
        points_new[:,0:2] = points
        points_new[:,3] = distribution
        init_indices = np.random.choice(points.shape[0], 10)
        on_points = copy.deepcopy(points_new[init_indices])
        localizations = []
        for i in range(frames):
            localizations.append(on_points)
            on_points[:,3] -= acquisition_time
            on_points = np.delete(on_points, np.where(on_points[:,3]<0))
            n_on = self.rs.poisson(switching_rate)
            if n_on > 0:
                ind = self.rs.choice(points_new.shape[0], n_on)
                on_points.append(copy.deepcopy(points_new[ind]))
        return localizations

    def apply_changing_linear_drift(self, localizations, x, y, delta=0):
        for i in range(localizations.shape[0]):
            localizations[i][:,0] += x
            localizations[i][:,1] += y
            x += delta
            y += delta
            if i%100 == 0:
                delta = float(np.random.rand()*0.1-0.05)
        return localizations

    def create_classifier_image(self, size, points, px_size):
        image = np.zeros((size[0], size[1], 4))
        for point in points:
            y = (point[0]%px_size)/px_size-0.5
            x = (point[1]%px_size)/px_size-0.5
            image[int(point[0]//px_size), int(point[1]//px_size), 2] = 1
            image[int(point[0]//px_size), int(point[1]//px_size), 3] = 1

            if y < 0:
                image[int(point[0]//px_size)-1, int(point[1]//px_size), 3] = 1
                image[int(point[0]//px_size)-1, int(point[1]//px_size), 0] = y+1
                image[int(point[0]//px_size)-1, int(point[1]//px_size), 1] = x

            else:
                image[int(point[0]//px_size)+1, int(point[1]//px_size), 3] = 1
                image[int(point[0] // px_size) + 1, int(point[1] // px_size), 0] = y-1
                image[int(point[0] // px_size) + 1, int(point[1] // px_size), 1] = x
            if x<0:
                image[int(point[0]//px_size), int(point[1]//px_size)-1, 3] = 1
                image[int(point[0]//px_size), int(point[1]//px_size)-1, 0] = y
                image[int(point[0]//px_size), int(point[1]//px_size)-1, 1] = x+1
            else:
                image[int(point[0]//px_size), int(point[1]//px_size)+1, 3] = 1
                image[int(point[0]//px_size), int(point[1]//px_size)+1, 0] = y
                image[int(point[0]//px_size), int(point[1]//px_size)+1, 1] = x-1


            #todo: linear interpolation over connceted pixels
            # image[:, :, 2] = scipy.ndimage.gaussian_filter(image[:,:,2], sigma=1)
            # if image[:,:,2].max() != 0:
            #     image[:,:,2] /= image[:,:,2].max()
            #image[np.where(image[:,:,2]==0),2] = -1
            image[int(point[0]//px_size), int(point[1]//px_size), 0] = (point[0]%px_size)/px_size-0.5#norm on subpixel E [0,1]
            image[int(point[0]//px_size), int(point[1]//px_size), 1] = (point[1]%px_size)/px_size-0.5
        return image

if __name__ == '__main__':
    factory = Factory()
    factory.create_microtuboli_point_set()