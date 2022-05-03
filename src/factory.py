import numpy as np
from astropy.convolution import Gaussian2DKernel,AiryDisk2DKernel
import cv2
import scipy
import copy
from collections import namedtuple

import tensorflow as tf
import tensorflow_addons as tfa

class Kernel():
    def __init__(self):
        self.px_size = 100
        seed = 42  # switch seed
        self.quantum_efficiency = 0.45
        self.dark_noise = 5 / 10  # dark noise was 25 / 10
        self.sensitivity = 1
        self.bitdepth = 12
        self.image_shape = (9,9)
        self.rs = np.random.RandomState(seed)

    def apply_intensity(self, image, intensities):
        im_max = tf.reduce_max(image, axis=(0,1,2))
        image /= im_max
        image *= tf.constant(intensities.astype(np.float32)/30)
        return image


    def point_set_simulator(self, t, photons=1200, average_lifetime=600, frames=50):
        #todo: image dimension in y average lifetime of the on state
        #todo: lifetime to line on line off
        #todo: integration time per pixel to calculate on time!

        #todo: coordinates x,y,z, switching off, switching on
        #todo: simulate 3 consecutive frames
        mean_loc_count = 1.5
        n_points = 10000
        #points = np.zeros((n_points, 5)).astype(np.float32)
        photon_distribution = np.random.normal(photons,0.3*photons, n_points)  # todo: higher distribution
        #on_time = np.random.poisson(average_lifetime, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
        #delay_time = np.random.poisson(30, n_points)
        points = []
        indices = []
        for i in range(frames):
            #todo: simulate 3 points in a row
            frame_points = int(np.random.normal(mean_loc_count, 0.2*mean_loc_count))#np.random.poisson(1.7)
            if frame_points<0:
                frame_points=0
            xy = np.random.randint(150, 900 - 150, size=(frame_points,2)) #9 is crop size
            on_time = np.random.poisson(average_lifetime, frame_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
            delay_time = np.random.poisson(30, frame_points)

            start = delay_time//t
            frame = start//9#cropsize
            px = start%9

            stop = (delay_time+on_time)//t
            stop_frame = stop//9
            stop_px = stop%9
            for j in range(3):
                n=0
                s = len(points)

                for frp in range(frame_points):
                    if frame[frp]<=j and stop_frame[frp]>=j:
                        n+=1
                        on_add = (frame[frp] -j)*9
                        off_add = (stop_frame[frp] -j)*9
                        p = np.array([xy[frp,0],xy[frp,1], photon_distribution[i], px[frp]+on_add, stop_px[frp]+off_add])
                        points.append(p)
                indices.append(np.arange(s,s+n))
        return points,indices
            # p = np.random.randint(150, self.shape[0]-150, size=2)
            # points[i, 0:2] = p
            # points[i, 2] = photon_distribution[i]  # todo: increase lifetime?
            # points[i, 3] = on_time[i]   #todo: compute xstart and x_end
            # points[i, 4] = delay_time[i]

        #pass

    @property
    def kernel(self):
        #todo: 3D or 2D
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        #todo: set array and turn it to tensor
        self._kernel = tf.constant(value)

    def create_images(self, image_batch, z, mod, tf_mask):
        kernel_point_tensor = tf.gather(self._kernel, z)

        #translate xy arrays here
        shifted_kernel = tfa.image.translate(tf.expand_dims(kernel_point_tensor,-1), tf.cast(mod, tf.float32))
        tf_image_batch=tf.expand_dims(image_batch.astype(np.float32),-1)
        n_shape = tf.shape(shifted_kernel)[1:3]//100

        #inter area interpolation of kernel
        #convolve
        resized_kernel = tf.transpose(tf.image.resize(shifted_kernel,n_shape ,method=tf.image.ResizeMethod.AREA),(2,1,0,3))
        resized_kernel = resized_kernel*tf_mask
        image = tf.nn.depthwise_conv2d(tf.transpose(tf_image_batch,(3,1,2,0)), resized_kernel[::-1,::-1],strides=[1, 1, 1, 1], padding='SAME')
        return image,resized_kernel

    def accurate_noise_simulations_camera(self, image):
        image = image.astype(np.uint16)

        shot_noise = self.rs.poisson(image, image.shape)

        # Round the result to ensure that we have a discrete number of electrons
        electrons = np.round(self.quantum_efficiency * shot_noise)

        # electrons dark noise 34 counts per second
        electrons_out = np.round(self.rs.normal(scale=self.dark_noise, size=electrons.shape) + electrons)

        # ADU/e-
        max_adu = np.int(2 ** self.bitdepth - 1)
        adu = (electrons_out * self.sensitivity).astype(np.int)
        adu[adu > max_adu] = max_adu  # models pixel saturation
        adu[adu < 0] = 0  # models pixel saturation
        return adu

    def create_data(self, points, size, index_list):
        batch_size= 150
        images = []
        noiseless = []
        p_array = []
        n = len(index_list)//batch_size
        start = 0
        if len(index_list)%batch_size !=0:
            n+=1
        for k in range(n):
            print(k)
            #pick 150 frames
            current_index_list = copy.deepcopy(index_list[k*batch_size:(k+1)*batch_size])
            z = np.concatenate(current_index_list,axis=0)
            current_points = points[z]
            for item in current_index_list:
                item -= start
            im, nl, p = self.compute_point_batch(current_points, size, current_index_list)
            images.append(im)
            noiseless.append(im)
            p_array.append(p)
            start = z.max()+1
        return np.concatenate(images, axis=0), np.concatenate(noiseless, axis=0), np.concatenate(p_array,axis=0)


    #todo: outsource to tf function
    def compute_point_batch(self, point, size, index_list):
        """takes array of points, size of images to construct and a list of indices which points correspond to which image"""
        #todo: batch points to fit on gpu
        #if len(point.shape) != 3:
        #    raise ValueError("point shape not fitting")
        #todo: indices to points of one image
        point[np.where(point[:,3]<0),3] = 0
        point[np.where(point[:,4]<0),4] = 0

        #create image batch0
        image_batch = np.zeros((tf.shape(point)[0],size[0],size[1]))

        indices = np.arange(0, point.shape[0], 1)[:,np.newaxis]
        point_xy = point[:,0:2]
        #get offset to pixel center with modulo
        #adjust range from -0.5 to +0.5
        pix = (point_xy//self.px_size).astype(np.int32)
        mod = point_xy%self.px_size-tf.constant(.5)

        z= tf.zeros((tf.shape(point)[0]),tf.int32)

        indices = np.concatenate([indices, pix], axis=1).T
        image_batch[(indices[0],indices[1],indices[2])] = 1

        #z = point[:,2]
        mask = np.zeros((12,12,tf.shape(point)[0],1))
        #todo: make large set of masks depending on lifetime
        #todo: define point set with int of switch on and switch off
        for i in range(point.shape[0]):
            mask[point[i,3].astype(np.int32):point[i,4].astype(np.int32),:,i] = 1
        tf_mask = tf.constant(mask.astype(np.float32))

        image, resized_kernel = self.create_images(image_batch, z, mod, tf_mask)
        image = self.apply_intensity(image, point[:,2])
        result_images = []
        #iterate over indices and gather and reduce sum
        points = [] #todo: points as array
        for i,point_set in enumerate(index_list):
            result_images.append(tf.reduce_sum(tf.gather(image, point_set, axis=-1),axis=-1, keepdims=True))
            if i%3 == 1:
                #todo: points as np array with 10 indices
                p = np.zeros((10,5))
                p[0:point_set.shape[0]] = point[point_set]
                points.append(p)
        reshaped_nl = []
        reshaped = []

        for i in range(len(result_images)//3):
            sublist_nl = [result_images[3*i+j][0]  for j in range(3)]
            reshaped_nl.append(tf.concat(sublist_nl, axis=-1).numpy())
            #add nosie
            sublist = [self.accurate_noise_simulations_camera(tf.squeeze(result_images[3*i+j]).numpy()+5)  for j in range(3)]
            reshaped.append(np.stack(sublist, axis=-1))

        reshaped_nl = np.array(reshaped_nl)#noiseless
        reshaped = np.array(reshaped)
        points = np.array(points)
        #todo: as numpy add noise restack and safe...
        return reshaped, reshaped_nl, points



class Factory():
    def __init__(self):
        self._kernel = Gaussian2DKernel(x_stddev=150, y_stddev=150)
        self.shape = (8192, 8192)
        self.image_shape = (64, 64)
        seed = 42  # switch seed
        self.rs = np.random.RandomState(seed)
        self.kernel_type = "Gaussian"
        self.quantum_efficiency = 0.45
        self.dark_noise = 5 / 10  # dark noise was 25 / 10
        self.sensitivity = 1
        self.bitdepth = 12

    @property
    def kernel(self):
        return self._kernel._array

    @kernel.setter
    def kernel(self, value):
        if self.kernel_type=="Airy":
            print("creating airy disc")
            s = int(8*int(value[0])+1)
            self._kernel = AiryDisk2DKernel(radius= 3*value[0],x_size=s,y_size=s)#r = 3 std
        else:
            print("creating gaussian")
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
        photon_distribution = np.random.normal(photons,0.3*photons, n_points)  # todo: higher distribution
        on_time = np.random.poisson(on_time, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
        delay_time = np.random.poisson(on_time, n_points)

        for i in range(n_points):
            p = np.random.randint(150, self.shape[0]-150, size=2)
            points[i, 0:2] = p
            points[i, 2] = photon_distribution[i]  # todo: increase lifetime?
            points[i, 3] = on_time[i]
            points[i, 4] = delay_time[i]
        return points

    def create_point_set(self, on_time=600):
        n_points = 100000
        points = np.zeros((n_points, 5)).astype(np.float32)
        distribution = np.random.poisson(on_time, n_points)  # maxwell.rvs(size=n_points, loc=12.5, scale=5)
        #todo: add photons
        for i in range(n_points):
            p = np.random.randint(0, self.shape[0], size=2)
            points[i, 0:2] = p
            points[i, 2] = distribution[i]  # todo: increase lifetime?
            points[i, 3] = np.random.randint(0, int(points[i, 2]))
            points[i, 4] = np.random.randint(-25, 25)
        return points

    def build_crop(self, ind, switching_array, points ):  # todo: pass points here
        image_s = np.zeros((self.image_shape[0], self.image_shape[1], 3))
        image_noiseless = np.zeros((self.image_shape[0], self.image_shape[1], 3))
        #factory stuff!

        local_bg = np.random.choice(a=[True, False], size=1, p=[0.1, 0.9])[0]  # activated recently
        for i in range(3):

            image, gt, on_points = self.simulate_accurate_flimbi(points, points[ind], switching_rate=0,
                                                                    inverted=switching_array[:, i])  # bottleneck
            image_noiseless[:, :, i] = self.reduce_size(gt).astype(np.float32)

            image = self.reduce_size(image).astype(np.float32)
            # local bg simulation:
            if local_bg:
                image += np.random.rand() * 5 + 15  # noise was 2

            # image_noiseless[:,:,i] = copy.deepcopy(image)
            image = self.accurate_noise_simulations_camera(image).astype(np.float32)
            # plt.scatter(on_points[:,1]/100,on_points[:,0]/100)
            # plt.imshow(image)
            # plt.show()

            image_s[:, :, i] = image
            if i == 1:
                truth_cs = self.create_classifier_image( on_points,
                                                           100)  # todo: variable px_size
                main_points = on_points

        return image_s, truth_cs, image_noiseless, main_points


    def accurate_noise_simulations_camera(self, image):
        image = image.astype(np.uint16)

        shot_noise = self.rs.poisson(image, image.shape)

        # Round the result to ensure that we have a discrete number of electrons
        electrons = np.round(self.quantum_efficiency * shot_noise)

        # electrons dark noise 34 counts per second
        electrons_out = np.round(self.rs.normal(scale=self.dark_noise, size=electrons.shape) + electrons)

        # ADU/e-
        max_adu = np.int(2 ** self.bitdepth - 1)
        adu = (electrons_out * self.sensitivity).astype(np.int)
        adu[adu > max_adu] = max_adu  # models pixel saturation
        adu[adu < 0] = 0  # models pixel saturation
        return adu

    def create_points_add_photons(self, image, points, photons):
        for i in range(points.shape[0]):
            image[int(points[i,0]),int(points[i,1])] = photons[i]
        data = cv2.filter2D(image, -1, self.kernel, borderType=cv2.BORDER_CONSTANT)
        return data

    def simulate_accurate_flimbi(self, points, on_points, switching_rate=8.0, inverted=None):
        """input random distributed convolved and resized image"""
        #todo: simulate world in 0.1 ms

        image = self.create_image()
        ground_truth = self.create_image()
        image, ground_truth, on_points = self.concolve_flimbi_style(image, ground_truth, points, on_points, switching_rate=switching_rate, inverted=inverted)
        return image, ground_truth, on_points

    def concolve_flimbi_style(self, image, ground_truth, points, on_points, switching_rate, inverted=None):
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
        #print("localization " + str(exclude) + " was not painted")
        if len(exclude)>0:
            on_points = np.delete(on_points, exclude, axis=0)
        ground_truth = self.create_points_add_photons(ground_truth, on_points, np.ones(on_points.shape[0])*800)#todo: ground truth as points
        return image, ground_truth, on_points

    @staticmethod
    #@numba.jit(nopython=True)
    def flimbi_paint_locs(image, on_points, kernel_array, line, j, flimbi_mode=True, inverted=None):
        #always decrease lifetime
        exclude = []
        off = []
        for i in range(on_points.shape[0]):
            Localization = namedtuple("Localization", ("x","y","x_start","x_end","y_start","y_end","intensity"))
            Localization.x = int(on_points[i,1])
            Localization.y = int(on_points[i,0])
            on_points[i,3] -= 6
            Localization.intensity = on_points[i, 2]#*np.abs(np.random.normal(1, 0.2, 1))*int(np.random.normal(1, 0.3, 1)+0.5) #todo random intensity fluctuations
            if np.any(inverted):
                if inverted[i] == 1:
                    if on_points[i, 4] > 0:  # todo kick point if it wasnt painted
                        off.append(i)

                        if j * line < Localization.y - 100:
                            exclude.append(i)
                        continue
                elif inverted[i] == 3:
                    if on_points[i,3] < 0:#todo kick point if it wasnt painted
                        off.append(i)

                        if j * line < Localization.y - 100:
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
        points_new[:,2] = distribution
        init_indices = np.random.choice(points.shape[0], 10)
        on_points = copy.deepcopy(points_new[init_indices])
        localizations = []
        for i in range(frames):
            localizations.append(on_points)
            on_points[:,2] -= acquisition_time
            on_points = np.delete(on_points, np.where(on_points[:,2]<0))
            n_on = self.rs.poisson(switching_rate)
            if n_on > 0:
                ind = self.rs.choice(points_new.shape[0], n_on)
                on_points = np.concatenate([on_points, copy.deepcopy(points_new[ind])], axis=0)
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

    def create_classifier_image(self,  points, px_size):
        image = np.zeros((self.image_shape[0], self.image_shape[1], 4))
        for point in points:
            y = (point[0]%px_size)/px_size-0.5
            x = (point[1]%px_size)/px_size-0.5
            image[int(point[0]//px_size), int(point[1]//px_size), 2] = 1
            image[int(point[0]//px_size), int(point[1]//px_size), 3] = 1

            y_s = int(np.sign(y))
            x_s = int(np.sign(x))
            #todo: 2x2
            #y neighbor
            image[int(point[0]//px_size)+y_s, int(point[1]//px_size), 3] = 1
            image[int(point[0]//px_size)+y_s, int(point[1]//px_size), 0] = y-y_s
            image[int(point[0]//px_size)+y_s, int(point[1]//px_size), 1] = x
            #x_neighbor
            image[int(point[0]//px_size), int(point[1]//px_size)+x_s, 3] = 1
            image[int(point[0]//px_size), int(point[1]//px_size)+x_s, 0] = y
            image[int(point[0]//px_size), int(point[1]//px_size)+x_s, 1] = x-x_s
            #diag
            image[int(point[0]//px_size)+y_s, int(point[1]//px_size)+x_s, 3] = 1
            image[int(point[0]//px_size)+y_s, int(point[1]//px_size)+x_s, 0] = y-y_s
            image[int(point[0]//px_size)+y_s, int(point[1]//px_size)+x_s, 1] = x-x_s

            #todo: linear interpolation over connceted pixels
            # image[:, :, 2] = scipy.ndimage.gaussian_filter(image[:,:,2], sigma=1)
            # if image[:,:,2].max() != 0:
            #     image[:,:,2] /= image[:,:,2].max()
            #image[np.where(image[:,:,2]==0),2] = -1
            image[int(point[0]//px_size), int(point[1]//px_size), 0] = y#norm on subpixel E [0,1]
            image[int(point[0]//px_size), int(point[1]//px_size), 1] = x
        return image

if __name__ == '__main__':
    factory = Factory()
    points = factory.create_microtuboli_point_set()
    localizations = factory.point_set_to_storm_point_set(points, 3000, )
    x=0