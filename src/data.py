import tensorflow as tf
from tifffile import TiffFile as TIF
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from src.factory import Factory
import copy
from astropy.convolution import Gaussian2DKernel
from src.utility import get_root_path

CROP_TRANSFORMS = 4
OFFSET=14

def build_switching_array(n):
    bef_after = np.random.randint(0, 2, 2 * n)
    test = np.ones((n, 3))
    test2 = np.zeros((n, 3))
    test[:, 0] = bef_after[0:n]
    test[:, 2] = bef_after[n:]
    for i in range(n):
        if test[i, 0] == 1 and test[i, 2] == 0:
            test2[i, 0] = 1  # switching on
            test2[i, 1] = 3  # switching off
        elif test[i, 2] == 1 and test[i, 0] == 0:
            test2[i, 1] = 1  # switching on
            test2[i, 2] = 3  # switching off
        elif test[i, 2] == 1 and test[i, 0] == 1:
            test2[i, 0] = 1  # switching on
            test2[i, 1] = 2  # switching on
            test2[i, 2] = 3  # switching off
        else:
            test2[i, 1] = 2
    return test2


def crop_generator_u_net(im_shape, sigma_x=150, sigma_y=150, seed=0, noiseless_ground_truth=False):
    #todo: create dynamical

    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        for z in range(100):
            ph = np.random.randint(1000,2000)
            points = factory.create_crop_point_set(photons=ph, on_time=30)
            #sigma_y = np.random.randint(100, 250)
            sig_y = sigma_x+np.random.rand()*10-5
            sig_x = sigma_y+np.random.rand()*10-5

            factory.kernel = (sig_x, sig_y)
            truth_cs_list = []
            image_list = []
            image_noiseless_list = []
            for i in range(100): #todo: while loop here
                print(i)

                ind = np.random.randint(0,points.shape[0])
                n = int(np.random.normal(1.5,0.4,1))#np.random.poisson(1.7)
                if n>3:
                    n=3
                def build_image(ind, switching_array,):#todo: points to image additional parameters sigma and intensity
                    image_s = np.zeros((im_shape, im_shape, 3))
                    truth_cs = factory.create_classifier_image((9, 9), points[ind], 100)  # todo: variable px_size
                    image_noiseless = np.zeros((im_shape, im_shape, 3))
                    for i in range(3):
                        #todo: if before is zero activate switching
                        #todo: if after is zero activate inverted
                            #todo: for points with right variables
                        image,_,_ = factory.simulate_accurate_flimbi(points, points[ind], switching_rate=0, inverted=switching_array[:,i])

                        # plt.imshow(image)
                        # plt.show()
                        image = factory.reduce_size(image).astype(np.float32)
                        image_noiseless[:,:,i] = image
                        image += np.random.rand()*20+10 #noise was 2
                        image = factory.accurate_noise_simulations_camera(image).astype(np.float32)
                        image_s[:,:,i] = image

                    return image_s, truth_cs, image_noiseless

                #todo: build named tuple (i.e. true true flase)
                #todo: set switching accordingly(i.e. true, inverted=true, false
                ind = np.arange(ind, ind + n, 1).astype(np.int32)#todo: set switching true
                switching_array = build_switching_array(n)

                image_s,truth_cs, image_noiseless = build_image(ind, switching_array)

                image_s -= image_s.min()
                image_s += 0.0001
                image_s /= image_s.max()
                image_s *= np.random.rand()*0.3+0.7
                #fig,axs = plt.subplots(3)
                #axs[0].imshow(truth_cs[:,:,0])
                #axs[1].imshow(truth_cs[:,:,1])
                #axs[2].imshow(truth_cs[:,:,2])

                #plt.show()

                truth_cs_list.append(truth_cs)
                image_list.append(image_s)
                image_noiseless_list.append(image_noiseless)
            if noiseless_ground_truth:
                yield tf.convert_to_tensor(image_list), tf.convert_to_tensor(truth_cs_list), tf.convert_to_tensor(image_noiseless_list)  # todo: shuffle?
            else:
                yield tf.convert_to_tensor(image_list), tf.convert_to_tensor(truth_cs_list)#todo: shuffle?
    return generator

def crop_generator(im_shape, sigma_x=150, sigma_y=150):
    #todo: create dynamical

    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        for z in range(100):
            ph = np.random.randint(1000,2000)
            points = factory.create_crop_point_set(photons=ph)
            #sigma_y = np.random.randint(100, 250)
            factory.kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)
            points_list = []
            image_list = []
            for i in range(100): #todo: while loop here
                print(i)

                ind = np.random.randint(0,points.shape[0])
                n = int(np.random.normal(1.5,0.4,1))#np.random.poisson(1.7)
                if n>3:
                    n=3
                def build_image(ind):
                    image = factory.create_image()
                    p = np.zeros(9)
                    for z in range(n):
                        p[6+z] = 1
                    p[0:ind.shape[0]*2] = points[ind,0:2].flatten()
                    image = factory.create_points_add_photons(image, points[ind], points[ind,2])
                    image = factory.reduce_size(image).astype(np.float32)
                    image += 3
                    return image, p

                ind = np.arange(ind, ind + n, 1).astype(np.int32)
                image_s = np.zeros((im_shape,im_shape, 3))

                image_s[:, :, 1],p = build_image(ind)

                bef_after = np.random.randint(0,2,2*n)
                ind_new_b = ind
                ind_new_a = ind
                ind_new_b = np.delete(ind_new_b, np.where(bef_after[:n]==0))
                ind_new_a = np.delete(ind_new_a, np.where(bef_after[n:]==0))
                image_s[:, :, 2],_ = build_image(ind_new_a)
                image_s[:, :, 0],_ = build_image(ind_new_b)


                #done: random new noise in next image random switch off
                for t in range(CROP_TRANSFORMS):
                    image_s_copy = copy.deepcopy(image_s)
                    p_n = copy.deepcopy(p)
                    for k in range(3):
                        image_s_copy[:,:,k] = factory.accurate_noise_simulations_camera(image_s_copy[:,:,k])

                    image_s_copy -= image_s_copy.min()
                    image_s_copy /= image_s_copy.max()
                    if t == 1:
                        image_s_copy = np.fliplr(image_s_copy)
                        p_n[1:6:2] = ((factory.shape[1] ) - p_n[1:6:2])*p[6:]
                    elif t == 2:
                        image_s_copy = np.flipud(image_s_copy)
                        p_n[0:6:2] = ((factory.shape[0] - 1) - p_n[0:6:2])*p[6:]
                    elif t == 3:
                        image_s_copy = np.flipud(np.fliplr(image_s_copy))
                        p_n[1:6:2] = ((factory.shape[1] - 1) - p_n[1:6:2])*p[6:]
                        p_n[0:6:2] = ((factory.shape[0] - 1) - p_n[0:6:2])*p[6:]
                    points_list.append(p_n)
                    image_list.append(image_s_copy)


            yield tf.convert_to_tensor(image_list), tf.convert_to_tensor(np.array(points_list))#todo: shuffle?
    return generator

def real_data_generator(im_shape, switching_rate=0.2):
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        points = factory.create_point_set()
        init_indices = np.random.choice(points.shape[0], 10)
        on_points = points[init_indices]
        for i in range(10000): #todo: while loop here
            print(i)
            image, truth, on_points = factory.simulate_accurate_flimbi(points, on_points, switching_rate=switching_rate)#todo: simulate off
            image = factory.reduce_size(image)#todo: background base lvl?
            image += 1/3*image.max()
            image*=3
            image = np.pad(factory.accurate_noise_simulations_camera(image),(14,14))
            truth = np.pad(factory.reduce_size(truth).astype(np.int32),(14,14))
            yield image, truth, np.array(on_points)
    return generator


def generate_generator(image):
    def data_generator_real():
        for i in range(1):
            dat = image[i * 10000:(i + 1) * 10000,]#14:-14,14:-14]
            #dat = dat[:dat.shape[0]//4*4]
            #dat = dat[::4] + dat[1::4] + dat[2::4] + dat[3::4] #todo shift ungerade
            #dat[:, 1::2] = scipy.ndimage.shift(dat[:,1::2], (0,0,0.5))
            #dat[:,1::2,1:] = dat[:,1::2,:-1]
            dat -= dat.min()
            data = np.zeros((dat.shape[0], 128, 128, 3))  # todo: this is weird data
            data[:, OFFSET:OFFSET + dat.shape[1], OFFSET:OFFSET + dat.shape[2], 1] = dat
            data[1:, OFFSET:OFFSET + dat.shape[1], OFFSET:OFFSET + dat.shape[2], 0] = dat[:-1]
            data[:-1, OFFSET:OFFSET + dat.shape[1], OFFSET:OFFSET + dat.shape[2], 2] = dat[1:]
            yield data
    return data_generator_real

def data_generator_coords(file_path, offset=0):
    with TIF(file_path) as tif:
        dat = tif.asarray()
    data = np.zeros((dat.shape[0],128, 128, 3))
    data[:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],1] = dat
    data[1:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],0] = dat[:-1]
    data[:-1,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],2] = dat[1:]

    for i in range(dat.shape[0]):
        # px_coords = truth_cords[i+offset] / 100
        # im = np.zeros((64, 64, 3))
        # for coord in px_coords:
        #     n_x = int(coord[0])
        #     n_y = int(coord[1])
        #     r_x = coord[0] - n_x
        #     r_y = coord[1] - n_y
        #     im[OFFSET+n_x, OFFSET+n_y, 0] = 100
        #     im[OFFSET+n_x, OFFSET+n_y, 1] = r_x-0.5
        #     im[OFFSET+n_x, OFFSET+n_y, 2] = r_y-0.5
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(data[i,:,:,1])
        # axs[1].imshow(im[:,:,1])
        # plt.show()
        yield data[i+ offset]#,  im[:,:,:], px_coords


def crop_generator_saved_file():
    data = np.load(get_root_path() +r"\crop_dataset_train_VS.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"\crop_dataset_truth_VS.npy", allow_pickle=True).astype(np.float32)
    for i in range(data.shape[0]):
        yield data[i], truth[i]

def crop_generator_saved_file_coords():
    data = np.load(get_root_path() +r"\crop_dataset_train.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"\crop_dataset_truth.npy", allow_pickle=True).astype(np.float32)
    for i in range(data.shape[0]):
        coords = []
        current = truth[i]
        for j in range(current.shape[0]):
            page = current[j]
            indices = np.array(np.where(page[ :, :, 2] == 1))
            per_image= -np.ones((3,2))
            for i,ind in enumerate(indices.T):
                per_image[i] = ind + np.array([page[ind[0],ind[1], 0], page[ind[0],ind[1], 1]])
            coords.append(np.array(per_image))
        yield data[i], np.array(coords)


if __name__ == '__main__':
    g = crop_generator_saved_file()
    for data in g:
        x=0


