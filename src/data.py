import tensorflow as tf
from tifffile import TiffFile as TIF
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from src.factory import Factory
import copy
from astropy.convolution import Gaussian2DKernel


OFFSET=14

def crop_generator(im_shape, sigma_x=150, sigma_y=150):
    #todo: create dynamical

    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        ph = np.random.randint(5000,30000)
        points = factory.create_crop_point_set(photons=ph)
        sigma_x = np.random.randint(100, 400)
        sigma_y = np.random.randint(100, 400)
        factory.kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)

        for i in range(10000): #todo: while loop here
            print(i)

            ind = np.random.randint(0,points.shape[0])
            n = int(np.random.normal(1.5,0.4,1))#np.random.poisson(1.7)
            image = factory.create_image()
            image = factory.create_points_add_photons(image, points[ind:ind+n], points[ind:ind+n,2])
            image = factory.reduce_size(image)
            image += 1/3*image.max()
            image*=3
            image = factory.accurate_noise_simulations_camera(image)
            yield image, np.array([sigma_x,sigma_y,0])
    return generator

def real_data_generator(im_shape, switching_rate=0.2):
    x_shape=100
    #todo: create dynamical

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
# def data_generator(file_path, loc_path):
#     bin = Binning()
#     tif = TiffFile(file_path)
#     truth_cords = np.load(loc_path, allow_pickle=True)['arr_0']
#     for i in range(len(tif.frames)):
#         data = np.zeros((64, 64, 3))
#         for j in range(3):
#             k = i+j-1
#             if k>=0 and k<len(tif.frames):
#                 image = tif.read_frame(k, 0)[0].astype(np.float32)
#                 data[0:tif.frames[0].tags[256][1], 0:tif.frames[0].tags[256][1], j] = image
#         reconstruct = bin.filter(data[...,1]/data[...,1].max()*255)#todo: crop image to coords!
#         coords = bin.get_coords(reconstruct)
#         yield data[...,1], coords, truth_cords[i][:,0:2]/10


def data_generator():
    #todo: simulate acurate data here
    pass

def generate_generator(file_path):
    def data_generator_real():
        for i in range(1):
            with TIF(file_path) as tif:
                dat = tif.asarray()[i * 1000:(i + 1) * 1000,14:-14,14:-14]
            #dat = dat[:dat.shape[0]//4*4]
            #dat = dat[::4] + dat[1::4] + dat[2::4] + dat[3::4] #todo shift ungerade
            #dat[:, 1::2] = scipy.ndimage.shift(dat[:,1::2], (0,0,0.5))
            #dat[:,1::2,1:] = dat[:,1::2,:-1]
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

def data_generator_image(file_path, truth_path):
    with TIF(file_path) as tif:
        dat = tif.asarray()
    data = np.zeros((dat.shape[0],128, 128, 3))#todo: this is weird data
    data[:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],1] = dat
    data[1:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],0] = dat[:-1]
    data[:-1,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],2] = dat[1:]
    with TIF(truth_path) as tif:
        tru = tif.asarray()
    data_truth = np.zeros((dat.shape[0],128, 128, 3))#todo: this is weird data
    data_truth[:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],1] = tru
    data_truth[1:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],0] = tru[:-1]
    data_truth[:-1,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],2] = tru[1:]
    for i in range(data.shape[0]):
        yield data[i], data_truth[i]