import tensorflow as tf
from quicktiff import TiffFile
import numpy as np
from localisations import *

def data_generator(file_path, loc_path):
    bin = Binning()
    tif = TiffFile(file_path)
    truth_cords = np.load(loc_path, allow_pickle=True)['arr_0']
    for i in range(len(tif.frames)):
        data = np.zeros((64, 64, 3))
        for j in range(3):
            k = i+j-1
            if k>=0 and k<len(tif.frames):
                image = tif.read_frame(k, 0)[0].astype(np.float32)
                data[0:tif.frames[0].tags[256][1], 0:tif.frames[0].tags[256][1], j] = image
        reconstruct = bin.filter(data[...,1]/data[...,1].max()*255)#todo: crop image to coords!
        coords = bin.get_coords(reconstruct)
        yield data[...,1], coords, truth_cords[i][:,0:2]/100

def data_generator_coords(file_path, loc_path):
    tif = TiffFile(file_path)
    truth_cords = np.load(loc_path, allow_pickle=True)['arr_0']


    for i in range(len(tif.frames)):
        px_coords = truth_cords[i] / 100
        im = np.zeros((64, 64, 3))
        for coord in px_coords:
            n_x = int(coord[0])
            n_y = int(coord[1])
            r_x = coord[0] - n_x
            r_y = coord[1] - n_y
            im[n_x, n_y, 0] = 100
            im[n_x, n_y, 1] = r_x
            im[n_x, n_y, 2] = r_y

        data = np.zeros((64, 64, 3))
        for j in range(3):
            k = i+j-1
            if k>=0 and k<len(tif.frames):
                image = tif.read_frame(k, 0)[0].astype(np.float32)
                data[0:tif.frames[0].tags[256][1], 0:tif.frames[0].tags[256][1], j] = image
        yield data,  im[:,:,:], px_coords

def data_generator_image(file_path, truth_path):
    tif = TiffFile(file_path)
    truth = TiffFile(truth_path)
    for i in range(len(tif.frames)):
        data = np.zeros((64, 64))
        data_truth = np.zeros((64,64))
        image = tif.read_frame(i, 0)[0].astype(np.float32)
        image_truth = truth.read_frame(i, 0)[0].astype(np.float32)
        data[0:tif.frames[0].tags[256][1], 0:tif.frames[0].tags[256][1]] = image
        data_truth[0:tif.frames[0].tags[256][1], 0:tif.frames[0].tags[256][1]] = image_truth
        data = (data)/data.max()
        data_truth = (data_truth)/data_truth.max()
        yield data, data_truth