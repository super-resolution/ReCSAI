import tensorflow as tf
from tifffile import TiffFile as TIF
import numpy as np

OFFSET=8
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
#         yield data[...,1], coords, truth_cords[i][:,0:2]/100



def generate_generator(file_path):
    def data_generator_real():
        for i in range(3):
            with TIF(file_path) as tif:
                dat = tif.asarray()[i * 5000:(i + 1) * 5000, 0:45, 0:45]
            data = np.zeros((dat.shape[0], 64, 64, 3))  # todo: this is weird data
            data[:, OFFSET:OFFSET + dat.shape[1], OFFSET:OFFSET + dat.shape[2], 1] = dat
            data[1:, OFFSET:OFFSET + dat.shape[1], OFFSET:OFFSET + dat.shape[2], 0] = dat[:-1]
            data[:-1, OFFSET:OFFSET + dat.shape[1], OFFSET:OFFSET + dat.shape[2], 2] = dat[1:]
            yield data
    return data_generator_real

def data_generator_coords(file_path, loc_path, offset=0):
    with TIF(file_path) as tif:
        dat = tif.asarray()
    truth_cords = np.load(loc_path, allow_pickle=True)['arr_0']

    data = np.zeros((dat.shape[0],64, 64, 3))#todo: this is weird data
    data[:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],1] = dat
    data[1:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],0] = dat[:-1]
    data[:-1,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],2] = dat[1:]

    for i in range(dat.shape[0]):
        px_coords = truth_cords[i+offset] / 100
        im = np.zeros((64, 64, 3))
        for coord in px_coords:
            n_x = int(coord[0])
            n_y = int(coord[1])
            r_x = coord[0] - n_x
            r_y = coord[1] - n_y
            im[OFFSET+n_x, OFFSET+n_y, 0] = 100
            im[OFFSET+n_x, OFFSET+n_y, 1] = r_x-0.5
            im[OFFSET+n_x, OFFSET+n_y, 2] = r_y-0.5
        yield data[i+ offset],  im[:,:,:], px_coords

def data_generator_image(file_path, truth_path):
    with TIF(file_path) as tif:
        dat = tif.asarray()
    data = np.zeros((dat.shape[0],64, 64, 3))#todo: this is weird data
    data[:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],1] = dat
    data[1:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],0] = dat[:-1]
    data[:-1,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],2] = dat[1:]
    with TIF(truth_path) as tif:
        tru = tif.asarray()
    data_truth = np.zeros((dat.shape[0],64, 64, 3))#todo: this is weird data
    data_truth[:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],1] = tru
    data_truth[1:,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],0] = tru[:-1]
    data_truth[:-1,OFFSET:OFFSET+dat.shape[1], OFFSET:OFFSET+dat.shape[2],2] = tru[1:]
    for i in range(data.shape[0]):
        yield data[i], data_truth[i]