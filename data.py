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