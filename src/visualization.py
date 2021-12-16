import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from tifffile.tifffile import TiffWriter

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def display_storm_data(data_in):
    #todo: show first and second half for FRC
    #data_in = data_in[np.where(data_in[:,2]<data_in[:,2].max()/3)]
    #data_in = data_in[1::2]

    print(data_in.shape[0])
    localizations = data_in[:,0:2]
    localizations *= 100
    array = np.zeros((int(localizations[:,0].max())+1, int(localizations[:,1].max())+1))#create better rendering...
    for i in range(localizations.shape[0]):
        array[int(localizations[i,0]),int(localizations[i,1])] += (200/data_in[i,4])*data_in[i,5]
    array = cv2.GaussianBlur(array, (21, 21), 0)
    array -= 10
    array = np.clip(array,0,255)
    downsampled = cv2.resize(array, (int(array.shape[0]/10),int(array.shape[1]/10)), interpolation=cv2.INTER_AREA)
    with TiffWriter('tmp/temp.tif', bigtiff=True) as tif:
        tif.save(downsampled)
    cm = plt.get_cmap('hot')
    v = cm(downsampled/255)
    v[:,:,3] =255
    with TiffWriter('tmp/temp2.tif', bigtiff=True) as tif:
        tif.save(v)
    #todo: save array...
    #array = np.log(array+1)
    plt.imshow(array, cmap='hot')
    plt.show()
    


if __name__ == '__main__':
    import os
    from Thunderstorm_jaccard import read_Thunderstorm
    import pandas as pd
    data = pd.read_csv(r"C:\Users\biophys\PycharmProjects\TfWaveletLayers\test_data\thunderstorm_results_Cy5.csv").as_matrix()
    display_storm_data(data[:,1:]/100)