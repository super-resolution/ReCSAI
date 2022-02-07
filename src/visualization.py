import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from tifffile.tifffile import TiffWriter
import copy

def plot_wavelet_bin_results(data, wave, coords):
    def plot_rect(axes, coord, size=9):
        for ax in axes:
            rect = plt.Rectangle((coord[1]-size/2, coord[0]-size/2), 9, 9,  facecolor="none", ec='r', lw=2)
            ax.add_patch(rect)
            #ax.scatter([1], [2], s=36, color="k", zorder=3)
    fig, axs = plt.subplots(2)
    axs[0].imshow(wave)
    axs[1].imshow(data)
    for coord in coords:
        plot_rect(axs, coord)
    plt.show()

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def render(data_in, thunderstorm=False):
    localizations = copy.deepcopy(data_in[:,0:2])#+np.random.random((data_in.shape[0],2))
    localizations *= 100
    array = np.zeros((int(localizations[:,0].max())+1, int(localizations[:,1].max())+1))#create better rendering...
    for i in range(localizations.shape[0]):
        if not thunderstorm:
            array[int(localizations[i,0]),int(localizations[i,1])] += (2000)#/data_in[i,4])*data_in[i,5]
        else:
            array[int(localizations[i,0]),int(localizations[i,1])] += (2000)

    array = cv2.GaussianBlur(array, (21, 21), 0)
    #array -= 10
    array = np.clip(array,0,255)
    downsampled = cv2.resize(array, (int(array.shape[0]/10),int(array.shape[1]/10)), interpolation=cv2.INTER_AREA)
    return downsampled


def display_storm_data(data_in, thunderstorm=False):
    #todo: show first and second half for FRC
    #data_in = data_in[np.where(data_in[:,2]<data_in[:,2].max()/3)]
    #data_in = data_in[1::2]

    print(data_in.shape[0])
    localizations = data_in[:,0:2]#+np.random.random((data_in.shape[0],2))
    localizations *= 100
    array = np.zeros((int(localizations[:,0].max())+1, int(localizations[:,1].max())+1))#create better rendering...
    for i in range(localizations.shape[0]):
        if not thunderstorm:
            array[int(localizations[i,0]),int(localizations[i,1])] += (1000/data_in[i,4])*data_in[i,5]
        else:
            array[int(localizations[i,0]),int(localizations[i,1])] += (3000)

    array = cv2.GaussianBlur(array, (21, 21), 0)
    #array -= 10
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
    #todo: plot thunderstorm... and compute frc + lineprofiler
    import pandas as pd
    from src.utility import FRC_loss
    data = pd.read_csv(r"C:\Users\biophys\PycharmProjects\TfWaveletLayers\test_data\thunderstorm_results_Cy5.csv").as_matrix()
    data = data[:,(2,1,3)]/100
    print(FRC_loss(render(data[:data.shape[0]//2]),render(data[data.shape[0]//2:])))
    display_storm_data(data, thunderstorm=True)
