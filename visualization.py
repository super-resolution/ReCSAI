import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def display_storm_data(localizations):
    localizations *= 100
    array = np.zeros((int(localizations[:,0].max())+1, int(localizations[:,1].max())+1))
    for loc in localizations:
        array[int(loc[0]),int(loc[1])] +=255
    array = cv2.GaussianBlur(array, (5, 5), 0)
    heatmap = cv2.applyColorMap(array.astype(np.uint8), cv2.COLORMAP_HOT)
    plt.imshow(heatmap)
    plt.show()
