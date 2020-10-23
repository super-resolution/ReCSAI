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
        array[int(loc[0]),int(loc[1])] +=2500
    array = cv2.GaussianBlur(array, (7, 7), 0)
    array = np.log(array+1)
    plt.imshow(array, cmap='hot')
    plt.show()
    


if __name__ == '__main__':
    import os
    one = np.load(os.getcwd()+r"\1.npy")
    two = np.load(os.getcwd()+r"\2.npy")
    three = np.load(os.getcwd()+r"\3.npy")
    four = np.load(os.getcwd()+r"\4.npy")
    data = np.concatenate([one,two,three,four], axis=0)
    np.save(os.getcwd()+r"\data.npy", data)

    display_storm_data(data)