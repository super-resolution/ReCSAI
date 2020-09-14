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
    array = cv2.GaussianBlur(array, (5, 5), 0)
    array = np.log(array+1)
    plt.imshow(array, cmap='hot')
    plt.show()
    
def distance(reconstruction,ground_truth):
    """
    define false positive: localisation without ground truth for distance > 200nm
    define false negative: ground truth without localisation for distance > 200nm
    :param ground_truth:
    :param reconstruction:
    """
    false_positive = []
    result = []
    this_ground_truth = copy.deepcopy(ground_truth)
    for i in range(reconstruction.shape[0]):
        distance = 200
        current_j = -1
        for j in range(this_ground_truth.shape[0]):
            #dis = np.linalg.norm(reconstruction[i] - this_ground_truth[j])
            if np.linalg.norm(reconstruction[i] - this_ground_truth[j]) < distance:
                distance = np.linalg.norm(reconstruction[i] - this_ground_truth[j])
                current_j = j
                vec = reconstruction[i] - this_ground_truth[j]
        if current_j != -1:
            result.append(np.array([*reconstruction[i],*vec]))
            this_ground_truth = np.delete(this_ground_truth, current_j, axis=0)
        else:
            false_positive.append(reconstruction[i])
    result = np.array(result)
    false_positive = np.array(false_positive)
    false_negative = this_ground_truth
    jac = result.shape[0] / (result.shape[0] + false_positive.shape[0] + false_negative.shape[0])
    error = 0
    for i in result:
        error += i[2] ** 2 + i[3] ** 2
    rmse = np.sqrt(error / result.shape[0])
    print("Jaccard index: = ", jac, " rmse = ", rmse)

    return result, false_positive, false_negative, jac, rmse