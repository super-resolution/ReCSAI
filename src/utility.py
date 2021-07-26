#from src.data import *
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from scipy.ndimage import filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter


def result_image_to_coordinates(result, coords=None, threshold=0.3):
    result_array = []
    for i in range(result.shape[0]):
        classifier = result[i, :, :, 2]
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(classifier)
        # axs[1].imshow(crop_tensor[i,:,:,1])
        # plt.show()
        classifier = gaussian_filter(classifier, sigma=0.6, truncate=1) * 3/2
        classifier[np.where(classifier<0.2)] = 0
        indices = get_coords(classifier).T

        #indices = np.where(classifier > threshold)
        x = result[i, indices[0], indices[1], 0]
        y = result[i, indices[0], indices[1], 1]
        #data = []
        for j in range(indices[0].shape[0]):
            c = np.array([indices[0][j] + x[j], indices[1][j] + y[j]])
            #data.append(c)
            if coords:
                result_array.append(coords[i][0:2] + c)
            else:
                result_array.append(np.array([indices[0][j] + x[j], indices[1][j] + y[j], i]))

    return np.array(result_array)

def predict_sigma(psf_crops, result_array, ai):
    #perfect_result_array = []
    perfect_psf_array = []
    print("evaluated")
    indices = tf.where(result_array[:,2]<1.3)
    #x = psf_crops[indices[:,0]]
    for i in range(result_array.shape[0]):
        #print(i)
        if result_array[i,2]>0.95 and result_array[i,2]<1.25:
            #perfect_result_array.append(result_array[i,0:2])
            perfect_psf_array.append(psf_crops[i,:,:,1].numpy())
    perfect_psfs = np.transpose(np.array(perfect_psf_array),(1,2,0))
    x=[]
    for i in range(perfect_psfs.shape[2]//10):
       x.append(ai.predict(tf.convert_to_tensor(perfect_psfs[np.newaxis,:,:,i*10:10+i*10])))
    x = np.array(x)
    print(np.mean(x,axis=0))

def extrude_perfect_psf(psf_crops, result_array):
    perfect_result_array = []
    perfect_psf_array = []
    for i in range(result_array.shape[0]):
        if result_array[i,2]>0.95 and result_array[i,2]<1.3:
            perfect_result_array.append(result_array[i,0:2])
            perfect_psf_array.append(psf_crops[i])
    coords = np.array(perfect_result_array)
    perfect_psfs = np.array(perfect_psf_array)

    full_psf = np.zeros((72,72))
    for i in range(perfect_psfs.shape[0]):
        psf = perfect_psfs[i]
        point = coords[i]
        new = scipy.ndimage.shift(psf[:, :, 1], (4.5-point[1]/8, 4.5-point[0]/8))

        c_spline = interpolate.interp2d(np.arange(0, 9, 1), np.arange(0, 9, 1), new, kind='cubic')

        full_psf += c_spline(np.arange(0, 9, 0.125), np.arange(0, 9, 0.125))
    plt.imshow(full_psf)
    plt.show()
    return full_psf[32:,32:]/np.sum(full_psf[32:,32:])
    #todo: return psf and use as matrix

def bin_localisations_v2(data_tensor, denoising, truth_array=None, th=0.1):
    train_new = []
    truth_new = []
    coord_list = []
    data = tf.cast(data_tensor, tf.float32).numpy()
    data /= tf.keras.backend.max(data)

    one = denoising.predict(data_tensor[:, :, :, 0:1])
    two = denoising.predict(data_tensor[:, :, :, 1:2])
    three = np.nan_to_num(denoising.predict(data_tensor[:, :, :, 2:3]))
    im = tf.concat([one, two, three], -1)
    im /= tf.keras.backend.max(im)
    #data_tensor /= tf.keras.backend.max(data_tensor)
    for i in range(im.shape[0]):
        # todo: denoise all at once
        current_data = data[i, :, :, :]/tf.keras.backend.max(data[i, :, :, :])

        wave = im[i:i + 1, :, :, 1:2]/tf.keras.backend.max(im[i:i + 1, :, :, 1:2])
        y = tf.constant([th])
        mask = tf.greater(wave, y)
        wave_mask = wave * tf.cast(mask, tf.float32)
        # fig, axs = plt.subplots(3)
        # axs[0].imshow(wave_mask[0, :, :, 0])
        # axs[1].imshow(im[i, :, :, 1])
        # axs[2].imshow(im[i, :, :, 2])
        # plt.show()
        #ax.imshow(wave[0,:,:,0])
        coords = get_coords(wave_mask.numpy()[0, :, :, 0])
        # todo: crop PSFs done here
        for coord in coords:
            # ax.add_patch(rect)
            # ax.set_title("original", fontsize=10)
            if coord[0] - 4 > 0 and coord[1] - 4 > 0 and coord[0] + 4 < im.shape[-2] and coord[1] + 4 < im.shape[-2]:
                crop = current_data[ coord[0] - 4:coord[0] + 5, coord[1] - 4:coord[1] + 5, :]#todo: set current data
                crop = crop-tf.keras.backend.min(crop)
                crop = crop/tf.keras.backend.max(crop)

                #np.save(os.getcwd() + r"\crop.npy", crop)
                #
                # fig, axs = plt.subplots(3)
                # axs[0].imshow(crop[ :, :, 0])
                # axs[1].imshow(crop[ :, :, 1])
                # axs[2].imshow(crop[ :, :, 2])
                # plt.show()
                if truth_array is not None:
                    # todo: implement recursive nonlinear drift estimator
                    # todo: is coord[0] gerade or ungerade
                    truth = truth_array[i]
                    ind = np.where(np.logical_and(
                        np.logical_and(truth[:,0]>coord[0]-4,truth[:,1]>coord[1]-4),
                        np.logical_and(truth[:,0]<coord[0]+5,truth[:,1]<coord[1]+5)))
                    if ind[0].shape[0] !=0:
                        x = (truth[ind[0][0],0]-coord[0]+4)*72/73*8
                        y = (truth[ind[0][0],1]-coord[1]+4)*72/73*8
                        truth_new.append(tf.stack([x, y, ind[0].shape[0]]))
                        train_new.append(crop)
                        coord_list.append(np.append(coord - 4, i))
                    else:
                        truth_new.append(tf.stack(np.array([0.0, 0.0, 0.0],dtype=np.float64)))
                        train_new.append(crop)
                        coord_list.append(np.append(coord - 4, i))
                        # fig, axs = plt.subplots()
                        # axs.imshow(crop[:,:,1])
                        # axs.scatter(y,x)
                        # plt.show()

                else:
                    train_new.append(crop)
                    coord_list.append(np.append(coord - 4, i))

                # todo: calc coordinates why?
    train_new = tf.stack(train_new)
    truth_new = tf.stack(truth_new)
    return train_new, truth_new, coord_list

def create_shift_data(data_tensor, denoising, truth_array=None, th=0.1):
    train_new = []
    truth_new = []
    coord_list = []
    t = tf.identity(data_tensor)

    one = denoising.predict(data_tensor[:, :, :, 0:1])
    two = denoising.predict(data_tensor[:, :, :, 1:2])
    three = denoising.predict(data_tensor[:, :, :, 2:3])
    im = tf.concat([one, two, three], -1)
    im = im/tf.keras.backend.max(im)

    for i in range(im.shape[0]):
        # todo: denoise all at once

        wave = im[i:i + 1, :, :, 1:2]
        y = tf.constant([th])
        mask = tf.greater(wave, y)
        wave_mask = wave * tf.cast(mask, tf.float32)
        # fig, axs = plt.subplots(3)
        # axs[0].imshow(data_tensor[i, :, :, 0])
        # axs[1].imshow(data_tensor[i, :, :, 1])
        # axs[2].imshow(data_tensor[i, :, :, 2])
        # plt.show()
        #ax.imshow(wave[0,:,:,0])
        coords = get_coords(wave_mask.numpy()[0, :, :, 0])
        # todo: crop PSFs done here
        for coord in coords:
            # ax.add_patch(rect)
            # ax.set_title("original", fontsize=10)
            if coord[0] - 4 > 0 and coord[1] - 4 > 0 and coord[0] + 4 < 64 and coord[1] + 4 < 64:
                crop = t[i, coord[0] - 4:coord[0] + 5, coord[1] - 4:coord[1] + 5, :]
                td = tf.unstack(crop, axis=-1)
                if coord[0]%2==0:
                    td[0] = tf.ones_like(td[0])
                else:
                    td[0] = tf.zeros_like(td[0])
                crop = tf.stack(td, axis=-1)

                # fig, axs = plt.subplots(3)
                # axs[0].imshow(crop[ :, :, 0])
                # axs[1].imshow(crop[ :, :, 1])
                # axs[2].imshow(crop[ :, :, 2])
                # plt.show()
                #train_new.append(crop)
                if truth_array is not None:
                    # todo: implement recursive nonlinear drift estimator
                    # todo: is coord[0] gerade or ungerade
                    truth = truth_array[i]
                    ind = np.where(np.logical_and(
                        np.logical_and(truth[:,0]>coord[0]-2,truth[:,1]>coord[1]-2),
                        np.logical_and(truth[:,0]<coord[0]+2,truth[:,1]<coord[1]+2)))
                    if ind[0].shape[0] !=0:
                        x = (truth[ind[0][0],0]-coord[0]+4)*72/73*8
                        y = (truth[ind[0][0],1]-coord[1]+4)*72/73*8
                        truth_new.append(tf.stack([x, y,truth[ind[0][0],2],truth[ind[0][0],3],truth[ind[0][0],4]]))
                        train_new.append(crop)
                        coord_list.append(coord - 4)

    train_new = tf.stack(train_new)
    truth_new = tf.stack(truth_new)
    return train_new, truth_new, coord_list










def get_psf(sigma, px_size):
    size = 256 # should always be enough

    sigma_x = sigma/81*8#neglecting coarse and.. resolution
    sigma_y = sigma_x
    psf = np.zeros((size, size))#todo: somewhere factor 2 missing!
    psf_tf = tf.zeros((size,size))
    # tf_sigma_x = sigma_x
    # tf_sigma_y = sigma_y
    # i = tf.cast(tf.square(tf.range(size)), tf.float32)
    # j = tf.cast(tf.square(tf.range(size)), tf.float32)
    # I,J = tf.meshgrid(i,j)
    # PI = tf.constant(np.pi)
    # psf_tf = 1  / (tf.sqrt(2.0 * PI * tf_sigma_x ** 2.0) * tf.sqrt(2.0 * PI * tf_sigma_y ** 2.0)) \
    #                     * tf.exp(-(I/ (2.0 * tf_sigma_x ** 2.0) + J / (2.0 * tf_sigma_y** 2.0)))

    skaling_fine = 1/71
    skaling_coarse = 1/8

    for i in range(psf.shape[0]):
        for j in range(psf.shape[1]):#todo: normed j is not an even number...
            normed_i = (i ) ** 2
            normed_j = (j ) ** 2
            psf[i, j] = 1  / (np.sqrt(2 * np.pi * sigma_x ** 2) * np.sqrt(2 * np.pi * sigma_y ** 2)) \
                        * np.exp(-(normed_i / (2 * sigma_x ** 2) + normed_j / (2 * sigma_y ** 2)))
    return psf

def create_psf_block(idy, size_x, psf,size_y):
    #create row for one block in y direction
    blockA = np.zeros((size_y, size_x))
    for i in range(blockA.shape[0]):
        for j in range(blockA.shape[1]):
            try:
                blockA[i, j] = psf[idy, abs(i - size_x * j)]
            except IndexError:
                print(i, j)
    return blockA

def create_psf_matrix(size_x, magnification, psf):
    size_y = size_x*magnification
    matrix = np.zeros((size_y**2, size_x**2)).astype(np.float32)
    for i in range(size_x):
        for j in range(size_y):
            matrix[j * size_y:(j + 1) * size_y, i * size_x:(i + 1) * size_x] = create_psf_block(abs(size_x * i - j), size_x, psf, size_y)
    return matrix

def old_psf_matrix(M_x,M_y,N_x,N_y,sigma_x,sigma_y):
    m = M_x*M_y    # Anzahl der Gitterpunkte des groben Gitters
    N = N_x*N_y
    if sigma_x == None and sigma_y == None:
        sigma_x = 5/N
        sigma_y = 5/N
        a_x=1
        a_y=1
    else:
        a_x=1
        a_y=1

    h_M_x = a_x/(M_x-1)  # Gitterweite des groben Gitters in x-Richtung
    h_M_y = a_y/(M_y-1)  # Gitterweite des groben Gitters in y-Richtung
    h_N_x = a_x/(N_x-1)  # Gitterweite des feinen Gitters in
    h_N_y = a_y/(N_y-1)  # x- bzw. y-Richtung
    A = np.zeros((m,N))

    # Die Punkte der Gitter werden spaltenweise als Vektor untereinandergehängt
    # und umbenannt: grobe-Gitter-Matrix U wird zu Vektor y der Länge m,
    # feine-Gitter-Matrix V wird zu Vektor x der Länge N
    K = np.zeros([m,2]) # die i-te Zeile von K entspricht den ursprünglichen Zeilen-
                        # und Spaltenindizes der i-ten Komponente von y
    L = np.zeros([N,2]) # die i-te Zeile von L entspricht den ursprünglichen Zeilen-
                        # und Spaltenindizes der i-ten Komponente von x
    #x_K = [(M_y-1),-1:0]# Vektor für zweite Spalte von K
    #x_L = [(N_y-1),-1:0]# Vektor für zweite Spalte von L

    x_K = np.arange(M_y-1,-1,-1)
    x_L = np.arange(N_y-1,-1,-1)

    # Gauß-Kern
    def gaus_kernel(x,y):
        return h_N_x*h_N_y*1/(np.sqrt(2*np.pi*sigma_x**2)*np.sqrt(2*np.pi*sigma_y**2))*np.exp(-(x**2/(2*sigma_x**2)+y**2/(2*sigma_y**2)))


    for i in range(M_x):
        y_K=i*np.ones(M_y) # todo: 1+???
        K[(i*M_y):(M_y+i*M_y),0]=y_K    # in der ersten Spalte erhöht sich der Wert
                                           # eines Elements nach M Zeilen um 1
        K[(i*M_y):(M_y+i*M_y),1]=x_K    # in der zweiten Spalte erscheint M-mal die
                                           # Zahlenfolge M-1:-1:0
    for i in range(N_x):
        y_L=i*np.ones(N_y)
        L[(i*N_y):(N_y+i*N_y),0]=y_L # in der ersten Spalte erhöht sich der Wert
                                        # eines Elements nach N_y Zeilen um 1
        L[(i*N_y):(N_y+i*N_y),1]=x_L # in der zweiten Spalte erscheint N_x-mal die
                                        # Zahlenfolge N_y-1:-1:0
    for i in range(m):   # i-te Zeile
        for j in range(N):   # j-te Spalte
            A[i,j] = gaus_kernel(K[i,0]*h_M_x-L[j,0]*h_N_x,  K[i,1]*h_M_y-L[j,1]*h_N_y)
    return A

def get_coords(reconstruct):
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = filters.maximum_filter(reconstruct, footprint=neighborhood) == reconstruct
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (reconstruct == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    coords = np.array(np.where(detected_peaks != 0))
    return coords.T

def get_root_path():
    return str(Path(__file__).parent.parent)