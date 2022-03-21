#from src.data import *
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from scipy.ndimage import filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from time import time
from functools import wraps

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} args:[{args}{kw}] took: {te-ts} sec')
        return result
    return wrap


def result_image_to_coordinates(result_tensor, coord_list=None, threshold=0.2, sigma_thresh=0.1):
    result_array = []
    test = []
    for i in range(result_tensor.shape[0]):
        classifier =result_tensor[i, :, :, 2]

        #classifier = uniform_filter(result_tensor[i, :, :, 2], size=3) * 9
        # plt.imshow(classifier)
        # plt.show()
        # classifier = result_tensor[i, :, :, 2]
        if np.sum(classifier) > threshold:
            classifier[np.where(classifier < threshold)] = 0
            # indices = np.where(classifier>self.threshold)

            indices = get_coords(classifier).T
            x = result_tensor[i, indices[0], indices[1], 0]
            y = result_tensor[i, indices[0], indices[1], 1]
            p = result_tensor[i, indices[0], indices[1], 2]
            dx = result_tensor[i, indices[0], indices[1], 3]  # todo: if present
            dy = result_tensor[i, indices[0], indices[1], 4]
            N = result_tensor[i, indices[0], indices[1], 5]

            for j in range(indices[0].shape[0]):
                #if dx[j] < sigma_thresh and dy[j] < sigma_thresh:
                    if coord_list:
                        result_array.append(np.array([coord_list[i][0] + float(indices[0][j]) + (x[j])
                                                         , coord_list[i][1] + float(indices[1][j]) + y[j], coord_list[i][2],
                                                      dx[j], dy[j], N[j], p[j]]))
                        test.append(np.array([x[j], y[j]]))
                    else:
                        result_array.append(np.array([float(indices[0][j]) + (x[j])
                                                    ,float(indices[1][j]) + y[j], i,
                                                      dx[j], dy[j], N[j], p[j]]))
                        test.append(np.array([x[j], y[j]]))
    print(np.mean(np.array(test), axis=0))
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

def FRC_loss(i1, i2):
    dim = min(i1.shape[0], i1.shape[1], i2.shape[0], i2.shape[1])
    if dim % 2 != 0:
        dim -= 1
    i1 = i1[np.newaxis, 0:dim, 0:dim, np.newaxis]
    i2 = i2[np.newaxis, 0:dim, 0:dim, np.newaxis]
    size = dim
    size_half = size // 2

    r = np.zeros([size])
    r[:size_half] = np.arange(size_half) + 1
    r[size_half:] = np.arange(size_half, 0, -1)

    c = np.zeros([size])
    c[:size_half] = np.arange(size_half) + 1
    c[size_half:] = np.arange(size_half, 0, -1)

    [R, C] = np.meshgrid(r, c)

    help_index = np.round(np.sqrt(R ** 2 + C ** 2))
    kernel_list = []

    for i in range(1, 102):
        new_matrix = np.zeros(shape=[size, size])
        new_matrix[help_index == i] = 1
        kernel_list.append(new_matrix)

    kernel_list = tf.constant(kernel_list, dtype=tf.complex64)

    i1 = tf.squeeze(i1, axis=0)
    i1 = tf.squeeze(i1, axis=-1)

    i2 = tf.squeeze(i2, axis=0)
    i2 = tf.squeeze(i2, axis=-1)

    i1 = tf.cast(i1, dtype=tf.complex64)
    i2 = tf.cast(i2, dtype=tf.complex64)

    I1 = tf.signal.fft2d(i1)
    I2 = tf.signal.fft2d(i2)

    A = tf.multiply(I1, tf.math.conj(I2))
    B = tf.multiply(I1, tf.math.conj(I1))
    C = tf.multiply(I2, tf.math.conj(I2))

    A_val = tf.reduce_mean(tf.multiply(A, kernel_list), axis=(1, 2))
    B_val = tf.reduce_mean(tf.multiply(B, kernel_list), axis=(1, 2))
    C_val = tf.reduce_mean(tf.multiply(C, kernel_list), axis=(1, 2))

    res = tf.abs(A_val) / tf.sqrt(tf.abs(tf.multiply(B_val, C_val)))

    return 1.0 - tf.reduce_sum(res) / 102.0




def bin_localisations_v2(data_tensor, denoising, truth_array=None, th=0.1):
    train_new = []
    truth_new = []
    coord_list = []
    #data_tensor = data_tensor/tf.keras.backend.max(data_tensor)
    data = tf.cast(data_tensor, tf.float32).numpy()
    data /= tf.keras.backend.max(data)

    one = denoising.predict(data_tensor[:, :, :, 0:1])
    two = denoising.predict(data_tensor[:, :, :, 1:2])
    three = np.nan_to_num(denoising.predict(data_tensor[:, :, :, 2:3]))
    im = tf.concat([one, two, three], -1)
    #im /= tf.keras.backend.max(im)
    #data_tensor /= tf.keras.backend.max(data_tensor)
    for i in range(im.shape[0]):
        # todo: denoise all at once
        current_data = data[i, :, :, :]/tf.keras.backend.max(data[i, :, :, :])
        #current_data = tf.repeat(current_data,3, axis=-1)

        wave = im[i:i + 1, :, :, 1:2]/tf.keras.backend.max(im[i:i + 1, :, :, 1:2])
        y = tf.constant([th])
        mask = tf.greater(wave, y)
        wave_mask = wave * tf.cast(mask, tf.float32)
        c_data_masked = current_data*tf.cast(tf.greater(current_data, y), tf.float32)
        # fig, axs = plt.subplots(3)
        # axs[0].imshow(wave_mask[0, :, :, 0])
        # axs[1].imshow(data[i, :, :, 1])
        # axs[2].imshow(im[i, :, :, 1])
        # plt.show()
        #ax.imshow(wave[0,:,:,0])
        #coords = get_coords(c_data_masked.numpy()[ :, :, 1])

        coords = get_coords(wave_mask.numpy()[0, :, :, 0])
        # from src.visualization import plot_wavelet_bin_results
        # plot_wavelet_bin_results(current_data[:,:,1].numpy(), wave_mask.numpy()[0, :, :, 0], coords)
        # todo: crop PSFs done here
        for coord in coords:
            # ax.add_patch(rect)
            # ax.set_title("original", fontsize=10)
            if coord[0] - 4 > 0 and coord[1] - 4 > 0 and coord[0] + 4 < im.shape[-2] and coord[1] + 4 < im.shape[-2]:
                crop = current_data[ coord[0] - 4:coord[0] + 5, coord[1] - 4:coord[1] + 5, :]#todo: set current data
                crop = crop-tf.keras.backend.min(crop)
                #crop = crop/tf.keras.backend.max(crop)

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
    print(len(train_new))
    train_new = tf.stack(train_new)
    truth_new = tf.stack(truth_new)
    return train_new, truth_new, np.array(coord_list)

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

    sigma_x = sigma*8#neglecting coarse and.. resolution
    sigma_y = sigma_x
    psf = np.zeros((size, size))#todo: somewhere factor 2 missing!
    #psf_tf = tf.zeros((size,size))
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

def create_psf_block(idy, size_x, psf,size_y, dy):
    #create row for one block in y direction
    blockA = np.zeros((size_y, size_x))
    for i in range(blockA.shape[0]):

        for j in range(blockA.shape[1]):
            dx = int(np.around(7 * (j / size_x) - 3))  # todo fix

            try:
                blockA[i, j] = psf[abs(idy+dy), abs(i - size_x * j+dx)]
            except IndexError:
                print(i, j)
    return blockA

def create_psf_matrix(size_x, magnification, psf):
    size_y = size_x*magnification
    matrix = np.zeros((size_y**2, size_x**2)).astype(np.float32)
    #todo: interpolate -4 to +4 in range of size... over siz x indices
    for i in range(size_x):
        dy = int(np.around(7*(i/size_x)-3))#0.5 pixel offset to sublattice 1px=8points in sublattice -1 one for pixel in sublatice
        for j in range(size_y):
            matrix[j * size_y:(j + 1) * size_y, i * size_x:(i + 1) * size_x] = create_psf_block(abs(size_x * i - j), size_x, psf, size_y, dy)
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


def read_thunderstorm_drift_json(path):
    import json
    from scipy.interpolate import CubicSpline
    with open(path, 'r') as f:
        data = json.load(f)

    def get_knots_drift(name):
        knots = data[name]["knots"]
        drift = []
        polynom = data[name]['polynomials']

        for poly in polynom:
            coeff = poly["coefficients"]
            drift.append(coeff[0])
        drift.append(coeff[0] + coeff[1] * (knots[-1] - knots[-2] - 1))
        return knots, drift

    knots_x, drift_x = get_knots_drift("xFunction")

    x = np.arange(knots_x[0], knots_x[-1] + 1)
    poly_x = CubicSpline(knots_x, drift_x)
    x_drift = poly_x(x)
    knots_y, drift_y = get_knots_drift("yFunction")
    poly_y = CubicSpline(knots_y, drift_y)
    y_drift = poly_y(x)
    return np.stack([x_drift, y_drift],axis=-1)

def read_thunderstorm_drift(path):
    import pandas as pd
    from scipy.interpolate import interp1d
    #done: read json
    data = pd.read_csv(path, sep=",")
    frames_x = data['X2'].values
    frames_y = data['X3'].values
    drift_x = data['Y2'].values
    drift_y = data['Y3'].values
    x_pol = interp1d(frames_x, drift_x, kind='cubic')
    y_pol = interp1d(frames_y, drift_y, kind='cubic')
    X = np.arange(1, 4800)#todo: image shape
    Y = np.arange(1, 4800)
    driftx_n = x_pol(X)
    drifty_n = y_pol(Y)
    return np.stack([driftx_n, drifty_n],axis=-1)



    #todo: pick X2 Y2 X3 Y3

    #todo: interpolate linear between entries

    #todo: return drift per frame

    pass

def get_reconstruct_coords(tensor, th, neighbors=3):
    import copy
    filter = np.ones((neighbors,neighbors))
    filter[0::2,0::2] = 0
    tensor = copy.deepcopy(tensor)
    convolved = scipy.ndimage.convolve(tensor, filter)
    #todo: if convolved >1 pick maximum
    #todo: if convolved > 2 pick both

    indices = np.where(np.logical_and(tensor > th, convolved >2*th))
    x = []
    y = []
    for i in range(indices[0].shape[0]):
        ind_x_min = indices[0][i]-1
        ind_y_min = indices[1][i]-1
        if ind_x_min<0:
            ind_x_min = 0
        if ind_y_min < 0:
            ind_y_min = 0
        t = tensor[ind_x_min:indices[0][i]+2, ind_y_min:indices[1][i]+2]
        max_ind = np.where(t==t.max())
        x.append(max_ind[0][0] + indices[0][i]-1)
        y.append(max_ind[1][0] + indices[1][i]-1)

    indices = np.unique(np.array((np.array(x).astype(np.int), np.array(y).astype(np.int))),axis=1)
    return indices
    #todo: where convolved > 1.5 threshold

def get_coords(reconstruct, neighbors=5):
    neighborhood = np.ones((neighbors,neighbors)).astype(np.bool)
    #create cross structure
    #if neighbors ==3:
        #neighborhood[0::2,0::2] = False
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
    #fig,axs = plt.subplots(2)

    #axs[0].imshow(detected_peaks)
    #axs[1].imshow(reconstruct)
    #plt.show()
    coords = np.array(np.where(detected_peaks != 0))
    return coords.T

def get_root_path():
    return str(Path(__file__).parent.parent)