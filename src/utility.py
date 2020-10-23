from .data import *
from .localisations import *
import matplotlib.pyplot as plt
import os
from scipy import interpolate

def extrude_perfect_psf(psf_crops, result_array):
    perfect_result_array = []
    perfect_psf_array = []
    for i in range(result_array.shape[0]):
        if result_array[i,2]>0.95 and result_array[i,2]<1.05:
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
    bining = Binning()
    data = tf.cast(data_tensor, tf.float32)
    data /= tf.keras.backend.max(data)

    one = denoising.predict(data_tensor[:, :, :, 0:1])
    two = denoising.predict(data_tensor[:, :, :, 1:2])
    three = denoising.predict(data_tensor[:, :, :, 2:3])
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
        coords = bining.get_coords(wave_mask.numpy()[0, :, :, 0])
        # todo: crop PSFs done here
        for coord in coords:
            # ax.add_patch(rect)
            # ax.set_title("original", fontsize=10)
            if coord[0] - 4 > 0 and coord[1] - 4 > 0 and coord[0] + 4 < im.shape[-2] and coord[1] + 4 < im.shape[-2]:
                crop = current_data[coord[0] - 4:coord[0] + 5, coord[1] - 4:coord[1] + 5, :]#todo: append frame
                np.save(os.getcwd() + r"\crop.npy", crop)
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
    bining = Binning()
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
        coords = bining.get_coords(wave_mask.numpy()[0, :, :, 0])
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


def bin_localisations(data_tensor, denoising, truth_tensor=None, th=0.1):
    train_new = []
    truth_new = []
    coord_list = []
    bining = Binning()

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
        fig, axs = plt.subplots(3)
        axs[0].imshow(im[i, :, :, 0])
        axs[1].imshow(im[i, :, :, 1])
        axs[2].imshow(im[i, :, :, 2])
        plt.show()
        #ax.imshow(wave[0,:,:,0])
        coords = bining.get_coords(wave_mask.numpy()[0, :, :, 0])
        # todo: crop PSFs done here
        for coord in coords:
            # ax.add_patch(rect)
            # ax.set_title("original", fontsize=10)
            if coord[0] - 4 > 0 and coord[1] - 4 > 0 and coord[0] + 4 < 64 and coord[1] + 4 < 64:
                crop = data_tensor[i, coord[0] - 4:coord[0] + 5, coord[1] - 4:coord[1] + 5, :]

                fig, axs = plt.subplots(3)
                axs[0].imshow(crop[ :, :, 0])
                axs[1].imshow(crop[ :, :, 1])
                axs[2].imshow(crop[ :, :, 2])
                plt.show()
                train_new.append(crop)
                if truth_tensor is not None:
                    truth = truth_tensor[i:i + 1, coord[0] - 1:coord[0] + 2, coord[1] - 1:coord[1] + 2, :]
                    ind = tf.argmax(tf.reshape(truth[0, :, :, 0], [-1]))
                    x = ind // truth.shape[2]
                    y = ind % truth.shape[2]
                    x_f = tf.cast(x, tf.float64)#todo: care theres somthing wrong here
                    y_f = tf.cast(y, tf.float64)
                    x_f += truth[0, x, y, 1] + 4  # add difference to crop dimension
                    y_f += truth[0, x, y, 2] + 4
                    truth_new.append(tf.stack([x_f, y_f]))
                coord_list.append(coord - 4)

                # todo: calc coordinates why?
    train_new = tf.stack(train_new)
    truth_new = tf.stack(truth_new)
    return train_new, truth_new, coord_list


def generate_tf_dataset(image, truth):
    gen = data_generator_coords(image, truth)
    image = np.zeros((100, 64, 64, 3))
    truth = np.zeros((100, 64, 64, 3))

    for i in range(100):
        image[i], truth[i], _ = gen.__next__()

    image_tf1 = tf.convert_to_tensor(image[0:50, :, :])
    image_tf2 = tf.convert_to_tensor(image[50:100, :, :])
    truth_tf1 = tf.convert_to_tensor(truth[0:50, :, :])
    truth_tf2 = tf.convert_to_tensor(truth[50:100, :, :])


def simulate_where_add(input, lo_lim, up_lim, val, sparse_dense):
    condition = tf.logical_and(input>lo_lim,input<up_lim)
    indices_t = tf.where(condition)#[:,-1]
    indices_f = tf.where(tf.logical_not(condition))#[:,-1]

    values_remove = tf.tile([0], [tf.shape(indices_f)[0]])
    values_remove = tf.cast(values_remove, tf.float64)

    value = tf.gather_nd(input, indices_t)
    value += val

    idx_remove = indices_f#tf.stack([zeros_remove, indices_f],axis=1)
    idx_keep = indices_t#tf.stack([zeros_keep, indices_t], axis=1)

    #out =tf.SparseTensor(idx_keep, value, tf.shape(input, out_type=tf.int64))# sparse_add((tf.SparseTensor(idx_remove, values_remove, tf.shape(input, out_type=tf.int64)),
                        #tf.SparseTensor(idx_keep, value, tf.shape(input, out_type=tf.int64))))
    z = sparse_dense((idx_keep, value,input))
    return z


def get_psf(sigma, px_size):
    size = 256 # should always be enough

    sigma_x = sigma/(px_size)*8
    sigma_y = sigma_x
    psf = np.zeros((size, size))#todo: somewhere factor 2 missing!
    for i in range(psf.shape[0]):
        for j in range(psf.shape[1]):
            normed_i = (i ) ** 2
            normed_j = (j ) ** 2
            psf[i, j] = 1  / (np.sqrt(2 * np.pi * sigma_x ** 2) * np.sqrt(2 * np.pi * sigma_y ** 2)) \
                        * np.exp(-(normed_i / (2 * sigma_x ** 2) + normed_j / (2 * sigma_y ** 2)))
    return psf*2

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
    size_y = size_x*magnification+1
    matrix = np.zeros((size_y**2, size_x**2))
    for i in range(size_x):
        for j in range(size_y):
            matrix[j * size_y:(j + 1) * size_y, i * size_x:(i + 1) * size_x] = create_psf_block(abs(size_x * i - j), size_x, psf, size_y)
    return matrix

