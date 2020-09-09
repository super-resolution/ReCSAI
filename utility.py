import numpy as np
from .data import *
from .localisations import *


def bin_localisations(data_tensor, denoising, truth_tensor=None, th=8.0):
    train_new = []
    truth_new = []
    coord_list = []
    bining = Binning()

    one = denoising.predict(data_tensor[:, :, :, 0:1])
    two = denoising.predict(data_tensor[:, :, :, 1:2])
    three = denoising.predict(data_tensor[:, :, :, 2:3])
    im = tf.concat([one, two, three], -1)

    for i in range(im.shape[0]):
        # fig,ax = plt.subplots(1)
        # todo: denoise all at once

        wave = im[i:i + 1, :, :, 1:2]
        y = tf.constant([th])
        mask = tf.greater(wave, y)
        wave_mask = wave * tf.cast(mask, tf.float32)
        # ax.imshow(wave[0,:,:,0])
        coords = bining.get_coords(wave_mask.numpy()[0, :, :, 0])
        # todo: crop PSFs done here
        for coord in coords:
            # ax.add_patch(rect)
            # ax.set_title("original", fontsize=10)
            if coord[0] - 4 > 0 and coord[1] - 4 > 0 and coord[0] + 4 < 64 and coord[1] + 4 < 64:
                crop = im[i, coord[0] - 4:coord[0] + 5, coord[1] - 4:coord[1] + 5, :]
                train_new.append(crop)
                if truth_tensor is not None:
                    truth = truth_tensor[i:i + 1, coord[0] - 1:coord[0] + 2, coord[1] - 1:coord[1] + 2, :]
                    ind = tf.argmax(tf.reshape(truth[0, :, :, 0], [-1]))
                    x = ind // truth.shape[2]
                    y = ind % truth.shape[2]
                    x_f = tf.cast(x, tf.float64)
                    y_f = tf.cast(y, tf.float64)
                    x_f += truth[0, x, y, 1] + 3  # add difference to crop dimension
                    y_f += truth[0, x, y, 2] + 3
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


def simulate_where_add(input, lo_lim, up_lim, val):
    condition = tf.logical_and(input>lo_lim,input<up_lim)
    indices_t = tf.where(condition)[:,0]
    indices_f = tf.where(tf.logical_not(condition))[:,0]

    values_remove = tf.tile([0], [tf.shape(indices_f)[0]])
    values_remove = tf.cast(values_remove, tf.float64)

    value = tf.gather(input, indices_t)
    value += val

    zeros_remove = tf.zeros_like(indices_f)
    zeros_keep = tf.zeros_like(indices_t)
    idx_remove = tf.stack([zeros_remove, indices_f],axis=1)
    idx_keep = tf.stack([zeros_keep, indices_t], axis=1)

    out = tf.sparse.add(tf.SparseTensor(idx_remove, values_remove, tf.shape(input[tf.newaxis,:], out_type = tf.int64)),
                        tf.SparseTensor(idx_keep, value, tf.shape(input[tf.newaxis,:], out_type = tf.int64)))
    return tf.sparse.to_dense(out)[0,:]

def get_psf(sigma, px_size):
    size = 256 # should always be enough

    sigma_x = sigma/(px_size*64)
    sigma_y = sigma_x
    psf = np.zeros((size, size))
    lattice = 512 - 1
    for i in range(psf.shape[0]):
        for j in range(psf.shape[1]):
            normed_i = (i / lattice) ** 2
            normed_j = (j / lattice) ** 2
            psf[i, j] = 1 / (lattice ** 2) * 1 / (np.sqrt(2 * np.pi * sigma_x ** 2) * np.sqrt(2 * np.pi * sigma_y ** 2)) \
                        * np.exp(-(normed_i / (2 * sigma_x ** 2) + normed_j / (2 * sigma_y ** 2)))
    return psf

def create_psf_block(idy, size_x, magnification, psf):
    size_y = size_x*magnification
    #create row for one block in y direction
    blockA = np.zeros((size_y, size_x))
    for i in range(blockA.shape[0]):
        for j in range(blockA.shape[1]):
            try:
                blockA[i, j] = psf[idy, abs(i - magnification * j)]
            except IndexError:
                print(i, j)
    return blockA

def create_psf_matrix(size_x, magnification):
    size_y = size_x*magnification
    matrix = np.zeros((size_y**2, size_x**2))
    psf = get_psf(100, 100)
    for i in range(size_x):
        for j in range(size_y):
            matrix[j * size_y:(j + 1) * size_y, i * size_x:(i + 1) * size_x] = create_psf_block(abs(magnification * i - j), size_x, magnification, psf)
    return matrix

