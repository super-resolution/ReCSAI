import tensorflow as tf
from tifffile import TiffFile as TIF
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from src.factory import Factory
import copy
from astropy.convolution import Gaussian2DKernel
from src.utility import get_root_path

CROP_TRANSFORMS = 4

#todo: data generator creation factory
class DataGeneratorFactory():
    def __init__(self, path):
        self.path = path

    @property
    def shape(self):
        data = np.load(get_root_path() + r"/datasets" + self.path + "/coordinates.npy",
                allow_pickle=True).astype(np.float32)
        return ((data.shape[1],9,9,3),(data.shape[1],9,9,3),data.shape[1:],(data.shape[1],9,9,4),())

    def __call__(self, *args, **kwargs):
        data_path = get_root_path() + r"/datasets" + self.path

        def generator():
            data = np.load(data_path + "/train.npy",
                           allow_pickle=True).astype(np.float32)
            truth = np.load(data_path + "/truth.npy",
                            allow_pickle=True).astype(np.float32)
            noiseless = np.load(data_path + "/noiseless.npy",
                                allow_pickle=True).astype(np.float32)
            coords = np.load(data_path + "/coordinates.npy",
                             allow_pickle=True).astype(np.float32)
            sigma = np.load(data_path + "/sigma.npy",
                            allow_pickle=True).astype(np.float32)
            # todo: coords to pixel coords
            coords[:, :, :, 3] /= 0.001 + coords[:, :, :, 3].max()
            for i in range(data.shape[0]):
                c = coords[i]
                yield data[i]/data[i].max(), noiseless[i]/noiseless[i].max(), c, truth[i], sigma[i]
        return generator

def build_switching_array(n):
    bef_after = np.random.randint(0, 2, 2 * n)
    test = np.ones((n, 3))
    test2 = np.zeros((n, 3))
    test[:, 0] = bef_after[0:n]
    test[:, 2] = bef_after[n:]
    for i in range(n):
        if test[i, 0] == 1 and test[i, 2] == 0:
            test2[i, 0] = 1  # switching on
            test2[i, 1] = 3  # switching off
        elif test[i, 2] == 1 and test[i, 0] == 0:
            test2[i, 1] = 1  # switching on
            test2[i, 2] = 3  # switching off
        elif test[i, 2] == 1 and test[i, 0] == 1:
            test2[i, 0] = 1  # switching on
            test2[i, 1] = 2  # switching on
            test2[i, 2] = 3  # switching off
        else:
            test2[i, 1] = 2
    return test2


def crop_generator_u_net(im_shape, sigma_x=150, sigma_y=150,size=100, seed=0, noiseless_ground_truth=False):
    #todo: create dynamical
    mean_loc_count = 1.5#2
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        for z in range(100):
            ph = np.random.randint(200,1000)#todo: add photons to dataset was 500,1500
            points = factory.create_crop_point_set(photons=ph, on_time=30)
            #sigma_y = np.random.randint(100, 250)
            sig_y = sigma_x+np.random.rand()*10-5
            sig_x = sigma_y+np.random.rand()*10-5
            factory.kernel_type = "Airy"
            factory.kernel = (sig_x, sig_y)
            truth_cs_list = []
            image_list = []
            image_noiseless_list = []
            co=[]
            for i in range(size): #todo: while loop here
                print(i)

                n = int(np.random.normal(mean_loc_count, 0.2*mean_loc_count))#np.random.poisson(1.7)

                if n<0:
                    n=0
                ind = np.random.randint(0, points.shape[0] - n)

                #if n>3:
                #    n=3
                def build_image(ind, switching_array,):#todo: points to image additional parameters sigma and intensity
                    image_s = np.zeros((im_shape, im_shape, 3))
                    image_noiseless = np.zeros((im_shape, im_shape, 3))
                    local_bg= np.random.choice(a=[True, False], size=1, p=[0.1,0.9])[0]#activated recently
                    for i in range(3):

                        image,gt,on_points = factory.simulate_accurate_flimbi(points, points[ind], switching_rate=0, inverted=switching_array[:,i])
                        image_noiseless[:,:, i] = factory.reduce_size(gt).astype(np.float32)

                        image = factory.reduce_size(image).astype(np.float32)

                        #local bg simulation:
                        if local_bg:
                            image += np.random.rand()*5+15 #noise was 2

                        #image_noiseless[:,:,i] = copy.deepcopy(image)
                        image = factory.accurate_noise_simulations_camera(image).astype(np.float32)
                        #plt.scatter(on_points[:,1]/100,on_points[:,0]/100)
                        # plt.imshow(image)
                        # plt.show()

                        image_s[:,:,i] = image
                        if i == 1:
                            truth_cs = factory.create_classifier_image((im_shape, im_shape), on_points,
                                                                       100)  # todo: variable px_size
                            co.append(on_points)

                    return image_s, truth_cs, image_noiseless

                #todo: build named tuple (i.e. true true flase)
                #todo: set switching accordingly(i.e. true, inverted=true, false
                ind = np.arange(ind, ind + n, 1).astype(np.int32)#todo: set switching true
                switching_array = build_switching_array(n)

                image_s,truth_cs, image_noiseless = build_image(ind, switching_array)

                #image_s -= image_s.min()#todo: skip normalization
                #image_s += 0.0001
                # image_s /= image_s.max()
                # image_s *= np.random.rand()*0.3+0.7
                #fig,axs = plt.subplots(3)
                #axs[0].imshow(truth_cs[:,:,0])
                #axs[1].imshow(truth_cs[:,:,1])
                #axs[2].imshow(truth_cs[:,:,2])

                #plt.show()
                #if image_noiseless.max() >0:
                #    image_noiseless /= image_noiseless.max()#todo: no normalization
                truth_cs_list.append(truth_cs)
                image_list.append(image_s)
                image_noiseless_list.append(image_noiseless)
            if noiseless_ground_truth:
                current = np.array(truth_cs_list)
                coords = []
                for j in range(current.shape[0]):
                    page = current[j]
                    indices = np.array(np.where(page[:, :, 2] == 1))
                    per_image = np.zeros((10, 4))#todo first size was 10
                    for k, ind in enumerate(indices.T):
                        c = ind + np.array(
                            [page[ind[0], ind[1], 0], page[ind[0], ind[1], 1]]) + 0.5  # this is probably wrong!
                        per_image[k, 0:2] = c
                        per_image[k, 2] = 1
                        per_image[k, 3] = co[j][k][2]
                    coords.append(np.array(per_image))#todo: not used to build tensor
                yield tf.convert_to_tensor(image_list, dtype=tf.float32),  tf.convert_to_tensor(image_noiseless_list, dtype=tf.float32), \
                      tf.convert_to_tensor(coords, dtype=tf.float32), tf.convert_to_tensor(truth_cs_list, dtype=tf.float32),# todo: change in create data
            else:
                yield tf.convert_to_tensor(image_list), tf.convert_to_tensor(truth_cs_list)#todo: shuffle?
    return generator

def crop_generator(im_shape, sigma_x=150, sigma_y=150):
    #todo: create dynamical

    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        for z in range(100):
            ph = np.random.randint(1000,2000)
            points = factory.create_crop_point_set(photons=ph)
            #sigma_y = np.random.randint(100, 250)
            factory.kernel = Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y)
            points_list = []
            image_list = []
            for i in range(100): #todo: while loop here
                print(i)

                ind = np.random.randint(0,points.shape[0])
                n = int(np.random.normal(1.5,0.4,1))#np.random.poisson(1.7)
                if n>3:
                    n=3
                def build_image(ind):
                    image = factory.create_image()
                    p = np.zeros(9)
                    for z in range(n):
                        p[6+z] = 1
                    p[0:ind.shape[0]*2] = points[ind,0:2].flatten()
                    image = factory.create_points_add_photons(image, points[ind], points[ind,2])
                    image = factory.reduce_size(image).astype(np.float32)
                    image += 3
                    return image, p

                ind = np.arange(ind, ind + n, 1).astype(np.int32)
                image_s = np.zeros((im_shape,im_shape, 3))

                image_s[:, :, 1],p = build_image(ind)

                bef_after = np.random.randint(0,2,2*n)
                ind_new_b = ind
                ind_new_a = ind
                ind_new_b = np.delete(ind_new_b, np.where(bef_after[:n]==0))
                ind_new_a = np.delete(ind_new_a, np.where(bef_after[n:]==0))
                image_s[:, :, 2],_ = build_image(ind_new_a)
                image_s[:, :, 0],_ = build_image(ind_new_b)


                #done: random new noise in next image random switch off
                for t in range(CROP_TRANSFORMS):
                    image_s_copy = copy.deepcopy(image_s)
                    p_n = copy.deepcopy(p)
                    for k in range(3):
                        image_s_copy[:,:,k] = factory.accurate_noise_simulations_camera(image_s_copy[:,:,k])

                    image_s_copy -= image_s_copy.min()
                    image_s_copy /= image_s_copy.max()
                    if t == 1:
                        image_s_copy = np.fliplr(image_s_copy)
                        p_n[1:6:2] = ((factory.shape[1] ) - p_n[1:6:2])*p[6:]
                    elif t == 2:
                        image_s_copy = np.flipud(image_s_copy)
                        p_n[0:6:2] = ((factory.shape[0] - 1) - p_n[0:6:2])*p[6:]
                    elif t == 3:
                        image_s_copy = np.flipud(np.fliplr(image_s_copy))
                        p_n[1:6:2] = ((factory.shape[1] - 1) - p_n[1:6:2])*p[6:]
                        p_n[0:6:2] = ((factory.shape[0] - 1) - p_n[0:6:2])*p[6:]
                    points_list.append(p_n)
                    image_list.append(image_s_copy)


            yield tf.convert_to_tensor(image_list), tf.convert_to_tensor(np.array(points_list))#todo: shuffle?
    return generator

def real_data_generator(im_shape, switching_rate=0.2):
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)# select points here
    def generator():
        points = factory.create_point_set()
        init_indices = np.random.choice(points.shape[0], 10)
        on_points = points[init_indices]
        for i in range(10000): #todo: while loop here
            print(i)
            image, truth, on_points = factory.simulate_accurate_flimbi(points, on_points, switching_rate=switching_rate)#todo: simulate off
            image = factory.reduce_size(image)#todo: background base lvl?
            image += 1/3*image.max()
            image*=3
            image = np.pad(factory.accurate_noise_simulations_camera(image),(14,14))
            truth = np.pad(factory.reduce_size(truth).astype(np.int32),(14,14))
            yield image, truth, np.array(on_points)
    return generator


def generate_generator(image):#todo needs image size
    if image.shape[1]>128:
        offset = int((256 - image.shape[1]) / 2)
    else:
        offset = int((128 - image.shape[1]) / 2)  # pad to 128
    batch_size = 2000
    def data_generator_real():
        if image.shape[0]% batch_size == 0:
            batches_count = image.shape[0]//batch_size
        else:
            batches_count = 1+image.shape[0]// batch_size
        print(batches_count)

        for i in range(batches_count):
            dat = image[i * batch_size:(i + 1) * batch_size,]#14:-14,14:-14]
            # dat = dat[:dat.shape[0]//4*4]
            #dat = dat[::4] + dat[1::4] + dat[2::4] + dat[3::4] #todo shift ungerade
            # dat[:, 1::2] = scipy.ndimage.shift(dat[:,1::2], (0,0,0.5))
            #dat[:,1::2,1:] = dat[:,1::2,:-1]
            dat -= dat.min()
            data = np.ones((dat.shape[0], dat.shape[1]+2*offset, dat.shape[1]+2*offset, 3))*np.mean(dat)  # todo: this is weird data
            data[:, offset:offset + dat.shape[1], offset:offset + dat.shape[2], 1] = dat
            data[1:, offset:offset + dat.shape[1], offset:offset + dat.shape[2], 0] = dat[:-1]
            data[:-1, offset:offset + dat.shape[1], offset:offset + dat.shape[2], 2] = dat[1:]
            yield data
    return data_generator_real, offset

def data_generator_coords(file_path, offset_slice=0):
    with TIF(file_path) as tif:
        dat = tif.asarray()
    offset = int((128 - dat.shape[0]) / 2 ) # pad to 128

    data = np.zeros((dat.shape[0],128, 128, 3))
    data[:, offset:offset + dat.shape[1], offset:offset + dat.shape[2], 1] = dat
    data[1:, offset:offset + dat.shape[1], offset:offset + dat.shape[2], 0] = dat[:-1]
    data[:-1, offset:offset + dat.shape[1], offset:offset + dat.shape[2], 2] = dat[1:]

    for i in range(dat.shape[0]):
        # px_coords = truth_cords[i+offset] / 100
        # im = np.zeros((64, 64, 3))
        # for coord in px_coords:
        #     n_x = int(coord[0])
        #     n_y = int(coord[1])
        #     r_x = coord[0] - n_x
        #     r_y = coord[1] - n_y
        #     im[OFFSET+n_x, OFFSET+n_y, 0] = 100
        #     im[OFFSET+n_x, OFFSET+n_y, 1] = r_x-0.5
        #     im[OFFSET+n_x, OFFSET+n_y, 2] = r_y-0.5
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(data[i,:,:,1])
        # axs[1].imshow(im[:,:,1])
        # plt.show()
        yield data[i+ offset_slice]#,  im[:,:,:], px_coords


def crop_generator_saved_file():
    data = np.load(get_root_path() +r"\crop_dataset_train_VS.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"\crop_dataset_truth_VS.npy", allow_pickle=True).astype(np.float32)
    for i in range(data.shape[0]):
        yield data[i], truth[i]

def crop_generator_saved_file_EX():
    data = np.load(get_root_path() +r"\crop_dataset_train.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"\crop_dataset_truth.npy", allow_pickle=True).astype(np.float32)
    noiseless = np.load(get_root_path() +r"\crop_dataset_noiseless_VS.npy", allow_pickle=True).astype(np.float32)

    for i in range(data.shape[0]):
        yield data[i], truth[i], noiseless[i]

def crop_generator_saved_file_coords():
    data = np.load(get_root_path() +r"/crop_dataset_train_VS_1000.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"/crop_dataset_truth_VS_1000.npy", allow_pickle=True).astype(np.float32)
    noiseless = np.load(get_root_path() +r"/crop_dataset_noiseless_VS_1000.npy", allow_pickle=True).astype(np.float32)

    for i in range(data.shape[0]):
        coords = []
        current = truth[i]
        for j in range(current.shape[0]):
            page = current[j]
            indices = np.array(np.where(page[ :, :, 2] == 1))
            per_image= np.zeros((10,3))
            for k,ind in enumerate(indices.T):
                c = ind + np.array([page[ind[0],ind[1], 0], page[ind[0],ind[1], 1]])+0.5#this is probably wrong!
                per_image[k,0:2] = c
                per_image[k, 2] = 1
            coords.append(np.array(per_image))
        nl = noiseless[i]/(0.001+np.amax(noiseless[i],(1,2,3),keepdims=True))
        yield data[i], nl, np.array(coords), truth[i]

def crop_generator_save_file_wavelet():
    data = np.load(get_root_path() +r"/current_dataset/dataset_train_wavelet_1000_Airy.npy", allow_pickle=True).astype(np.float32)
    noiseless = np.load(get_root_path() +r"/current_dataset/dataset_noiseless_wavelet_1000_Airy.npy", allow_pickle=True).astype(np.float32)
    for i in range(data.shape[0]):
        yield data[i], noiseless[i]

def crop_generator_saved_file_specific():
    data = np.load(get_root_path() +r"/crop_dataset_train_1.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"/crop_dataset_truth_1.npy", allow_pickle=True).astype(np.float32)
    noiseless = np.load(get_root_path() +r"/crop_dataset_noiseless_1.npy", allow_pickle=True).astype(np.float32)
    coords = np.load(get_root_path() +r"/crop_dataset_coordinates_1.npy", allow_pickle=True).astype(np.float32)
    #todo: coords to pixel coords
    for i in range(data.shape[0]):
        c = coords[i]
        c[:,:,3]/=0.001+c[:,:3].max()
        yield data[i], noiseless[i], c, truth[i]

def crop_generator_saved_file_coords_airy():
    data = np.load(get_root_path() +r"/current_dataset/crop_dataset_train_VS_1000_Airy.npy", allow_pickle=True).astype(np.float32)
    truth = np.load(get_root_path() +r"/current_dataset/crop_dataset_truth_VS_1000_Airy.npy", allow_pickle=True).astype(np.float32)
    noiseless = np.load(get_root_path() +r"/current_dataset/crop_dataset_noiseless_VS_1000_Airy.npy", allow_pickle=True).astype(np.float32)
    coords = np.load(get_root_path() +r"/current_dataset/crop_dataset_coordinates_VS_1000_Airy.npy", allow_pickle=True).astype(np.float32)
    sigma = np.load(get_root_path() + r"/current_dataset/crop_dataset_sigma_VS_1000_Airy.npy",
                    allow_pickle=True).astype(np.float32)

    #todo: coords to pixel coords
    coords[:,:,:,3] /= 0.001+coords[:,:,:,3].max()
    for i in range(data.shape[0]):
        c = coords[i]
        yield data[i], noiseless[i], c, truth[i], sigma[i]

if __name__ == '__main__':
    g = crop_generator_saved_file()
    for data in g:
        x=0


