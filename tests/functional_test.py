import tensorflow as tf
from src.factory import Factory
from src.custom_layers.cs_layers import CompressedSensing, CompressedSensingInception
from src.utility import get_root_path
from astropy.convolution import Gaussian2DKernel
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from src.models.cs_model import CompressedSensingInceptionNet, CompressedSensingCVNet, CompressedSensingConvNet
from src.data import crop_generator_saved_file_coords
from src.trainings import train_cs_net
from src.facade import NetworkFacade
from tests.create_test_datasets import TestDatasets

CURRENT1 = get_root_path()+r"/trainings/cs_cnn/_final_training_100_100"
CURRENT2 = get_root_path()+r"/trainings/cs_inception/_final_10_100"



# #todo: functional_test
def create(im_shape):
    factory = Factory()
    factory.kernel_type="Airy"

    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)
    ph = np.random.randint(5000, 30000)
    points = factory.create_crop_point_set(photons=ph)
    sigma_x = 150
    sigma_y = 150
    factory.kernel = (sigma_x, sigma_y)


    ind = np.random.randint(0, points.shape[0])
    n = 2  # np.random.poisson(1.7)
    image = factory.create_image()
    image = factory.create_points_add_photons(image, points[ind:ind + n], points[ind:ind + n, 2])
    print(points[ind:ind + n]*73/(10*100))
    image = factory.reduce_size(image)
    image = factory.accurate_noise_simulations_camera(image)
    return image, points[ind:ind + n]
#
# def test_output():
#     #todo: useful for debugging might be outsourced to display?
#     #done input bigger tensor
#     crop = create(9)
#     crop_new = np.zeros((crop.shape[0], crop.shape[1],3))
#     for i in range(3):
#         crop_new[:,:,i] = crop
#     crop = crop_new.astype(np.float32)
#     crop/=crop.max()
#
#     #crop = np.load(os.getcwd() + r"\crop.npy")
#     layer = CompressedSensing()
#     crop_tensor = tf.constant((crop),dtype=tf.float64)
#     im = tf.stack([crop_tensor, crop_tensor])
#     y = layer(im)
#     x = layer(im)
#     fig,axs = plt.subplots(3)
#     y = tf.reshape(y, (-1, 73,73,3))
#     x = tf.reshape(x, (-1, 73,73,3))
#
#     c_spline = interpolate.interp2d(np.arange(0,9,1), np.arange(0,9,1), crop[:,:,1], kind='cubic')
#
#     new = c_spline(np.arange(0,9,0.125),np.arange(0,9,0.125))
#
#     axs[0].imshow(x[0,:,:,1])
#     axs[1].imshow(y[0,:,:,1])
#     axs[2].imshow(new)
#     plt.show()
#     x=0
    #done: load file
    #done: run layer
class ViewLayerOutputs():
    def __init__(self):
        self.data_fac = TestDatasets(9)


        self.facade_hcs = NetworkFacade(CompressedSensingCVNet, CURRENT1,
                                                get_root_path()+r"/trainings/wavelet/training_lvl2/cp-10000.ckpt")
        self.facade_hcs.threshold = 0.15  # todo: still has artefacts...
        self.facade_hcs.sigma_thresh = 0.3
        self.facade_hcs.photon_filter = 0.1
        self.network1 = self.facade_hcs.network
        self.network1.sigma = 150

        self.facade_lcs = NetworkFacade(CompressedSensingInceptionNet, CURRENT2,
                                                get_root_path()+r"/trainings/wavelet/training_lvl2/cp-10000.ckpt")
        self.network2 = self.facade_lcs.network
        #self.network2.inception1.cs.iterations=1
        #self.network2.inception2.cs.iterations=1

        self.network2.sigma = 150


    def cs_inception_path_output(self):
        crop, points = create(9)
        crop_new = np.zeros((crop.shape[0], crop.shape[1], 3))
        for i in range(3):
            crop_new[:, :, i] = crop
        crop = crop_new.astype(np.float32)
        crop /= crop.max()
        crop_tensor = tf.constant((crop), dtype=tf.float64)
        im = tf.stack([crop_tensor, crop_tensor])
        #cs_out = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear', )(im)
        out,cs_out = self.network1.inception1(im)
        t = self.network1.inception1.cs.conv2D(cs_out)
        #print(self.network.inception1.cs.lam)
        #test = tf.reshape(tf.linalg.matvec(tf.transpose(self.network1.inception1.cs.mat),tf.reshape(cs_out[0,:,:,1],5184), ),(9,9))
        #plt.imshow(test/tf.reduce_max(test,axis=[0,1], keepdims=True))
        #plt.show()

        fig, axs = plt.subplots(3,4)

        axs[0][0].imshow(cs_out[0,:,:,1])
        axs[0][1].imshow(t[0,:,:,1])
        axs[0][2].imshow(out[0,:,:,2])
        axs[0][3].imshow(out[0,:,:,3])

        axs[1][0].imshow(out[0,:,:,4])
        axs[1][1].imshow(out[0,:,:,5])
        axs[1][2].imshow(out[0,:,:,6])
        axs[1][3].imshow(out[0,:,:,7])

        axs[2][0].imshow(out[0,:,:,8])
        axs[2][1].imshow(out[0,:,:,9])
        #axs[2][2].imshow(tf.reshape(test,(9,9)))

        axs[2][3].imshow(im[0,:,:,1])
        plt.show()

    def cs_conv(self):
        crop, points = create(9)
        crop_new = np.zeros((crop.shape[0], crop.shape[1], 3))
        for i in range(3):
            crop_new[:, :, i] = crop
        crop = crop_new.astype(np.float32)
        crop /= crop.max()
        crop_tensor = tf.constant((crop), dtype=tf.float64)
        im = tf.stack([crop_tensor, crop_tensor])
        cs_out = tf.keras.layers.Conv2DTranspose(3, (1,1), strides=(8,8),padding="same", use_bias=False)(im)
        #out,cs_out = self.network1.inception1(im)
        #t = self.network1.inception1.cs.conv2D(cs_out)
        #print(self.network.inception1.cs.lam)
        #test = tf.reshape(tf.linalg.matvec(tf.transpose(self.network1.inception1.cs.mat),tf.reshape(cs_out[0,:,:,1],5184), ),(9,9))
        #plt.imshow(test/tf.reduce_max(test,axis=[0,1], keepdims=True))
        #plt.show()

        fig, axs = plt.subplots(2,3)

        axs[0][0].imshow(cs_out[0,:,:,1])
        axs[0][1].imshow(cs_out[0,:,:,2])
        axs[0][2].imshow(cs_out[0,:,:,0])

        axs[1][0].imshow(im[0,:,:,1])

        # axs[0][2].imshow(out[0,:,:,2])
        # axs[0][3].imshow(out[0,:,:,3])
        #
        # axs[1][0].imshow(out[0,:,:,4])
        # axs[1][1].imshow(out[0,:,:,5])
        # axs[1][2].imshow(out[0,:,:,6])
        # axs[1][3].imshow(out[0,:,:,7])
        #
        # axs[2][0].imshow(out[0,:,:,8])
        # axs[2][1].imshow(out[0,:,:,9])
        #axs[2][2].imshow(tf.reshape(test,(9,9)))

        plt.show()

    def noiseless_gt_output_loss(self):
        crop, points = create(9)
        plt.imshow(crop)
        plt.scatter(points[:,1]/100-0.5, points[:,0]/100-0.5)
        plt.show()
        crop_new = np.zeros((crop.shape[0], crop.shape[1], 3))
        for i in range(3):
            crop_new[:, :, i] = crop
        crop = crop_new.astype(np.float32)
        crop /= crop.max()
        crop_tensor = tf.constant((crop), dtype=tf.float64)
        im = tf.stack([crop_tensor, crop_tensor])
        out, cs_out = self.network.inception1(im)
        #        print(self.network.inception1.cs.lam)
        test = tf.reshape(
            tf.linalg.matvec(tf.transpose(self.network.inception1.cs.mat), tf.reshape(cs_out[0, :, :, 1], 5184), ),
            (9, 9))
        self.network.compute_loss_decode()


    def plot_facade_generator_output(self):
        dataset = tf.data.Dataset.from_generator(self.facade_hcs.data_factory(), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=self.facade_hcs.data_factory.shape)
        for image, noiseless, coords, truth,_ in dataset.take(10):
            for i in range(1):
                i+=299
                c_image = image[i,:,:,1]
                c_noiseless = noiseless[i,:,:,1]
                c_coords = coords[i]

                fig,axs = plt.subplots((2))
                axs[0].imshow(c_image)
                axs[1].imshow(c_noiseless)
                axs[0].scatter(c_coords[:,1],c_coords[:,0])
                axs[1].scatter(c_coords[:,1],c_coords[:,0])
                plt.show()


    def plot_generator_output(self):
        sigma = np.load(get_root_path() + r"/crop_dataset_sigma.npy", allow_pickle=True).astype(np.float32)
        # dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_EX, (tf.float32, tf.float32, tf.float32),
        #                                         output_shapes=((1 * 100, 9, 9, 3), (1 * 100, 9, 9, 4),(1 * 100, 9, 9, 3)))
        dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_coords, (tf.float32, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=((1 * 1000, 9, 9, 3),(1*1000, 9, 9, 3), (1 * 1000, 10, 3),(1*1000, 9, 9, 4) ))
        for image, noiseless, coords, truth in dataset.take(3):
            for i in range(100):
                c_image = image[i,:,:,1]
                c_noiseless = noiseless[i,:,:,1]
                c_coords = coords[i]

                fig,axs = plt.subplots((2))
                axs[0].imshow(c_image)
                axs[1].imshow(c_noiseless)
                axs[0].scatter(c_coords[:,1],c_coords[:,0])
                axs[1].scatter(c_coords[:,1],c_coords[:,0])
                plt.show()

    def t_perfect_reconstruction_loss_is_zero(self):
        dataset = tf.data.Dataset.from_generator(crop_generator_saved_file_coords, (tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(
                                                 (1 * 1000, 9, 9, 3), (1 * 1000, 9, 9, 3), (1 * 1000, 10, 3),
                                                 (1 * 1000, 9, 9, 4)))
        for train_image, noiseless,truth_c, truth_i in dataset.take(3):
            test = truth_i.numpy()
            r = [0,200]
            x= truth_c.numpy()[r[0]:r[1]]
            new = np.ones((test.shape[0],test.shape[1],test.shape[2],6))*1/(20*np.pi)
            new[:,:,:,0:3] = test[:,:,:,0:3]+0.00001
            new[:,:,:,2] += 0.01
            new = new[r[0]:r[1]]
            if np.all(new==1):
                print("all one test failed")
            res = self.network.compute_loss_decode_ncs(tf.constant(x, tf.float32), tf.constant(new, tf.float32))#use truth image instead of predict
            print(res)

    def distance_test(self):
        im, points = self.data_fac.create_decreasing_distance()
        #cs_out = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear', )(im)
        out = self.network1.predict(im)

        c = [np.array([0,0,i]) for i in range(5)]
        coords = self.facade_hcs.get_localizations_from_image_tensor(out, c)

        out = self.network2.predict(im)
        coords_lcs = self.facade_hcs.get_localizations_from_image_tensor(out, c)

        self.plot(coords, coords_lcs, im, points)


    def noise_test(self):
        im, points = self.data_fac.create_increasing_noise()
        #cs_out = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear', )(im)
        out = self.network1.predict(im)

        c = [np.array([0,0,i]) for i in range(5)]
        coords = self.facade_hcs.get_localizations_from_image_tensor(out, c)

        out = self.network2.predict(im)
        coords_lcs = self.facade_hcs.get_localizations_from_image_tensor(out, c)

        self.plot(coords, coords_lcs, im, points)


    def lifetime_test(self):
        im, points = self.data_fac.create_decreasing_lifetime()
        dataset = im[:,:,:,1].numpy()
        import pandas as pd

        thunderstorm = pd.read_csv('test_data/crops/lifetime_evaluation_thunderstorm.csv').as_matrix()
        th_data = []
        frame=1
        current = []
        for i in thunderstorm:
            if i[0] == frame:
                current.append(i[1:3]/100-0.5)
            else:
                frame = i[0]
                th_data.append(np.array(current))
                current=[]
                current.append(i[1:3]/100-0.5)
        th_data.append(np.array(current))

        #from tifffile import TiffWriter, TiffFile
        # with TiffWriter('tmp/temp2.tif', bigtiff=True) as tif:
        #     tif.save(dataset)
        #todo: safe tensor for thunderstorm evaluation
        #cs_out = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear', )(im)
        out = self.network1.predict(im)

        c = [np.array([0,0,i]) for i in range(5)]
        coords = self.facade_hcs.get_localizations_from_image_tensor(out, c)

        out = self.network2.predict(im)
        coords_lcs = self.facade_hcs.get_localizations_from_image_tensor(out, c)
        self.plot(coords, coords_lcs, im, points, th=th_data)

    def plot(self, coords, coords_lcs, im, points, th=None):
        fig, axs = plt.subplots(2,5)
        #
        for i in range(5):
            co = coords[np.where(coords[:,2]==i)]
            co_lcs = coords_lcs[np.where(coords_lcs[:,2]==i)]

            axs[0][i].imshow(im[i,:,:,1], cmap="gray")
            try:
                axs[0][i].scatter(th[i][:,0],th[i][:,1],marker="x", c="b", label="thunderstorm")
            except:
                pass
            if i == 0:
                axs[0][i].scatter(co[:,1],co[:,0],marker="x", c="r", label="predict")
                axs[0][i].scatter(points[i][:,1]/100-0.5,points[i][:,0]/100-0.5,marker="+", c="g", label="truth")
            else:
                axs[0][i].scatter(co[:,1],co[:,0],marker="x", c="r")
                axs[0][i].scatter(points[i][:,1]/100-0.5,points[i][:,0]/100-0.5,marker="+", c="g")
            axs[0][i].set_yticklabels([])
            axs[0][i].set_xticklabels([])
            axs[1][i].imshow(im[i,:,:,1],cmap="gray")
            axs[1][i].scatter(co_lcs[:,1],co_lcs[:,0],marker="x", c="r")
            axs[1][i].scatter(points[i][:,1]/100-0.5,points[i][:,0]/100-0.5,marker="+", c="g")
            axs[1][i].set_yticklabels([])
            axs[1][i].set_xticklabels([])

        fig.legend(loc='center')
        plt.show()

if __name__ == '__main__':
    V = ViewLayerOutputs()
    V.plot_facade_generator_output()
