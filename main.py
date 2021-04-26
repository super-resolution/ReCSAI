from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet, CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.models.wavelet_model import WaveletAI
from src.data import *
from src.utility import *
from src.visualization import display_storm_data
import matplotlib.pyplot as plt
import pandas as pd
import os


#done: load wavelet checkpoints
denoising = WaveletAI()

checkpoint_path = "training_lvl2/cp-10000.ckpt"

denoising.load_weights(checkpoint_path)


def predict_localizations_u_net(path):
    cs_net = CompressedSensingInceptionNet()
    cs_net.update(180,100)
    # checkpoint_path = "cs_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint
    optimizer = tf.keras.optimizers.Adam()
    # accuracy = tf.metrics.Accuracy()
    #todo: outsource this to model
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
    manager = tf.train.CheckpointManager(ckpt, './src/trainings/cs_training_inception_increased_depth', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    drift = pd.read_csv(r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv").as_matrix()
    result_array = []
    gen = generate_generator(path)
    dataset = tf.data.Dataset.from_generator(gen, tf.float64)
    j = 0
    for image in dataset:
        # plt.imshow(image[2,:,:,0])
        # plt.show()
        # plt.imshow(image[2,:,:,1])
        # plt.show()
        crop_tensor, _, coord_list = bin_localisations_v2(image, denoising, th=0.2)
        for z in range(len(coord_list)):
            coord_list[z][2] += j * 5000
        print(crop_tensor.shape[0])
        result_tensor = cs_net.predict(crop_tensor)

        # from localisations import Binning
        # b = Binning()
        for i in range(result_tensor.shape[0]):
            thresh = 0.2
            thresh_sum = 1
            classifier = result_tensor[i,:,:,2]#todo: sum this up!
            indices = np.where(classifier>thresh)
            # if np.sum(classifier)>thresh_sum:
            #     classifier[np.where(classifier<0.1)] = 0
            #     indices = b.get_coords(classifier).T
            #     fig,axs = plt.subplots(5)
            #     axs[0].imshow(classifier)
            #     axs[1].imshow(crop_tensor[i,:,:,0])
            #
            #     axs[2].imshow(crop_tensor[i,:,:,1])
            #     axs[3].imshow(crop_tensor[i,:,:,2])
            #
            #     axs[4].imshow(result_tensor[i,:,:,1])
            #
            #     plt.show()
            x = result_tensor[i,indices[0], indices[1],0]
            y = result_tensor[i,indices[0], indices[1],1]
            #print(x,y)

            for j in range(indices[0].shape[0]):
                result_array.append(coord_list[i][0:2] + np.array([float(indices[0][j])+x[j], float(indices[1][j])+y[j]]))
            #todo: non maximum supression



        del result_tensor
        j += 1
    result_array = np.array(result_array)
    # result_array[:,0] += 45
    print(result_array.shape[0])

    print("finished AI")
    display_storm_data(result_array)
    np.save(os.getcwd() + r"\cy5_flim_inception.npy", result_array)


def predict_localizations(path):
    cs_net = CompressedSensingNet()
    cs_net.update(122,100)

    # checkpoint_path = "cs_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint
    optimizer = tf.keras.optimizers.Adam()
    # accuracy = tf.metrics.Accuracy()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
    manager = tf.train.CheckpointManager(ckpt, './cs_training_permute_loss_downsample', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    drift = pd.read_csv(r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv").as_matrix()
    result_array = []
    gen = generate_generator(path)
    dataset = tf.data.Dataset.from_generator(gen, tf.float64)
    j = 0
    for image in dataset:
        # plt.imshow(image[2,:,:,0])
        # plt.show()
        # plt.imshow(image[2,:,:,1])
        # plt.show()
        crop_tensor, _, coord_list = bin_localisations_v2(image, denoising, th=0.2)
        for z in range(len(coord_list)):
            coord_list[z][2] += j*5000
        print(crop_tensor.shape[0])
        result_tensor = cs_net.predict(crop_tensor)
        fig, axs = plt.subplots(3)
        axs[0].imshow(crop_tensor[100,:,:,0])
        axs[0].scatter(result_tensor[100,1]/8,result_tensor[100,0]/8)
        axs[1].imshow(crop_tensor[100,:,:,1])
        axs[1].scatter(result_tensor[100,1]/8,result_tensor[100,0]/8)
        axs[2].imshow(crop_tensor[100,:,:,2])
        axs[2].scatter(result_tensor[100,1]/8,result_tensor[100,0]/8)
        plt.show()
        # del crop_tensor
        frame=0
        for i in range(result_tensor.shape[0]):
                current_drift = drift[int(coord_list[i][2]*0.4),1:3]
                #current_drift[1] *= -1
#                if coord_list[i][2] == frame:
                #limit = [4,1,0.3]
                limit = [0.8, 0.6, 0.5]
                for n in range(3):
                    #if result_tensor[i,2*n]/8 >1 and result_tensor[i,2*n+1]/8>1:
                    if result_tensor[i, 6 + n] > limit[n]:
                        result_array.append(coord_list[i][0:2] + np.array([result_tensor[i,2*n]/8, result_tensor[i,2*n+1]/8]))
                # else:
                #     frame +=1
                #     result_array = np.array(result_array)
                #     plt.scatter(result_array[:,0],result_array[:,1])
                #     plt.show()
                #     result_array = []

        del result_tensor
        j+=1
    result_array = np.array(result_array)
    #result_array[:,0] += 45
    print(result_array.shape[0])

    print("finished AI")
    display_storm_data(result_array)
    np.save(os.getcwd()+r"\cy5_flim_dense.npy",result_array)






#validate_cs_model()
#train_cs_net()
#train_nonlinear_shifter_ai()
#learn_psf()



image = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#image = r"D:\Daten\AI\COS7_LT1_beta-tub-Alexa647_new_D2O+MEA5mM_power6_OD0p6_3_crop.tif"
#image = r"C:\Users\biophys\matlab\test2_crop_BP.tif"
#image = r"D:\Daten\Artificial\ContestHD.tif"
#image = r"D:\Daten\Domi\origami\201203_10nM-Trolox_ScSystem_50mM-MgCl2_kA_TiRF_568nm_100ms_45min_no-gain-10MHz_zirk.tif"
class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, './src/trainings/cs_training_inception_increased_depth',
                                                r"C:\Users\biophys\PycharmProjects\TfWaveletLayers\training_lvl2\cp-10000.ckpt")
facade = TrainInceptionNet()
facade.sigma = 180
result_array = facade.predict(image)[:,0:2]
print(result_array.shape[0])
print("finished AI")
display_storm_data(result_array)
predict_localizations_u_net(image)

