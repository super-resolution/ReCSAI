from src.models.cs_model import CompressedSensingNet
from src.models.wavelet_model import WaveletAI
from src.trainings.train_cs_net import train_cs_net
from src.trainings.train_wavelet_ai import train as train_wavelet_ai
from src.data import *
from src.utility import *
from src.visualization import display_storm_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd



#done: load wavelet checkpoints
denoising = WaveletAI()

checkpoint_path = "training_lvl2/cp-10000.ckpt"

denoising.load_weights(checkpoint_path)


def predict_localizations(path):
    cs_net = CompressedSensingNet()

    # checkpoint_path = "cs_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint
    optimizer = tf.keras.optimizers.Adam()
    # accuracy = tf.metrics.Accuracy()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
    manager = tf.train.CheckpointManager(ckpt, './cs_training3', max_to_keep=3)
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
        crop_tensor, _, coord_list = bin_localisations_v2(image, denoising, th=0.35)
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
                limit = [4,1,0.3]
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
    np.save(os.getcwd()+r"\DNApaint.npy",result_array)






#validate_cs_model()
#train_cs_net(crop_generator)
#train_nonlinear_shifter_ai()
#learn_psf()



image = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#image = r"D:\Daten\AI\COS7_LT1_beta-tub-Alexa647_new_D2O+MEA5mM_power6_OD0p6_3_crop.tif"
#image = r"C:\Users\biophys\matlab\test2_crop_BP.tif"
#image = r"D:\Daten\Artificial\ContestHD.tif"
#image = r"D:\Daten\Domi\origami\201203_10nM-Trolox_ScSystem_50mM-MgCl2_kA_TiRF_568nm_100ms_45min_no-gain-10MHz_zirk.tif"
predict_localizations(image)

