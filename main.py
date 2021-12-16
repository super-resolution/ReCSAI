from src.models.cs_model import CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.visualization import display_storm_data
from src.utility import get_root_path
from tifffile.tifffile import TiffFile
import os
import numpy as np





#validate_cs_model()
#train_cs_net()
#train_nonlinear_shifter_ai()
#learn_psf()

#path = r"D:\Daten\Dominik_B\Cy5_AI_enhanced_5.tif"

path = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#path = r"D:\Daten\Dominik_B\JF646_Aktin_100us_45px_100nm_Framefrq2.4Hz_linefrq108.7Hz_100LP_4000Frames.tif kept stack.tif"
path = r"D:\Daten\Artificial\ContestHD.tif"
#path = r"D:\Daten\Christina\U2OS_+Ac4ManAz_5uM.tif"
#path = r"C:\Users\biophys\matlab\test2_crop_BP.tif"

with TiffFile(path) as tif:
    image = tif.asarray()
    #image = np.rot90(image, axes=(1, 2))


#image = r"D:\Daten\AI\COS7_LT1_beta-tub-Alexa647_new_D2O+MEA5mM_power6_OD0p6_3_crop.tif"
#image = r"D:\Daten\Artificial\ContestHD.tif"
#image = r"D:\Daten\Domi\origami\201203_10nM-Trolox_ScSystem_50mM-MgCl2_kA_TiRF_568nm_100ms_45min_no-gain-10MHz_zirk.tif"
class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, get_root_path()+r"/trainings/cs_inception/_crazy_test_specific",
                                                get_root_path()+r"\src\trainings\training_lvl5\cp-5000.ckpt",shape=128)



facade = TrainInceptionNet()
facade.sigma = 150
#facade.pretrain_current_sigma_d()
facade.wavelet_thresh = 0.13#todo: retrain wavelet
facade.threshold = 0.1 # todo: still has artefacts...
facade.sigma_thresh = 0.1
facade.photon_filter = 0.1
result_tensor,coord_list = facade.predict(image, raw=True)
if not os.path.exists(os.getcwd()+r"\tmp"):
    os.mkdir(os.getcwd()+r"\tmp")
np.save(os.getcwd()+r"\tmp\current_result.npy",result_tensor)
np.save(os.getcwd()+ r"\tmp\current_coordinate_list.npy", coord_list)
result_tensor = np.load(os.getcwd()+r"\tmp\current_result.npy",allow_pickle=True)
coord_list = np.load(os.getcwd()+ r"\tmp\current_coordinate_list.npy",allow_pickle=True)
result_array = facade.get_localizations_from_image_tensor(result_tensor, coord_list)
print(result_array.shape[0])
print("finished AI")
np.save(os.getcwd()+r"\HDContest_extended_path.npy",result_array)
display_storm_data(result_array)#todo: save all datapoints/tensors? to array and filter in display

