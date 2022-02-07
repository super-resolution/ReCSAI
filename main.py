from src.models.cs_model import CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.visualization import display_storm_data, render
from src.utility import get_root_path, FRC_loss
from tifffile.tifffile import TiffFile
import os
import numpy as np
from src.trainings.train_cs_net import CSUNetFacade, InceptionNetFacade, CVNetFacade

#path = r"D:\Daten\Dominik_B\Cy5_AI_enhanced_5.tif"

path = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#path = r"D:\Daten\Dominik_B\JF646_Aktin_100us_45px_100nm_Framefrq2.4Hz_linefrq108.7Hz_100LP_4000Frames.tif kept stack.tif"
#path = r"D:\Daten\Artificial\ContestHD.tif"
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
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, get_root_path()+r"/trainings/cs_inception/_final_training_100_10_ndata",
                                                get_root_path()+r"\src\trainings\training_lvl5\cp-5000.ckpt",shape=128)
class TrainCVNet(NetworkFacade):
    def __init__(self):
        super(TrainCVNet, self).__init__(CompressedSensingCVNet, CURRENT_CV_NETWORK_PATH,
                                         get_root_path()+r"\trainings\wavelet\training_lvl2\cp-10000.ckpt")
        self.train_loops = 60


facade = TrainInceptionNet()
facade.sigma = 180
#facade.pretrain_current_sigma_d()
facade.wavelet_thresh = 0.09
facade.threshold = 0.2
facade.sigma_thresh = 0.15
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
print(FRC_loss(render(result_array[:result_array.shape[0]//2,]), render(result_array[result_array.shape[0]//2:,])))

#todo: compute fourier ring correlation in display?
display_storm_data(result_array)

