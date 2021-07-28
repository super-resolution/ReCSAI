from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet, CompressedSensingInceptionNet
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



#path = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
path = r"D:\Daten\Artificial\ContestHD.tif"
with TiffFile(path) as tif:
    image = tif.asarray()


#image = r"D:\Daten\AI\COS7_LT1_beta-tub-Alexa647_new_D2O+MEA5mM_power6_OD0p6_3_crop.tif"
#image = r"C:\Users\biophys\matlab\test2_crop_BP.tif"
#image = r"D:\Daten\Artificial\ContestHD.tif"
#image = r"D:\Daten\Domi\origami\201203_10nM-Trolox_ScSystem_50mM-MgCl2_kA_TiRF_568nm_100ms_45min_no-gain-10MHz_zirk.tif"
class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, get_root_path()+r"/trainings/cs_inception/_new_decodeL_ndata",
                                                get_root_path()+r"\trainings\wavelet\training_lvl2\cp-10000.ckpt")

class TrainCVNet(NetworkFacade):
    def __init__(self):
        super(TrainCVNet, self).__init__(CompressedSensingCVNet, get_root_path() +r"\trainings\cs_cnn\learn_all_new_data_gen_Test",
                                         get_root_path()+r"\trainings\wavelet\training_lvl2\cp-10000.ckpt")
        self.train_loops = 0

facade = TrainInceptionNet()
facade.sigma = 150
#facade.pretrain_current_sigma()
#todo: try to train one iteration on sigma
facade.threshold = 0.2
facade.wavelet_thresh = 0.1
facade.sigma_thresh = 0.08
result_array = facade.predict(image)
print(result_array.shape[0])
print("finished AI")
np.save(os.getcwd()+r"\HDContest_extended_path.npy",result_array)
display_storm_data(result_array[:,0:2])

