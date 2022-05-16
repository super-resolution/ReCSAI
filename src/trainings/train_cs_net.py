import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.cs_model import CompressedSensingInceptionNet, CompressedSensingCVNet, CompressedSensingUNet,\
    CompressedSensingResUNet, StandardUNet, CompressedSensingConvNet
from src.facade import NetworkFacade
from src.utility import get_root_path

CURRENT_INCEPTION_PATH = get_root_path()+r"/trainings/cs_inception/_final_training_100_10_ndata"
CURRENT_CNN_PATH = get_root_path()+r"/trainings/cs_cnn/_final_training_100_ndata_test_thresholding"
CURRENT_U_PATH = get_root_path()+r"/trainings/cs_u/_final_training_100_ndata2"
CURRENT_RES_U_PATH = get_root_path()+r"/trainings/cs_u/_final2_training_100_ndata_test_compare"#best so far
#CURRENT_STANDARD_U_NETWORK = get_root_path()+r"/trainings/cs_u/standard_unet_ndata"
#CURRENT_CONV_NETWORK_PATH = get_root_path()+r"/trainings/cs_conv/_conv_training_ndata"

CURRENT_WAVELET_PATH = get_root_path()+r"/trainings/wavelet/training_lvl5/cp-5000.ckpt"


class CSInceptionNetFacade(NetworkFacade):
    def __init__(self):
        super(CSInceptionNetFacade, self).__init__(CompressedSensingInceptionNet, CURRENT_INCEPTION_PATH,
                                                 CURRENT_WAVELET_PATH)
        self.train_loops = 200


class CNNNetFacade(NetworkFacade):
    def __init__(self):
        super(CNNNetFacade, self).__init__(CompressedSensingCVNet, CURRENT_CNN_PATH,
                                          CURRENT_WAVELET_PATH)
        self.train_loops = 50


class CSUNetFacade(NetworkFacade):
    def __init__(self):
        super(CSUNetFacade, self).__init__(CompressedSensingUNet, CURRENT_U_PATH,
                                         CURRENT_WAVELET_PATH)
        self.train_loops = 120


class ResUNetFacade(NetworkFacade):
    def __init__(self):
        super(ResUNetFacade, self).__init__(CompressedSensingResUNet, CURRENT_RES_U_PATH,
                                           CURRENT_WAVELET_PATH, shape=128)
        self.train_loops = 120

# class StandardUNetFacade(NetworkFacade):
#     def __init__(self):
#         super(StandardUNetFacade, self).__init__(StandardUNet, CURRENT_STANDARD_U_NETWORK,
#                                          CURRENT_WAVELET_PATH)
#         self.train_loops = 120

if __name__ == '__main__':
    training = ResUNetFacade()
    training.train_saved_data()