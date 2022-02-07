import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.cs_model import CompressedSensingInceptionNet, CompressedSensingCVNet, CompressedSensingUNet, CompressedSensingConvNet, CompressedSensingResUNet
from src.facade import NetworkFacade
from src.utility import get_root_path

CURRENT_INCEPTION_NETWORK_PATH = get_root_path()+r"/trainings/cs_inception/_final_training_100_10_ndata"
CURRENT_CV_NETWORK_PATH = get_root_path()+r"/trainings/cs_cnn/_final_training_100_ndata"
CURRENT_U_NETWORK_PATH = get_root_path()+r"/trainings/cs_u/_final_training_100_ndata"
CURRENT_U2_NETWORK_PATH = get_root_path()+r"/trainings/cs_u/_final2_training_100_ndata_further_higherit"

CURRENT_CONV_NETWORK_PATH = get_root_path()+r"/trainings/cs_conv/_conv_training_ndata"

CURRENT_WAVELET_PATH = get_root_path()+r"/trainings/wavelet/training_lvl5/cp-5000.ckpt"


class InceptionNetFacade(NetworkFacade):
    def __init__(self):
        super(InceptionNetFacade, self).__init__(CompressedSensingInceptionNet, CURRENT_INCEPTION_NETWORK_PATH,
                                                 CURRENT_WAVELET_PATH)
        self.train_loops = 200

class CVNetFacade(NetworkFacade):
    def __init__(self):
        super(CVNetFacade, self).__init__(CompressedSensingCVNet, CURRENT_CV_NETWORK_PATH,
                                          CURRENT_WAVELET_PATH)
        self.train_loops = 60

class UNetFacade(NetworkFacade):
    def __init__(self):
        super(UNetFacade, self).__init__(CompressedSensingUNet, CURRENT_U_NETWORK_PATH,
                                         CURRENT_WAVELET_PATH)
        self.train_loops = 120


class CSUNetFacade(NetworkFacade):
    def __init__(self):
        super(CSUNetFacade, self).__init__(CompressedSensingResUNet, CURRENT_U2_NETWORK_PATH,
                                           CURRENT_WAVELET_PATH)
        self.train_loops = 120



if __name__ == '__main__':
    training = CSUNetFacade()
    training.train_saved_data()