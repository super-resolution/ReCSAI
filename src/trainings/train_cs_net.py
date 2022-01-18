from src.models.cs_model import CompressedSensingInceptionNet, CompressedSensingCVNet
from src.facade import NetworkFacade
from src.utility import get_root_path

CURRENT_INCEPTION_NETWORK_PATH = get_root_path()+r"/trainings/cs_inception/__transfer_hd_100_100"
CURRENT_CV_NETWORK_PATH = get_root_path()+r"/trainings/cs_cnn/_final_training_100_100"

#todo: final training rounds.... transfer learning?



class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, CURRENT_INCEPTION_NETWORK_PATH,
                                                get_root_path()+r"/trainings/wavelet/training_lvl2/cp-10000.ckpt")
        self.train_loops = 60

class TrainCVNet(NetworkFacade):
    def __init__(self):
        super(TrainCVNet, self).__init__(CompressedSensingCVNet, CURRENT_CV_NETWORK_PATH,
                                         get_root_path()+r"\trainings\wavelet\training_lvl2\cp-10000.ckpt")
        self.train_loops = 60





if __name__ == '__main__':
    training = TrainInceptionNet()
    training.train_saved_data()