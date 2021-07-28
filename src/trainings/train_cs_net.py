from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet,CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.utility import get_root_path


class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, get_root_path()+r"/trainings/cs_inception/_new_decodeL_ndata_id",
                                                get_root_path()+r"/trainings/wavelet/training_lvl2/cp-10000.ckpt")
        self.train_loops = 30

class TrainCVNet(NetworkFacade):
    def __init__(self):
        super(TrainCVNet, self).__init__(CompressedSensingCVNet, get_root_path() +r"\trainings\cs_cnn\learn_all_new_data_variable_sigma",
                                         get_root_path()+r"\trainings\wavelet\training_lvl2\cp-10000.ckpt")
        self.train_loops = 60





if __name__ == '__main__':
    training = TrainInceptionNet()
    training.train_saved_data()