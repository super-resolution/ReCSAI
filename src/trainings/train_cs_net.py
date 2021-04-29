from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet,CompressedSensingInceptionNet
from src.facade import NetworkFacade



class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet, "./cs_training_learn_sigma",
                                                r"C:\Users\biophys\PycharmProjects\TfWaveletLayers\training_lvl2\cp-10000.ckpt")
        self.train_loops = 10


class TrainCVNet(NetworkFacade):
    def __init__(self):
        super(TrainCVNet, self).__init__(CompressedSensingCVNet, "./cs_training_u_nmask_loss",
                                         r"C:\Users\biophys\PycharmProjects\TfWaveletLayers\training_lvl2\cp-10000.ckpt")




if __name__ == '__main__':
    training = TrainInceptionNet()
    training.train()