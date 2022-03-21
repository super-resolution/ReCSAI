from src.custom_metrics import JaccardIndex
import pandas as pd
from src.models.cs_model import CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.visualization import display_storm_data
from src.utility import get_root_path
from tifffile.tifffile import TiffFile
import os
import numpy as np
from matplotlib import pyplot as plt


class TrainInceptionNet(NetworkFacade):
    def __init__(self):
        super(TrainInceptionNet, self).__init__(CompressedSensingInceptionNet,
                                                get_root_path() + r"/trainings/cs_inception/_background_l_cs_100_large_dataset_airy4",
                                                get_root_path() + r"\trainings\wavelet\training_lvl2\cp-10000.ckpt",
                                                shape=128)


def exclude_duplicates(pred):
    indices = []
    pred = pred[pred[:,2].argsort()[::-1]]
    for i in range(pred.shape[0]):
        for j in range(pred.shape[0]-(i+1)):
            j = (i+1)+j
            dis = np.linalg.norm(pred[i] - pred[j])
            if dis<30:
                indices.append(j)
    return np.delete(pred, indices,axis=0)[:,0:2]


def compute_evaluation_jaccard(pred_frame_list, truth_frame_list):
    JCI = JaccardIndex(os.getcwd()+r"\tmp")
    for i in range(len(pred_frame_list)):
        pred = pred_frame_list[i][:,(1,0,6)]
        pred[:,0:2] = pred[:,0:2]*98+50
        #pred = exclude_duplicates(pred)
        #todo: sort localisations by probability?
        truth = truth_frame_list[i]
        # plt.scatter(truth[:,0], truth[:,1])
        # plt.scatter(pred[:,0],pred[:,1])
        # plt.show()

        JCI.update_state_frame(truth, pred[:,0:2])
    return JCI.result(0)



def truth_to_frame_list(path):
    frame_list = []
    files = os.listdir(path)
    for file in files:
        frame = pd.read_csv(path+"\\"+file)
        frame = frame.values
        frame_list.append(frame[:,2:4])
    return frame_list

def to_frame_list(result_array):
    #todo: frame is coords[2]
    current_frame=0
    frame_list = []
    current_frame_locs = []
    #todo: kick duplicated localisations
    for loc in result_array:
        if loc[2] == current_frame:
            current_frame_locs.append(loc)
        else:
            current_frame = loc[2]
            frame_list.append(np.array(current_frame_locs))
            current_frame_locs = []
            current_frame_locs.append(loc)
    frame_list.append(np.array(current_frame_locs))
    return frame_list


if __name__ == '__main__':



    facade = TrainInceptionNet()
    # facade.pretrain_current_sigma_d()
    # todo: try to train one iteration on sigma
    facade.threshold = 0.2 # todo: still has artefacts...
    facade.sigma_thresh = 0.05
    facade.photon_filter = 0.2
    # result_tensor,coord_list = facade.predict(image, raw=True)
    # if not os.path.exists(os.getcwd()+r"\tmp"):
    #     os.mkdir(os.getcwd()+r"\tmp")
    # np.save(os.getcwd()+r"\tmp\current_result.npy",result_tensor)
    # np.save(os.getcwd()+ r"\tmp\current_coordinate_list.npy", coord_list)
    result_tensor = np.load(os.getcwd() + r"\tmp\current_result.npy", allow_pickle=True)
    # for i in range(result_tensor.shape[0]):
    #     fig,axs = plt.subplots(4)
    #     axs[0].imshow(result_tensor[i,:,:,2])
    #     axs[1].imshow(result_tensor[i,:,:,3])
    #     axs[2].imshow(result_tensor[i,:,:,4])
    #     axs[3].imshow(result_tensor[i,:,:,5])
    #     plt.show()
    coord_list = np.load(os.getcwd() + r"\tmp\current_coordinate_list.npy", allow_pickle=True)
    result_array = facade.get_localizations_from_image_tensor(result_tensor, coord_list)
    pred_frame_list = to_frame_list(result_array)
    truth_frame_list = truth_to_frame_list(r"D:\Daten\Artificial\fluorophores\frames")
    print(compute_evaluation_jaccard(pred_frame_list, truth_frame_list))
