from src.models.cs_model import CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.visualization import display_storm_data
from src.utility import get_root_path
from tifffile.tifffile import TiffFile
import os
import numpy as np
import matplotlib.pyplot as plt
from src.trainings.train_cs_net import InceptionNetFacade
from src.visualization import plot_wavelet_bin_results

def plot_stuff(pred_frame_list, image, tensor):
    for i in range(len(pred_frame_list)):
        i+=1990
        t = pred_frame_list[i]
        pred = pred_frame_list[i][:,(1,0)]
        pred[:,0:2] = pred[:,0:2]+0.5
        #todo: sort localisations by probability?
        im = image[i]
        #todo: plot rect around localisation
        fig, axs = plt.subplots(2)
        axs[1].imshow(tensor[i,:,:,2])
        axs[0].imshow((im-im.min())/im.max(), )
        axs[0].scatter(pred[:,0],pred[:,1], marker="x", c="r")
        plt.show()


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

def coord_to_fram(result_array):
    #todo: frame is coords[2]
    current_frame=0
    frame_list = []
    current_frame_locs = []
    #todo: kick duplicated localisations
    for loc in result_array:
        loc[0:2] +=4
        if loc[2] == current_frame:
            current_frame_locs.append(loc)
        else:
            current_frame = loc[2]
            frame_list.append(np.array(current_frame_locs))
            current_frame_locs = []
            current_frame_locs.append(loc)
    frame_list.append(np.array(current_frame_locs))
    return frame_list

#path = r"D:\Daten\Dominik_B\Cy5_AI_enhanced_5.tif"

path = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
path = r"D:\Daten\Dominik_B\JF646_Aktin_100us_45px_100nm_Framefrq2.4Hz_linefrq108.7Hz_100LP_4000Frames.tif kept stack.tif"
path = r"D:\Daten\Dominik_B\Dyomics654_100us101nm45pxFramefrq2.4HzLinefrq108.7Hz_4850Frames_100LP.tif kept stack.tif"

#path = r"D:\Daten\Artificial\ContestHD.tif"
#path = r"D:\Daten\Christina\U2OS_+Ac4ManAz_5uM.tif"
path = r"C:\Users\biophys\matlab\test2.tif"
#path = r"D:\Daten\Artificial\sequence-as-stack-MT0.N1.HD-2D-Exp.tif"
#path = r"D:\Daten\Dominik_B\substack_cy5_lange_aufnahme.tif"


with TiffFile(path) as tif:
    image = tif.asarray()[:,0:-18]#[:,7:-5]
    #image = np.rot90(image, axes=(1, 2))



facade = InceptionNetFacade()

facade.threshold = 0.08 # todo: still has artefacts...
facade.sigma_thresh = 0.3
facade.photon_filter = 0.0
result_tensor = np.load(os.getcwd()+r"\tmp\current_result.npy",allow_pickle=True)
coord_list = np.load(os.getcwd()+ r"\tmp\current_coordinate_list.npy",allow_pickle=True)
test = coord_to_fram(coord_list)
#plot_wavelet_bin_results(image[3000], image[3000], test[3000]) #todo put this together
result_array = facade.get_localizations_from_image_tensor(result_tensor, coord_list)
pred_frame_list = to_frame_list(result_array)
plot_stuff(test, image, result_tensor)