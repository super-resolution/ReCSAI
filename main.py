import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from src.models.cs_model import CompressedSensingInceptionNet
from src.facade import NetworkFacade
from src.visualization import display_storm_data, render, plot_parameter_distribution, display_emitter_set
from src.utility import get_root_path, FRC_loss
from tifffile.tifffile import TiffFile
import os
import numpy as np
from src.trainings.train_cs_net import CSUNetFacade, InceptionNetFacade, CVNetFacade, UNetFacade
from src.emitters import Emitter


#path = r"D:\Daten\Dominik_B\Cy5_AI_enhanced_5.tif"

path = r"D:\Daten\Dominik_B\Cy5\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#path = r"D:\Daten\Dominik_B\JF646\JF646_Aktin_100us_45px_100nm_Framefrq2.4Hz_linefrq108.7Hz_100LP_4000Frames.tif kept stack.tif"
#path = r"D:\Daten\Dominik_B\Dyomics654\Dyomics654_100us101nm45pxFramefrq2.4HzLinefrq108.7Hz_4850Frames_100LP.tif kept stack.tif"

#path = r"D:\Daten\Artificial\ContestHD.tif"
#path = r"D:\Daten\Artificial\sequence-as-stack-MT0.N1.HD-2D-Exp.tif"

#path = r"D:\Daten\Christina\U2OS_+Ac4ManAz_5uM.tif"
path = r"C:\Users\biophys\matlab\test2.tif"

#path = r"D:\Daten\Dominik_B\191017_3xBiotin-StraptavidinAl647_7.8ms-128px-200gain-25000fr-kA-HILO_3.tif"

#todo: check path for drift
with TiffFile(path) as tif:
    image = tif.asarray()#[:,0:-18]
    #image = tif.asarray()[:,6:-2]
    #image = np.rot90(image, axes=(1, 2))




facade = CSUNetFacade()
if os.path.exists(os.path.dirname(path)+r"\drift.json"):
    facade.drift_path = os.path.dirname(path)

facade.drift_path = r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv"


facade.sigma = 180
#facade.pretrain_current_sigma_d()
facade.wavelet_thresh = 0.04
facade.threshold = 0.15
facade.sigma_thresh = 0.5
facade.photon_filter = 0.0
# result_tensor,coord_list = facade.predict(image, raw=True)#todo add drift if there is drift
# if not os.path.exists(os.getcwd()+r"\tmp"):
#     os.mkdir(os.getcwd()+r"\tmp")
# np.save(os.getcwd()+r"\tmp\current_result.npy",result_tensor)
# np.save(os.getcwd()+ r"\tmp\current_coordinate_list.npy", coord_list)
# #todo: save metadata and emitter set?
result_tensor = np.load(os.getcwd()+r"\tmp\current_result.npy",allow_pickle=True)
coord_list = np.load(os.getcwd()+ r"\tmp\current_coordinate_list.npy",allow_pickle=True)
#result_array = facade.get_localizations_from_image_tensor(result_tensor, coord_list)
# print(result_array.shape[0])
print("finished AI")

emitter = Emitter.from_result_tensor(result_tensor,0.15, coord_list )
#todo: filtering and drift correction
emitter_filtered = emitter.filter(0.5, 0.5, 0.1)
emitter_filtered.apply_drift(r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv")

display_emitter_set(emitter_filtered)
#cy5
# result_array[:,1] -= (1-result_array[:,2]/result_array[:,2].max())*0.6
# frc = FRC_loss(render(result_array[:result_array.shape[0]//2,]), render(result_array[result_array.shape[0]//2:]))
# print(frc)
# #plot_parameter_distribution(result_array)
# display_storm_data(result_array[:,], name="Cy5", frc=frc)

#dyomics etc
frc = FRC_loss(render(result_array[:result_array.shape[0]//4,]), render(result_array[result_array.shape[0]//4:result_array.shape[0]//2]))
print(frc)
plot_parameter_distribution(result_array)
display_storm_data(result_array[np.where(result_array[:,0]<90)], name="matlab", frc=frc)
