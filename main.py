import os
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from src.visualization import display_storm_data, render, plot_parameter_distribution, plot_emitter_set
from src.trainings.train_cs_net import CSUNetFacade, CSInceptionNetFacade, CNNNetFacade, ResUNetFacade
from src.emitters import Emitter
from src.utility import FRC_loss


#path = r"D:\Daten\Dominik_B\Cy5_AI_enhanced_5.tif"

path = r"D:\Daten\Dominik_B\Cy5\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#path = r"D:\Daten\Dominik_B\JF646\JF646_Aktin_100us_45px_100nm_Framefrq2.4Hz_linefrq108.7Hz_100LP_4000Frames.tif kept stack.tif"
#path = r"D:\Daten\Dominik_B\Dyomics654\Dyomics654_100us101nm45pxFramefrq2.4HzLinefrq108.7Hz_4850Frames_100LP.tif kept stack.tif"

#path = r"D:\Daten\Artificial\ContestHD.tif"
#path = r"D:\Daten\Artificial\sequence-as-stack-MT0.N1.HD-2D-Exp.tif"

#path = r"D:\Daten\Christina\U2OS_+Ac4ManAz_5uM.tif"
path = r"C:\Users\biophys\matlab\test2.tif"

#path = r"D:\Daten\Dominik_B\191017_3xBiotin-StraptavidinAl647_7.8ms-128px-200gain-25000fr-kA-HILO_3.tif"



facade = ResUNetFacade()
if os.path.exists(os.path.dirname(path)+r"\drift.json"):
    facade.drift_path = os.path.dirname(path)



facade.sigma = 180
#facade.pretrain_current_sigma_d()
facade.wavelet_thresh = 0.04
p_threshold = 0.15
result_tensor,coord_list = facade.predict(path, raw=True)
if not os.path.exists(os.getcwd()+r"\tmp"):
    os.mkdir(os.getcwd()+r"\tmp")
np.save(os.getcwd()+r"\tmp\current_result.npy",result_tensor)
np.save(os.getcwd()+ r"\tmp\current_coordinate_list.npy", coord_list)
# #todo: save metadata and emitter set?
result_tensor = np.load(os.getcwd()+r"\tmp\current_result.npy",allow_pickle=True)
coord_list = np.load(os.getcwd()+ r"\tmp\current_coordinate_list.npy",allow_pickle=True)
print("finished AI")

emitter = Emitter.from_result_tensor(result_tensor,p_threshold, coord_list )
emitter_filtered = emitter.filter(sig_x=0.5, sig_y=0.5, photons=0.1)
emitter_filtered.apply_drift(r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv")

plot_emitter_set(emitter_filtered)
#todo: emitterset frc
#todo: emitterset parameter distribution
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
