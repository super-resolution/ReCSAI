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



path= r"path_to_your_file.tif"

facade = ResUNetFacade()



facade.sigma = 180
facade.wavelet_thresh = 0.3
p_threshold = 0.4
result_tensor,coord_list = facade.predict(path, raw=True)
if not os.path.exists(os.getcwd()+r"\tmp"):
    os.mkdir(os.getcwd()+r"\tmp")
np.save(os.getcwd()+r"\tmp\current_result.npy",result_tensor)
np.save(os.getcwd()+ r"\tmp\current_coordinate_list.npy", coord_list)

result_tensor = np.load(os.getcwd()+r"\tmp\current_result.npy",allow_pickle=True)
coord_list = np.load(os.getcwd()+ r"\tmp\current_coordinate_list.npy",allow_pickle=True)
print("finished AI")

emitter = Emitter.from_result_tensor(result_tensor,p_threshold, coord_list )
print(emitter.xyz.shape[0])
emitter_filtered = emitter.filter(sig_x=0.1, sig_y=0.1, photons=0.1, )
#emitter_filtered.use_dme_drift_correct()

plot_emitter_set(emitter_filtered)

