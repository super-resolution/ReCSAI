import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from src.trainings.train_cs_net import CSUNetFacade, InceptionNetFacade, CVNetFacade

facade = CSUNetFacade()
facade.sigma = 180
facade.wavelet_thresh = 0.08
facade.threshold = 0.2
facade.sigma_thresh = 0.1
facade.photon_filter = 0.1
facade.validate_saved_data()