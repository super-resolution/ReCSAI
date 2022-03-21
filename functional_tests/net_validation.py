import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from src.trainings.train_cs_net import CSUNetFacade, InceptionNetFacade, CVNetFacade
from src.models.loss_functions import loc_loss, count_loss, compute_loss_decode_ncs
from src.emitters import Emitter

facade = CSUNetFacade()
facade.sigma = 180
#facade.wavelet_thresh = 0.08
#facade.threshold = 0.2
#facade.sigma_thresh = 0.1
#facade.photon_filter = 0.1
dataset = facade.get_current_dataset()
for i, (train_image, noiseless_gt, coords_t, t, sigma) in enumerate(dataset):
    if i % 4 == 0:
        # train_image,noiseless_gt, coords,t = iterator.get_next()
        pred = facade.network(train_image, training=False)  # todo compute loss components
        im = train_image
        vloss = compute_loss_decode_ncs(coords_t, pred)
        closs = count_loss(coords_t, pred)
        b_loss = tf.keras.losses.MeanSquaredError()(train_image[:, :, :, 1] - noiseless_gt[:, :, :, 1],
                                                    pred[:, :, :, 7])
        print(f"validation loss = {vloss} {closs} {b_loss}")

pred_set = Emitter.from_result_tensor(pred.numpy(), 0.2)
gen = pred_set.get_frameset_generator(im, np.arange(0,20))
for (image, pred) in gen():
    plt.imshow(image)
    plt.scatter(pred[:,1], pred[:,0])
    plt.show()