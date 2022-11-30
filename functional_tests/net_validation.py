import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from src.trainings.train_cs_net import CSUNetFacade, CSInceptionNetFacade, ResUNetFacade
from src.models.loss_functions import loc_loss, count_loss, compute_loss_decode_ncs
from src.emitters import Emitter

facade = ResUNetFacade()
facade.sigma = 180
dataset = facade.get_current_dataset()
res = []
truth = []
for i, (train_image, noiseless_gt, coords_t, t, sigma) in enumerate(dataset):
    if i % 4 == 0:
        # train_image,noiseless_gt, coords,t = iterator.get_next()
        pred = facade.network(train_image, training=False)  # todo compute loss components
        im = train_image
        vloss = compute_loss_decode_ncs(coords_t, pred)
        closs = count_loss(coords_t, pred)
        b_loss = tf.keras.losses.MeanSquaredError()(train_image[:, :, :, 1] - noiseless_gt[:, :, :, 1],
                                                    pred[:, :, :, 7])
        #print(f"validation loss = {vloss} {closs} {b_loss}")

        pred_set = Emitter.from_result_tensor(pred.numpy(), 0.35)
        truth_set = Emitter.from_ground_truth(coords_t.numpy())
        fn = truth_set - pred_set
        fp = pred_set - truth_set
        tp = pred_set%truth_set
        jac = tp.length / (tp.length + fp.length + fn.length)
        print(tp.error)
        #todo: statistic over jaccard
        print(jac)
        res.append(pred)
        truth.append(coords_t)
        if i> 12:
            break


d = []
for i in range(25):
    th = 0.1+0.025*i
    pred_set = Emitter.from_result_tensor(res[1].numpy(), th)
    truth_set = Emitter.from_ground_truth(truth[1].numpy())
    delta = truth_set - pred_set
    d2 = pred_set - truth_set
    jac = truth_set.length/(truth_set.length+delta.length+d2.length)
    d.append(np.array([2*th, jac]))
d = np.array(d)
plt.plot(d[:,0], d[:,1],marker="x", c="k")
plt.ylabel("JI")
plt.xlabel("Threshold")
plt.savefig("validation.svg")
plt.show()
    # gen = pred_set.get_frameset_generator(im, np.arange(0,20))
    # for (image, pred) in gen():
    #     plt.imshow(image)
    #     plt.scatter(pred[:,1], pred[:,0])
    #     plt.show()