import tensorflow as tf
import numpy as np
from unittest import TestCase
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from src.emitters import Emitter


DATASET_PATH = os.path.dirname(os.getcwd())+"/datasets/test_dataset/"

class BaseTest(TestCase):
    def setUp(self):
        coordinates = np.load(DATASET_PATH + "coordinates.npy", allow_pickle=True)[0]
        self.data = np.load(DATASET_PATH + "data.npy", allow_pickle=True).astype(np.float32)[0]
        feature_map = np.load(DATASET_PATH + "feature_maps.npy", allow_pickle=True).astype(np.float32)[0]
        #this is already a test for loading data
        self.gt_emitter = Emitter.from_ground_truth(coordinates)
        self.pred_emitter = Emitter.from_result_tensor(feature_map, 0.15)
        #todo: use any validation dataset together with any feature space raw...
        #todo: create emitter set from ground truth and predict...


    def test_filter(self):
        #apply basic filtering
        e_new = self.pred_emitter.filter(photons=0.3, sig_x=0.1, sig_y=0.6, frames=(10, self.pred_emitter.frames.max()))
        #test filtering filters...
        self.assertFalse(np.any(e_new.photons<0.3), "photon test failed")
        self.assertFalse(np.any(e_new.sigxsigy[:,0]>0.1), "sigma x test failed")
        self.assertFalse(np.any(e_new.sigxsigy[:,1]>0.6), "sigma y test failed")
        self.assertFalse(not np.any(np.logical_and(e_new.sigxsigy[:,1]<0.2,e_new.sigxsigy[:,1]>0.02)), "sigma test failed")
        self.assertFalse(np.any(e_new.frames < 10), "frame test failed")

    def test_add(self):
        pass

    def test_substract(self):
        pass

    def test_frame_plot(self):
        pass