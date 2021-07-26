import tensorflow as tf
import numpy as np
import copy
from src.utility import result_image_to_coordinates
import os

class JaccardIndex():
    def __init__(self, path):
        self.path = path+r"\accuracy.npy"
        self.result_manager = []
        if os.path.exists(self.path):
            values = np.load(self.path, allow_pickle=True )
            self.result_manager += list(values)
        else:
            print("initializing metrics from scratch")
        self.values= True
        #todo: load accuracy
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.error = []

    def save(self):
        np.save(self.path, np.array(self.result_manager))

    def update_state(self, y_true, y_pred, sample_weight=None):
        try:
            coords = result_image_to_coordinates(y_true)
        except: #this is for coordinate truth data
            coords = []
            for i,crop in enumerate(y_true):
                for coord in crop:
                    if coord[2] != 0:
                        coords.append(np.array([coord[0], coord[1], i]))
            coords = np.array(coords)

        coords[:,0:2] *= 100
        coords_pred = result_image_to_coordinates(y_pred, threshold=0.3)
        if len(coords_pred.shape)!=2:
            self.values =False
        else:
            self.values = True
            coords_pred[:,0:2] *= 100
            self.compute_jaccard(coords, coords_pred)

    def update_state_points(self, y_true, y_pred):
        if self.values:
            self.compute_jaccard(y_true, y_pred)


    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.error= []

    def result(self, step):
        if self.values:
            jac = self.tp/ (self.tp + self.fp + self.fn)
            rmse = np.std(np.array(self.error))
            self.result_manager.append(np.array([step, jac, rmse]))
            return jac, rmse, self.fp, self.fn
        else:
            self.result_manager.append(np.array([step, 0.0, 9000]))
            return 0.0, 9000, 0,0

    def compute_jaccard(self, truth, pred):
        """
        define false positive: localisation without ground truth for distance > 100nm
        define false negative: ground truth without localisation for distance > 100nm
        :param ground_truth:
        :param reconstruction:
        """
        distance_th= 100

        for i in range(int(pred[:,2].max())):
            pred_f = pred[np.where(pred[:,2]==i)]
            truth_f = truth[np.where(truth[:,2]==i)]
            t_pred = []
            t_truth = []
            for k in range(pred_f.shape[0]):
                for l in range(truth_f.shape[0]):
                    dis = np.linalg.norm(pred_f[k] - truth_f[l])
                    if dis < distance_th:
                        if k not in t_pred and l not in t_truth:
                            self.error.append(dis)
                            t_pred.append(k)
                            t_truth.append(l)

            self.fp += pred_f.shape[0] - len(t_pred)
            self.fn += truth_f.shape[0] - len(t_truth)
        self.tp += len(self.error)
