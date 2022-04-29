import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from src.utility import get_reconstruct_coords, read_thunderstorm_drift_json
import os
import json
try:
    from third_party.dme.dme import dme_estimate
    from third_party.dme.rcc import rcc3D
except:
    print("unable to load dme drift correct libraries")



class Emitter():
    """
    Emitter set for SMLM data
    """

    ATTRIBUTES = ["xyz", "photons", "sigxsigy", "frames", "ids", "p"]
    def __init__(self, xyz, photons,  frames,sigxsigy=None, p=None, ids=None):
        self.xyz = np.array(xyz)#in nm
        self.photons = np.array(photons)
        self.sigxsigy = np.array(sigxsigy)#in nm?
        self.frames = np.array(frames,dtype=np.int32)
        if np.any(sigxsigy):
            self.sigxsigy = np.array(sigxsigy)
        else:
            self.sigxsigy = np.zeros((xyz.shape[0],2))
        if np.any(ids):
            self.ids = np.array(ids,dtype=np.int32)
        else:
            self.ids = np.arange(0, self.xyz.shape[0])
        if np.any(p):
            self.p = np.array(p)
        else:
            self.p = np.ones(self.xyz.shape[0])
        self.check_data_integrety()
        self.metadata = ""

    def check_data_integrety(self):
        for attr1 in self.ATTRIBUTES:
            for attr2 in self.ATTRIBUTES:
                if getattr(self, attr1).shape[0] != getattr(self,attr2).shape[0] and getattr(self,attr2).shape[0] !=0 and getattr(self, attr1).shape[0] !=0:
                    warnings.warn(f"{attr1} and {attr2} dont have the some length data might be corrupted")


    def add_emitters(self, xyz, photons, sigxsigy, frames, ids):
        """
        Add emitters to current set
        """
        #todo use numpy append here
        self.xyz = np.append(self.xyz, xyz, axis=0)
        self.sigxsigy = np.append(self.sigxsigy, sigxsigy, axis=0)
        self.frames = np.append(self.frames, frames, axis=0)
        self.ids = np.append(self.ids, np.arange(self.ids[-1], xyz.shape[0], 1), axis=0)

        self.check_data_integrety()

    def __add__(self, other):
        """
        Concatenates two emitter set to a new one
        :param other: Emitter set
        """
        self.xyz = np.append(self.xyz, other.xyz, axis=0)
        self.sigxsigy = np.append(self.sigxsigy, other.sigxsigy, axis=0)
        self.frames = np.append(self.frames, other.frames, axis=0)
        self.ids = np.append(self.ids, np.arange(self.ids[-1], other.xyz.shape[0], 1), axis=0)
        self.photons = np.append(self.photons, other.photons, axis=0)

    #compute difference between emitter sets and return emitter set
    def __sub__(self, other):
        """
        Compute the difference between two emitter sets.
        :param other: Emitter set
        :return: Emitters that don`t overlap within a 100nm range
        """
        #todo: should have same frame length
        found_emitters = []
        for i in range(int(self.frames.max())):
            pred_f = self.xyz[np.where(self.frames==i)]
            truth_f = other.xyz[np.where(other.frames==i)]
            if np.any(self.ids[np.where(self.frames==i)]):
                id_f = self.ids[np.where(self.frames==i)][0]
                t_truth = []
                for k in range(pred_f.shape[0]):
                    emitter_id = id_f + k
                    found = False
                    min_dis = 100
                    for l in range(truth_f.shape[0]):
                        dis = np.linalg.norm(pred_f[k] - truth_f[l])
                        if dis < min_dis:
                            if emitter_id not in found_emitters and l not in t_truth:
                                min_dis = dis
                                current_diff = pred_f[k] - truth_f[l]
                                current_l = l
                                found = True
                    if found:
                        #todo append error to new emitter set?
                        found_emitters.append(emitter_id)
                        t_truth.append(current_l)
        diff = np.setdiff1d(self.ids, np.array(found_emitters,dtype=np.int32))
        return self.subset(diff)

    def subset(self, ids):
        """
        Returns a new emitter set from a list of given ids
        """
        new = Emitter(deepcopy(self.xyz[ids]), deepcopy(self.photons[ids]), deepcopy(self.frames[ids]), deepcopy(self.sigxsigy[ids]))
        return new

    def filter(self, sig_x=3, sig_y=3, photons=0, p=0, frames=None):
        """
        Filter emitter set
        :param sig_x:
        :param sig_y:
        :param photons:
        :param p:
        :param frames:
        :return: New emitter set with filters applied
        """
        metadata = f"""Filters:\n sigma x:{sig_x}\n sigma y:{sig_y}\n photons:{photons}\n  p:{p}\n frames:{frames}"""
        conditions=[]
        conditions.append(self.sigxsigy[:,0]<sig_x)
        conditions.append(self.sigxsigy[:,1]<sig_y)
        conditions.append(self.photons>photons)
        conditions.append(self.p>p)
        if frames:
            conditions.append(self.frames>frames[0])
            conditions.append(self.frames<frames[1])
        indices = self.ids[np.where(np.all(np.array(conditions),axis=0))]
        s = self.subset(indices)
        s.metadata += metadata
        return s

    def save(self, path, format="npy"):
        """
        Save emitter set in the given format
        :param path: save path
        :param format: save format
        :return:
        """
        names = {}
        data = []
        for col,att in enumerate(self.ATTRIBUTES):
            val = getattr(self, att)
            if val:
                names[att] = col
                data.append(val)
        names["metadata"] = self.metadata
        data = np.concatenate(data, axis=-1)
        if format =="npy":
            #make if exists test for non mandatory columns
            path_m = os.path.splitext(path)
            with open(path_m + '_metadata.json', 'w') as fp:
                json.dump(names, fp)
            np.save(path, data)
        elif format == "csv":
            pass
        elif format == "txt":
            pass

    def apply_drift(self, path):
        """
        Apply thunderstorm c-spline drift or raw drif in csv format
        :param path: path to drift correct file
        """
        if isinstance(path, np.ndarray):
            drift = path[:,(1,0)]
        elif path.split(".")[-1] == "csv":
            print("drift correction activated")
            drift = pd.read_csv(path).as_matrix()[::,(1,2)]
            #drift[:,0] *= -1
        else:
            print("drift correction activated")
            path = path+r"\drift.json"
            drift = read_thunderstorm_drift_json(path)
        for i in range(self.frames.max()):
            self.xyz[np.where(self.frames == i),0] += drift[i,1]*100
            self.xyz[np.where(self.frames == i),1] -= drift[i,0]*100

    def use_dme_drift_correct(self):
        use_cuda = True
        fov_width = 80
        loc_error = np.array((0.2, 0.2, 0.03))  # pixel, pixel, um
        loc = self.xyz
        localizations = np.zeros((loc.shape[0], 3))
        localizations[:, 0:2] = loc / 100
        crlb = np.ones(localizations.shape) * np.array(loc_error)[None]
        estimated_drift, _ = dme_estimate(localizations, self.frames,
                                          crlb,
                                          framesperbin=200,  # note that small frames per bin use many more iterations
                                          imgshape=[fov_width, fov_width],
                                          coarseFramesPerBin=200,
                                          coarseSigma=[0.2, 0.2, 0.2],  # run a coarse drift correction with large Z sigma
                                          useCuda=use_cuda,
                                          useDebugLibrary=False)
        self.apply_drift(estimated_drift[:,0:2])

    @classmethod
    def load(cls, path, raw=False):
        """
        Load emitter set from given path
        :param path:
        :param raw:
        :return:
        """
        #todo: if raw load put throught image_to_tensor
        pass

    @classmethod
    def from_result_tensor(cls, result_tensor, p_threshold, coord_list=None):
        """
        Build an emitter set from the neural network output
        :param result_tensor: Feature map output of the AI
        :param coord_list: Coordinates of the crops detected by the wavelet filter bank
        :param p_threshold: Threshold value for a classifier pixel to contain a localisation
        :return:
        """
        xyz = []
        N_list = []
        sigx_sigy = []
        sN_list = []
        p_list = []
        frames = []
        for i in range(result_tensor.shape[0]):
            classifier =result_tensor[i, :, :, 2]
            x= np.sum(classifier)
            if x > p_threshold:
                indices = get_reconstruct_coords(classifier, p_threshold)#todo: import and use function

                x = result_tensor[i, indices[0], indices[1], 0]
                y = result_tensor[i, indices[0], indices[1], 1]
                p = result_tensor[i, indices[0], indices[1], 2]
                dx = result_tensor[i, indices[0], indices[1], 3]#todo: if present
                dy = result_tensor[i, indices[0], indices[1], 4]
                N = result_tensor[i, indices[0], indices[1], 5]
                dN = result_tensor[i, indices[0], indices[1], 6]

                for j in range(indices[0].shape[0]):
                    if np.any(coord_list):
                        xyz.append(np.array([coord_list[i][0] +float(indices[0][j]) + (x[j]),#x
                                             coord_list[i][1] +float(indices[1][j]) + y[j]])*100)#y
                        frames.append(coord_list[i][2])

                    else:
                        print("warning no coordinate list loaded -> no offsets are added")
                        xyz.append(np.array([float(indices[0][j]) + (x[j]),#x
                                             float(indices[1][j]) + y[j]])*100)#y
                        frames.append(i)
                    p_list.append(p[j])
                    sigx_sigy.append(np.array([dx[j],dy[j]]))
                    N_list.append(N[j])
                    sN_list.append(dN[j])
        return cls(xyz, N_list, frames, sigx_sigy, p_list)

    @classmethod
    def from_ground_truth(cls, coordinate_tensor):
        coords = []
        for i, crop in enumerate(coordinate_tensor):
            for coord in crop:
                if coord[2] != 0:
                    #todo add photons
                    coords.append(np.array([coord[0], coord[1], i, coord[3]]))
        coords = np.array(coords)
        coords[:, 0:2] *= 100
        return cls(coords[:,0:2], coords[:,3], coords[:,2])



    def get_frameset_generator(self, images, frames, gt=None):
        """
        Generator for plotting emitter sets on images
        :param images:
        :param gt:
        :param frames:
        :return: Generator with image, emitters, ground truth
        """
        def frameset_generator():
            for frame in frames:
                emitters = self.xyz[np.where(self.frames==frame)]/100
                if np.any(emitters):
                    print(f"current frame: {frame}")
                    image = images[frame]
                    if gt:
                        gt_emitters = gt.xyz[np.where(gt.frames==frame)]/100
                        yield image[:,:,1], emitters, gt_emitters
                    else:
                        yield image, emitters
        return frameset_generator