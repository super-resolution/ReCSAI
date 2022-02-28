import warnings
import numpy as np

class Emitter():
    attributes = ["xyz", "photons", "sigxsigy", "frames", "ids"]
    def __init__(self, xyz, photons,  frames,sigxsigy=None, p=None, ids=None):
        self.xyz = np.array(xyz)
        self.photons = np.array(photons)
        self.sigxsigy = np.array(sigxsigy)
        self.frames = np.array(frames,dtype=np.int32)
        if np.any(sigxsigy):
            self.sigxsigy = np.array(sigxsigy)
        else:
            self.sigxsigy = np.zeros((xyz.shape[0],2))
        if np.any(ids):
            self.ids = np.array(ids,dtype=np.int32)
        else:
            self.ids = np.arange(0, xyz.shape[0])
        if np.any(p):
            self.p = np.array(p)
        else:
            self.p = np.ones(xyz.shape[0])
        self.check_data_integrety()

    def check_data_integrety(self):
        for attr1 in self.attributes:
            for attr2 in self.attributes:
                if getattr(self, attr1).shape[0] != getattr(self,attr2).shape[0] and getattr(self,attr2).shape[0] !=0 and getattr(self, attr1).shape[0] !=0:
                    warnings.warn(f"{attr1} and {attr2} dont have the some length data might be corrupted")


    def add_emitters(self, xyz, photons, sigxsigy, frames, ids):
        #todo use numpy append here
        self.xyz.append(xyz)
        self.photons.append(photons)
        self.sigxsigy.append(sigxsigy)
        self.frames.append(frames)
        self.ids.append(ids.astpye(np.int32))
        self.check_data_integrety()


    #todo: overwrite substract? and modulo?
    #todo: compute difference between emitter sets and return emitter set
    def __sub__(self, other):
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
        new = Emitter(self.xyz[ids], self.photons[ids], self.frames[ids], self.sigxsigy[ids])
        return new

    def plot_failures(self, images, result_array):
        #todo: plot
        pass