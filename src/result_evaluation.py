#todo: compute jaccard index
import numpy as np
import copy


def jaccard_index(reconstruction,ground_truth):
    """
    define false positive: localisation without ground truth for distance > 200nm
    define false negative: ground truth without localisation for distance > 200nm
    :param ground_truth:
    :param reconstruction:
    """
    false_positive = []
    result = []
    this_ground_truth = copy.deepcopy(ground_truth)
    for k in range(reconstruction.shape[0]):
        for i in range(reconstruction[k].shape[0]):
            distance = 100
            current_j = -1
            for j in range(this_ground_truth[k].shape[0]):
                current_ground_truth = this_ground_truth[k][:,0:2]
                #dis = np.linalg.norm(reconstruction[i] - this_ground_truth[j])
                if np.linalg.norm(reconstruction[k][i] - current_ground_truth[j]) < distance:
                    distance = np.linalg.norm(reconstruction[k][i] - current_ground_truth[j])
                    current_j = j
                    vec = reconstruction[k][i] - current_ground_truth[j]
            if current_j != -1:
                result.append(np.array([*reconstruction[k][i],*vec]))
                #this_ground_truth[k] = np.delete(this_ground_truth[k], current_j, axis=0)
            else:
                false_positive.append(reconstruction[k][i])

    f_n =0
    for s in this_ground_truth:
        f_n += s.shape[0]-1
    result = np.array(result)
    false_positive = np.array(false_positive)
    false_negative = this_ground_truth
    jac = result.shape[0] / (result.shape[0] + false_positive.shape[0] + f_n)
    error = 0
    for i in result:
        error += i[2] ** 2 + i[3] ** 2
    rmse = np.sqrt(error / result.shape[0])
    print("Jaccard index: = ", jac, " rmse = ", rmse)

    return result, false_positive, false_negative, jac, rmse