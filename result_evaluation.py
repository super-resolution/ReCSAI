#todo: compute jaccard index
import numpy as np
import copy
from factory import Factory
import os
from tifffile import TiffWriter, TiffFile
from src.utility import *
from src.models.cs_model import CompressedSensingNet, CompressedSensingInceptionNet
from src.models.wavelet_model import WaveletAI
from Thunderstorm_jaccard import read_Thunderstorm
import pandas as pd
from src.data import data_generator_coords
from src.utility import bin_localisations_v2, get_coords, get_root_path
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from src.trainings.train_cs_net import TrainInceptionNet

def read_Rapidstorm(path):
    data = pd.read_csv(path, header=1, delim_whitespace=True).as_matrix()
    data = data[:,(2,0,1)]
    points = []
    set = []
    frame = 0

    for point in data:
        if point[0].astype(np.uint8) == frame:
            set.append(np.array([point[2],point[1]]))
        else:
            frame = point[0].astype(np.uint8)
            points.append(np.array(set))
            set = []
            set.append(np.array([point[2],point[1]]))
    points.append(np.array(set))
    points = np.asarray(points)
    return points

def create_test_data(im_shape=100):
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)
    for j in range(10):
        image_stack = []
        coord=[]
        points = factory.create_point_set(on_time=1000-j*100)
        init_indices = np.random.choice(points.shape[0], 10)
        on_points = points[init_indices]
        for i in range(100): #todo: while loop here
            print(i)
            image, truth, on_points = factory.simulate_accurate_flimbi(points, on_points, switching_rate=0.1+0.02*j)#todo: simulate off
            image = factory.reduce_size(image)
            image = np.pad(factory.accurate_noise_simulations_camera(image),(14,14))
            truth = np.pad(factory.reduce_size(truth).astype(np.int32),(14,14))
            image_stack.append(image)
            coord.append(np.array(on_points))
        image_array = np.array(image_stack)
        if os.path.exists(os.getcwd() +r"\test_data"):
            os.mkdir(os.getcwd() +r"\test_data")
        with TiffWriter(os.getcwd() +r"\test_data" + str(j) + ".tif",
                        bigtiff=True) as tif:
            tif.save(image_array[:,14:-14,14:-14], photometric='minisblack')
        np.save(os.getcwd() +r"\test_data" + str(j) + ".npy", coord)

def jaccard_index(reconstruction,ground_truth):
    """
    define false positive: localisation without ground truth for distance > 200nm
    define false negative: ground truth without localisation for distance > 200nm
    :param ground_truth:
    :param reconstruction:
    """
    f_p = []
    result = []
    current_fp = []

    this_ground_truth = copy.deepcopy(ground_truth)
    for k in range(reconstruction.shape[0]):
        current_fp = []
        for i in range(reconstruction[k].shape[0]):
            distance = 100
            current_j = -1
            current_ground_truth = this_ground_truth[k][:, 0:2]

            for j in range(current_ground_truth.shape[0]):

                #dis = np.linalg.norm(reconstruction[i] - this_ground_truth[j])
                if np.linalg.norm(reconstruction[k][i] - current_ground_truth[j]) < distance:
                    distance = np.linalg.norm(reconstruction[k][i] - current_ground_truth[j])
                    current_j = j
                    vec = reconstruction[k][i] - current_ground_truth[j]
            if current_j != -1:
                result.append(np.array([*reconstruction[k][i],*vec]))
                this_ground_truth[k] = np.delete(this_ground_truth[k], current_j, axis=0)
            else:
                current_fp.append(reconstruction[k][i])
        f_p.append(np.array(current_fp))
    f_n =0
    for s in this_ground_truth:
        f_n += s.shape[0]
    result = np.array(result)
    f_p = np.array(f_p)
    f_p_count=0
    for s in f_p:
        f_p_count += s.shape[0]
    false_negative = this_ground_truth
    jac = result.shape[0] / (result.shape[0] + f_p_count + f_n)
    error = 0
    vec = []
    for i in result:
        error += i[2] ** 2 + i[3] ** 2
        vec.append(i[-2:])
    print(np.mean(np.array(vec),axis=0))
    rmse = np.sqrt(error / result.shape[0])
    print("Jaccard index: = ", jac, " rmse = ", rmse)

    return result, (f_p,f_p_count), (false_negative,f_n), jac, rmse

def validate_rapidstorm():
    result_list_rapid = [] #jac,rmse,fp,fn,tp
    for i in range(10):
        path = os.getcwd() + r"\test_data\rapidstorm\new\dataset_n"+str(i)+"_free PSF_200-800nm.txt"
        p = read_Rapidstorm(path)
        truth = os.getcwd() + r"\test_data\dataset_n"+str(i)+".npy"  # r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction.npz"
        truth_coords = np.load(truth, allow_pickle=True)-50
        result, false_positive, false_negative, jac, rmse = jaccard_index(p, truth_coords)
        result_list_rapid.append(np.array([jac, rmse, false_positive[1], false_negative[1], result.shape[0]]))
    rapidstorm_final = np.array(result_list_rapid).T

    np.save(os.getcwd() +r"\test_data\rapidstorm_results.npy", rapidstorm_final)

def validate_thunderstorm():
    result_list_th = [] #jac,rmse,fp,fn,tp
    for i in range(10):
        path = os.getcwd() + r"\test_data\thunderstorm_results_dataset"+str(i)+".csv"
        p = read_Thunderstorm(path)
        truth = os.getcwd() + r"\test_data\dataset_n"+str(i)+".npy"  # r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction.npz"
        truth_coords = np.load(truth, allow_pickle=True)
        result, false_positive, false_negative, jac, rmse = jaccard_index(p, truth_coords)
        result_list_th.append(np.array([jac, rmse, false_positive[1], false_negative[1], result.shape[0]]))
    thunderstorm_final = np.array(result_list_th).T

    np.save(os.getcwd() +r"\test_data\thunderstorm_results.npy", thunderstorm_final)

def validate_cs_model():
    cs_net =TrainInceptionNet
    # Create a callback that saves the model's weights every 5 epochs
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
    manager = tf.train.CheckpointManager(ckpt, './cs_training4', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    cs_net.update(150,100)

    denoising = WaveletAI()

    checkpoint_path = "training_lvl3/cp-10000.ckpt"

    denoising.load_weights(checkpoint_path)
    result_list_ai = [] #jac,rmse,fp,fn,tp

    for i in range(10):
        image = os.getcwd() + r"\test_data\dataset_n"+str(i)+".tif"#r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction_flim.tif"
        truth = os.getcwd() + r"\test_data\dataset_n"+str(i)+".npy"#r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction.npz"
        gen = data_generator_coords(image, offset=0)
        image = np.zeros((100, 128, 128, 3))
        #truth = np.zeros((100, 64, 64, 3))
        truth_coords = np.load(truth, allow_pickle=True)
        truth_coords /= 100
        for frame in truth_coords:
            frame[:,0] += 14#thats offset
            frame[:,1] += 14#thats offset

        for i in range(100):
            image[i] = gen.__next__()

        image_tf1 = tf.convert_to_tensor(image[0:100, :, :])
        #image_tf2 = tf.convert_to_tensor(image[90:100, :, :])#todo: use 20% test
        #truth_tf1 = tf.convert_to_tensor(truth[0:90, :, :])
        #truth_tf2 = tf.convert_to_tensor(truth[90:100, :, :])#todo: use 20% test

        data, truth_new, coord_list = bin_localisations_v2(image_tf1, denoising, truth_array=truth_coords[:], th=0.25)
        result_tensor = cs_net.predict(data)
        per_fram_locs = []
        current_frame = 0
        current_frame_locs = []
        limit = [0.8, 1.8, 1.8]
        t_array = truth_new.numpy()
        # for i in range(result_tensor.shape[0]):
        #     plt.imshow(data[i, :, :, 1])
        #     plt.scatter(t_array[i, 1]/8, t_array[i, 0]/8)
        #
        #     for n in range(3):
        #         if result_tensor[i, 6 + n] > 0.1:
        #             plt.scatter(result_tensor[i, 2 * n + 1]/8, result_tensor[i, 2 * n]/8, c="g")
        #     plt.show()


        for i in range(result_tensor.shape[0]):
            if coord_list[i][2] == current_frame:
                for n in range(3):
                    #if result_tensor[i,2*n]/8 >1 and result_tensor[i,2*n+1]/8>1:
                    if result_tensor[i, 6 + n] > limit[n]:
                        current_frame_locs.append(coord_list[i][0:2] + np.array([result_tensor[i,2*n]/8, result_tensor[i,2*n+1]/8]))
            else:
                per_fram_locs.append(np.array(current_frame_locs))
                current_frame = coord_list[i][2]
                current_frame_locs = []
                for n in range(3):
                    #if result_tensor[i,2*n]/8 >1 and result_tensor[i,2*n+1]/8>1:
                    if result_tensor[i, 6 + n] > limit[n]:
                        current_frame_locs.append(coord_list[i][0:2] + np.array([result_tensor[i,2*n]/8, result_tensor[i,2*n+1]/8]))
        #append last frame
        per_fram_locs.append(np.array(current_frame_locs))
        #todo: create a test function for jaccard
        #per_frame_multifit = np.array(multifit)*100
        per_fram_locs = np.array(per_fram_locs)*100
        current_truth_coords = truth_coords[:100]
        current_truth_coords *=100

        result, false_positive, false_negative, jac, rmse = jaccard_index(per_fram_locs, current_truth_coords)
        result_list_ai.append(np.array([jac, rmse, false_positive[1], false_negative[1], result.shape[0]]))

    #todo: plot stuff

    ai_final = np.array(result_list_ai).T

    np.save(os.getcwd() +r"\test_data\ai_results_cs4.npy", ai_final)

def validate_cs_inception_model():
        facade = TrainInceptionNet()
        # checkpoint_path = "cs_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint

        facade.sigma = 150
        facade.wavelet_thresh = 0.2

        #facade.pretrain_current_sigma_d()

        result_list_ai = []  # jac,rmse,fp,fn,tp

        for i in range(10):
            image_p = os.getcwd() + r"\test_data\dataset_n" + str(
                i) + ".tif"  # r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction_flim.tif"
            truth = os.getcwd() + r"\test_data\dataset_n" + str(
                i) + ".npy"  # r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction.npz"
            #gen = data_generator_coords(image, offset=0)
            gen = data_generator_coords(image_p, offset=0)
            image = np.zeros((100, 128, 128, 3))
            # truth = np.zeros((100, 64, 64, 3))
            truth_coords = np.load(truth, allow_pickle=True)
            truth_coords /= 100
            for frame in truth_coords:
                frame[:, 0] += 13.5 # thats offset
                frame[:, 1] += 13.5 # thats offset

            for i in range(100):
                image[i] = gen.__next__()
            # truth = np.zeros((100, 64, 64, 3))
            truth_coords = np.load(truth, allow_pickle=True)
            truth_coords /= 100
            for frame in truth_coords:
                frame[:, 0] += 13.5 # thats offset
                frame[:, 1] += 13.5 # thats offset

            #for i in range(100):
            #    image[i] = gen.__next__()

            image_tf1 = tf.convert_to_tensor(image[0:100, :, :])
            # image_tf2 = tf.convert_to_tensor(image[90:100, :, :])#todo: use 20% test
            # truth_tf1 = tf.convert_to_tensor(truth[0:90, :, :])
            # truth_tf2 = tf.convert_to_tensor(truth[90:100, :, :])#todo: use 20% test
            with TiffFile(image_p) as tif:
                x = tif.asarray()

            result_tensor, coord_list = facade.predict(x, raw=True)
            per_fram_locs = []
            current_frame = 0
            current_frame_locs = []

            thresh = 0.5


            for i in range(result_tensor.shape[0]):
                classifier = uniform_filter(result_tensor[i, :, :, 2], size=3)
                #classifier = result_tensor[i, :, :, 2]

                classifier[np.where(classifier < 0.01)] = 0

                indices = get_coords(classifier).T

                x = result_tensor[i, indices[0], indices[1], 0]
                y = result_tensor[i, indices[0], indices[1], 1]
                dx = result_tensor[i, indices[0], indices[1], 3]  # todo: if present
                dy = result_tensor[i, indices[0], indices[1], 4]
                if coord_list[i][2] == current_frame:

                    for j in range(indices[0].shape[0]):
                        if dx[j] < 0.02 and dy[j] < 0.02:
                            current_frame_locs.append(coord_list[i][0:2] + np.array(
                                [float(indices[0][j]) + x[j] , float(indices[1][j]) + y[j] ]))
                else:
                    per_fram_locs.append(np.array(current_frame_locs))
                    current_frame = coord_list[i][2]
                    current_frame_locs = []

                    for j in range(indices[0].shape[0]):
                        if dx[j] < 0.02 and dy[j] < 0.02:
                            current_frame_locs.append(coord_list[i][0:2] + np.array(
                                [float(indices[0][j]) + x[j] , float(indices[1][j]) + y[j]]))
            # append last frame
            per_fram_locs.append(np.array(current_frame_locs))
            # todo: create a test function for jaccard
            # per_frame_multifit = np.array(multifit)*100
            per_fram_locs = np.array(per_fram_locs) * 100
            current_truth_coords = truth_coords[:100]
            current_truth_coords *= 100

            result, false_positive, false_negative, jac, rmse = jaccard_index(per_fram_locs, current_truth_coords)
            result_list_ai.append(np.array([jac, rmse, false_positive[1], false_negative[1], result.shape[0]]))
            print(false_positive[1], false_negative[1])


        # todo: plot stuff

        ai_final = np.array(result_list_ai).T

        np.save(os.getcwd() + r"\test_data\ai_results_csI.npy", ai_final)





if __name__ == '__main__':
    #create_test_data()
    validate_cs_inception_model()
    # validate_rapidstorm()
    rapid = np.load(os.getcwd() +r"\test_data\rapidstorm_results.npy", allow_pickle=True)
    ai = np.load(os.getcwd() +r"\test_data\ai_results_csI.npy", allow_pickle=True)
    #ai_wave = np.load(os.getcwd() +r"\test_data\ai_results_wave.npy", allow_pickle=True)

    thund = np.load(os.getcwd() +r"\test_data\thunderstorm_results.npy", allow_pickle=True)
    cols = ["jaccard", "rmse", "fp", "fn"]
    for i in range(4):
        fig,axs = plt.subplots()
        axs.plot(ai[i], label="AI")
        #axs.plot(ai_wave[i], label="AIW")
        axs.plot(rapid[i], label="rapid")
        axs.plot(thund[i], label="Thunderstorm")
        axs.set_ylabel(cols[i])
        axs.set_xlabel("switching rate")
        plt.legend()
        plt.show()



    inception = np.load(get_root_path() +r"\trainings\cs_inception\learn_all_new_data_gen_Test\accuracy.npy", allow_pickle=True)
    inception2 = np.load(get_root_path()+r"\trainings\cs_inception\learn_all_new_data_variable_sigma\accuracy.npy",  allow_pickle=True)
    inception_d = np.load(get_root_path() +r"\trainings\cs_inception\learn_all_new_data_variable_sigma_custom_path_extended\accuracy.npy", allow_pickle=True)

    inception_dx = np.load(get_root_path() +r"\trainings\cs_inception\learn_all_new_data_variable_sigma_custom_path_x\accuracy.npy", allow_pickle=True)
    inception_dy = np.load(get_root_path() +r"\trainings\cs_inception\learn_all_new_data_variable_sigma_custom_path_y\accuracy.npy", allow_pickle=True)
    inception_dz = np.load(get_root_path() +r"\trainings\cs_inception\learn_all_new_data_variable_sigma_custom_path_z\accuracy.npy", allow_pickle=True)
    cv = np.load(get_root_path() +r"\trainings\cs_cnn\learn_all_new_data_variable_sigma\accuracy.npy", allow_pickle=True)
    #u = np.load(os.getcwd() +r"\src\trainings\cs_training_u\accuracy.npy", allow_pickle=True)
    #plt.plot(inception[:,0], inception[:,1], label="Inception CS Net")
    #plt.plot(inception2[:,0], inception2[:,1], label="Inception CS Net")
    plt.plot(inception_d[3:60,0], inception_d[3:60,2], label="Inception CS Net custom")

    # plt.plot(inception_dx[:60,0], inception_dx[:60,1], label="Inception CS Net only x")
    # plt.plot(inception_dy[:,0], inception_dy[:,1], label="Inception CS Net only y")
    # plt.plot(inception_dz[:,0], inception_dz[:,1], label="Inception CS Net only z")
    plt.plot(cv[3:60,0], cv[3:60,2], label="CV Net")


    plt.legend()
    plt.ylabel("Jaccard Index")
    plt.xlabel("Iterations")

    plt.show()


