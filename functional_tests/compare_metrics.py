#todo: compute jaccard index

from src.utility import *

from src.utility import bin_localisations_v2, get_coords, get_root_path
import matplotlib.patheffects as pe

class NetworkTrainingMetrics():
    def __init__(self, path, name):
        self.name = name
        data = np.load(path,
            allow_pickle=True)
        self._steps = data[:,0]
        self._jaccard = data[:,1]
        self._rmse = data[:,2]
        self._validation_loss = data[:,3]

    @property
    def jaccard(self):
        return np.stack([self._steps, self._jaccard], axis=0)

    @property
    def rmse(self):
        indices = np.where(np.logical_and(self._rmse<100, self._rmse>5))
        step = self._steps[indices]
        rmse = self._rmse[indices]
        return np.stack([step, rmse], axis=0)

    @property
    def validation_loss(self):
        return np.stack([self._steps, self._validation_loss], axis=0)



if __name__ == '__main__':
    cs_cnn = NetworkTrainingMetrics(get_root_path() + r"\trainings\cs_cnn\_final_training_100_ndata_test_thresholding\accuracy.npy", "CS CNN 100 iterations")

    cs_incept = NetworkTrainingMetrics(get_root_path() + r"\trainings\cs_inception\_final_training_100_10_ndata\accuracy.npy", "CS inception 100 iterations")

    cs_u = NetworkTrainingMetrics(get_root_path() + r"\trainings\cs_u\_final_training_100_ndata\accuracy.npy", "CS U 100 iterations")
    cs_u2 = NetworkTrainingMetrics(get_root_path() + r"\trainings\cs_u\_final2_training_100_ndata_test_compare\accuracy.npy", "CS Res U 4 iterations")
    cs_u2_hit = NetworkTrainingMetrics(get_root_path() + r"\trainings\cs_u\_final2_training_100_ndata_further_higherit\accuracy.npy", "CS Res U 8 iterations")

    #cs_conv = np.load(get_root_path() + r"\trainings\cs_conv\_conv_training_ndata\accuracy.npy",
    #                     allow_pickle=True)
    # cs_100 = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs100\accuracy.npy",
    #     allow_pickle=True)
    # cs_300 = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs_10_z\accuracy.npy",
    #     allow_pickle=True)

    # inc3 = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs_10_inc3\accuracy.npy",
    #     allow_pickle=True)
    # opt = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs_10_opt5e4\accuracy.npy",
    #     allow_pickle=True)
    # cs_100 = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs_100_large_dataset_airy6\accuracy.npy",
    #     allow_pickle=True)
    # bigger_dataset = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs_1_large_dataset_airy6\accuracy.npy",
    #     allow_pickle=True)
    # bigger_dataset2 = np.load(
    #     get_root_path() + r"\trainings\cs_inception\_background_l_cs_10_large_dataset2\accuracy.npy",
    #     allow_pickle=True)
    #cs_10 =  cs_10[3:]    # u = np.load(os.getcwd() +r"\src\trainings\cs_training_u\accuracy.npy", allow_pickle=True)
    # plt.plot(inception[:,0], inception[:,1], label="Inception CS Net")
    # plt.plot(inception2[:,0], inception2[:,1], label="Inception CS Net")
    networks = [cs_cnn,  cs_u, cs_u2, cs_u2_hit, cs_incept]
    names = ["Jaccard Index","RMSE","validation loss"]
    plt.figure(figsize=[6, 5])

    fig, axs  = plt.subplots(3)
    c_cmap = [(.0,0.0,.0,0.8),(0.9,.2,.2,0.8),(0,.0,0.8,0.8)]
    effects = {"lw":2,"path_effects":[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()]}

    x_range = 300
    plots = ["jaccard", "rmse", "validation_loss"]
    for i in range(3):
        j = i+1
        x_max = 999999
        for net in networks:
            data = getattr(net, plots[i])
            if data[0].max()<x_max:
                x_max = data[0].max()
            axs[i].plot(data[0,:x_range],data[1,:x_range], label=net.name, **effects)
        if i<2:
            axs[i].set_xticks([])
        axs[i].set_ylabel(names[i])
        #axs[i].set_xlim(xmin=0 , xmax=x_max)
    axs[2].set_xlabel("steps")
    plt.legend()
    plt.savefig("metrics1.svg")
    plt.show()

    names = ["validation loss"]
    plt.figure(figsize=[6, 5])

    c_cmap = [(.0,0.0,.0,0.8),(0.9,.2,.2,0.8),(0,.0,0.8,0.8)]
    effects = {"lw":2,"path_effects":[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()]}

    #x_range = 80
    plots = ["validation_loss"]
    for i in range(1):
        j = i+1
        x_max = 999999
        for net in networks:
            data = getattr(net, plots[i])
            if data[0].max()<x_max:
                x_max = data[0].max()
            plt.plot(data[0,:x_range],data[1,:x_range], label=net.name, **effects)

            plt.ylabel(names[i])
            plt.xlim(xmin=0 , xmax=x_max)
        plt.xlabel("steps")
    plt.legend()
    plt.savefig("metrics1.svg")
    plt.show()

    plt.figure(figsize=[6, 5])
    for net in networks:
        plt.scatter(net._rmse[-10], net._jaccard[-10], label=net.name, marker="X")
    plt.legend()
    plt.ylabel("Jaccard")
    plt.xlabel("RMSE")
    plt.savefig("metrics2.svg")
    plt.show()
    #todo: scatter plot



    # plt.plot(inception_dx[:60,0], inception_dx[:60,1], label="Inception CS Net only x")
    # plt.plot(inception_dy[:,0], inception_dy[:,1], label="Inception CS Net only y")
    # plt.plot(inception_dz[:,0], inception_dz[:,1], label="Inception CS Net only z")

    #plt.ylabel("Jaccard Index")
    #plt.xlabel("Iterations")

