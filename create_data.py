from src.data import *
from tifffile import TiffWriter
import os
import matplotlib.pyplot as plt

def create_crop_dataset(iterations):
    data_train = []
    data_truth = []
    sig = []
    data_noiseless = []

    for j in range(60):
        sigma = np.random.randint(100, 250) #todo: vary sigma with dataset
        generator = crop_generator_u_net(9, sigma_x=sigma, noiseless_ground_truth=True)
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32),
                                                 output_shapes=((1 * 100, 9, 9, 3), (1 * 100, 9, 9, 4), (1 * 100, 9, 9, 3)))
        for train_image, truth, noiseless in dataset.take(4):
            data_train.append(train_image.numpy())
            data_truth.append(truth.numpy())
            data_noiseless.append(noiseless.numpy())
            #fig, axs = plt.subplots(2)
            # axs[0].imshow(train_image.numpy()[0,:,:,1])
            # axs[1].imshow(truth.numpy()[0,:,:,1])
            # plt.show()
            sig.append(sigma)
    return np.array(data_train), data_truth, sig, np.array(data_noiseless)


def build_dataset(im_shape, dataset_size, file_name, switching_rate=0.1, on_time=30,):
    factory = Factory()
    #define ground truth size in nm
    factory.shape= (im_shape*100,im_shape*100)
    #define pixel count of simulated image
    factory.image_shape = (im_shape,im_shape)
    image_stack = []
    coord=[]
    #create points with poisson distributed "on_time"
    points = factory.create_point_set(on_time=on_time)
    init_indices = np.random.choice(points.shape[0], 10)
    on_points = points[init_indices]
    for i in range(dataset_size):
        #simulate flimbi detector painting localisations updating on points
        image, truth, on_points = factory.simulate_accurate_flimbi(points, on_points, switching_rate=switching_rate)
        #resize from nanometer space to pixel space
        image = factory.reduce_size(image)
        #pad image
        image = np.pad(factory.accurate_noise_simulations_camera(image),(14,14))
        #pad ground truth
        image_stack.append(image)
        #save painted localizations of this frame
        coord.append(np.array(on_points))
    image_array = np.array(image_stack)
    #save data
    if not os.path.exists(os.getcwd() + r"\test_data"):
        os.mkdir(os.getcwd() + r"\test_data")
    with TiffWriter(os.getcwd() + r"\test_data\\" + file_name + ".tif",
                    bigtiff=True) as tif:
        tif.save(image_array[:, 14:-14, 14:-14], photometric='minisblack')
    np.save(os.getcwd() + r"\test_data\\" + file_name + ".npy", coord)



if __name__ == '__main__':
    data = np.load("crop_dataset.npy", allow_pickle=True)

    data, truth, sigma, noiseless = create_crop_dataset(1)
    np.save("crop_dataset_train_VS.npy", data)#NS = non switching
    np.save("crop_dataset_truth_VS.npy", truth)#VS = variable sigma
    np.save("crop_dataset_noiseless_VS.npy", noiseless)#VS = variable sigma

    np.save("crop_dataset_sigma_VS.npy", sigma)


    # for j in range(10):
    #     build_dataset(100, 100, "dataset_"+str(j), switching_rate=0.1+0.02*j,on_time=1000-100*j)