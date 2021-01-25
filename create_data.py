from src.data import *
from tifffile import TiffWriter
import os

def build_dataset(im_shape, dataset_size, file_name, switching_rate=0.1, on_time=1000,):
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
    for j in range(10):
        build_dataset(100, 100, "dataset_"+str(j), switching_rate=0.1+0.02*j,on_time=1000-100*j)