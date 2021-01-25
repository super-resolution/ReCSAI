from src.data import *
from tifffile import TiffWriter
import os

def build_dataset(im_shape, dataset_size, file_name, switching_rate=0.1, on_time=1000,):
    factory = Factory()
    factory.shape= (im_shape*100,im_shape*100)
    factory.image_shape = (im_shape,im_shape)
    image_stack = []
    coord=[]
    points = factory.create_point_set(on_time=on_time)
    init_indices = np.random.choice(points.shape[0], 10)
    on_points = points[init_indices]
    for i in range(dataset_size): #todo: while loop here
        image, truth, on_points = factory.simulate_accurate_flimbi(points, on_points, switching_rate=switching_rate)#todo: simulate off
        image = factory.reduce_size(image)
        image = np.pad(factory.accurate_noise_simulations_camera(image),(14,14))
        truth = np.pad(factory.reduce_size(truth).astype(np.int32),(14,14))
        image_stack.append(image)
        coord.append(np.array(on_points))
    image_array = np.array(image_stack)
    if not os.path.exists(os.getcwd() + r"\test_data"):
        os.mkdir(os.getcwd() + r"\test_data")
    with TiffWriter(os.getcwd() + r"\test_data\\" + file_name + ".tif",
                    bigtiff=True) as tif:
        tif.save(image_array[:, 14:-14, 14:-14], photometric='minisblack')
    np.save(os.getcwd() + r"\test_data\\" + file_name + ".npy", coord)



if __name__ == '__main__':
    for j in range(10):
        build_dataset(100, 100, "dataset_"+str(j), switching_rate=0.1+0.02*j,on_time=1000-100*j)