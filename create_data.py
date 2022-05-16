from src.data import *
from src.data import DataGeneration, GPUDataGeneration
from src.visualization import plot_data_gen
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    gener = GPUDataGeneration(9)
    generator, shape = gener.create_data_generator(1, noiseless_ground_truth=True)
    dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32, tf.float32),
                                             output_shapes=shape)
    plot_data_gen(dataset)

    gener.create_dataset("test_data_creation")