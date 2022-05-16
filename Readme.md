# *ReCSAI:* Recursive Compressed Sensing Artificial Intelligence
is a SMLM (Single Molecule Localisation Microscopy) software to fit the location of sparse fluorescent emitters with subpixel accuracy. The current version is optimized to work with disrupted, nonlinear or varying PSF's occuring in confocal dSTORM measurements. The software currently implements the following features:
* Prefilter ROIs with potential localisations using a trainable wavelet filterbank
* Choose from several CS (Compressive Sensing) based network architectures
* Use the *Emitter* class to filter localisations, apply drift corrections, concatenate sets, save and read localisation files in multiple formats or compute metrics like the jaccard index
* Render your data in a visualization pipeline

## Usage
* Ready to use Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/super-resolution/ReCSAI/blob/airy_disc/notebooks/ReCSAI_reconstruction.ipynb)
* Train on your own network: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/super-resolution/ReCSAI/blob/airy_disc/notebooks/ReCSAI_training.ipynb)

## Local setup
To install ReCSAI on your local machine, you need a CUDA capable GPU and a [Tensorflow](https://www.tensorflow.org/install/gpu) installation.
For the Wavelet prefiltering, [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) is needed. The rest of the packages can be installed using the *requirements.txt*.
1. Create an [Anaconda](https://www.anaconda.com/products/distribution) environment and activate it:
``` 
conda create --name recsai
conda activate recsai
```
2. Clone [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) 
3. Open a command line, *cd* into the cloned repository and run:
```
python setup.py --install
```
4. Clone this repository
5. cd into the cloned folder and run:
```
pip install -r requirements.txt
```
6. Edit *main.py* for your fitting purposes.
## Customizing main.py
1. Import stuff and enable memory growth in your GPU
```
import os
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from plot_emitter_set
from src.trainings.train_cs_net import ResUNetFacade
from src.emitters import Emitter
```
2. Define the path to your data
```
path = r"path_to_your_data.tif"
```
3. Build a network instance and set parameters
```
facade = ResUNetFacade()
facade.sigma = 180
facade.wavelet_thresh = 0.1
```
4. Run and safe the evaluation in a tmp file
```
result_tensor,coord_list = facade.predict(path, raw=True)
if not os.path.exists(os.getcwd()+r"\tmp"):
    os.mkdir(os.getcwd()+r"\tmp")
np.save(os.getcwd()+r"\tmp\current_result.npy",result_tensor)
np.save(os.getcwd()+ r"\tmp\current_coordinate_list.npy", coord_list)

result_tensor = np.load(os.getcwd()+r"\tmp\current_result.npy",allow_pickle=True)
coord_list = np.load(os.getcwd()+ r"\tmp\current_coordinate_list.npy",allow_pickle=True)
```
5. Filter for parameters
```
p_threshold = 0.3
emitter = Emitter.from_result_tensor(result_tensor,p_threshold, coord_list )
print(emitter.xyz.shape[0])
emitter_filtered = emitter.filter(sig_x=0.1, sig_y=0.1, photons=0.1, )
```
5.1. Apply dme drift correct (should work out of the box for windows) [skip if not required]
```
emitter_filtered.use_dme_drift_correct()
```
5.2. Or apply your own drift correction [skip if not required]
```
emitter_filtered.apply_drift(r"your_path.csv")
```
6. Plot stuff
```
plot_emitter_set(emitter_filtered)
```
If something doesnt work out, feel free to contact me.



## Creating data
You can extend the create_data.py file to create custom data for your purposes.
1. Import data generation and visualization:
```
from src.data import *
from src.data import DataGeneration, GPUDataGeneration
from src.visualization import plot_data_gen
```
2. Build a data generator instance:
```
gener = GPUDataGeneration(9)
```
3. Get a small test batch and plot it to visually check your parameters
```
test_datasize = 1
#create a small generator to visually check your parameters
generator, shape = gener.create_data_generator(test_datasize, noiseless_ground_truth=True)
dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32, tf.float32),
                                         output_shapes=shape)
plot_data_gen(dataset)
```
4. Run the dataset generation.
```
gener.create_dataset("test_data_creation")
```

## Running a custom training
A basic file for training is implemented under src.trainings.train_cs_net.py
1. Import the training interface and some helper packages:
```
import os, sys

from src.models.cs_model import CompressedSensingInceptionNet, CompressedSensingCVNet, CompressedSensingUNet,\
    CompressedSensingResUNet, StandardUNet, CompressedSensingConvNet
from src.facade import NetworkFacade
from src.utility import get_root_path
```
2. Set up a facade instance
```
CURRENT_RES_U_PATH = get_root_path()+r"/trainings/cs_u/_final2_training_100_gpu_data"

CURRENT_WAVELET_PATH = get_root_path()+r"/trainings/wavelet/training_lvl5/cp-5000.ckpt"
class ResUNetFacade(NetworkFacade):
    def __init__(self):
        super(ResUNetFacade, self).__init__(CompressedSensingResUNet, CURRENT_RES_U_PATH,
                                           CURRENT_WAVELET_PATH, shape=128)
        self.train_loops = 120
```
3. Start the training
```
training = ResUNetFacade()
training.train_saved_data()
```
If you want to use your own data you can put it under current dataset and update the path in the NetworkFacade initialization


Custom trainings can be implemented by editing train_cs_net.py, creating a folder in datasets and defining this folder as the current dataset in facade.py.
## Paper

