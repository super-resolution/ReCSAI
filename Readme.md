

# ReCSAI Recurrent Compressed Sensing Artificial Intelligence
is a SMLM (Single Molecule Localisation Microscopy) software, to fit the location sparse fluorescent emitters with subpixel accuracy. The current version is optimized to work with disrupted, nonlinear or varying PSF's occuring in confocal dSTORM measurements. The software currently implements the following features:
* Prefilter ROIs with potential localisations using a trainable wavelet filterbank
* Choose from several CS (Compressive Sensing) based network architectures
* Use the Emitter class to filter localisations, apply drift corrections, concatenate sets, save and read localisation files in multiple formats or compute metrics like the jaccard index
* Render your data in a visualization pipeline

## Usage
* Ready to use Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GQI5KXUymahWzkJ_m4ZVx4LPRGPdVbQf#scrollTo=j6zaRBylyEpW)
* Train on your own data: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mo8UzX817JoE1EkLN0oLbdp--PvcW4Z3#scrollTo=fDwyeQv8z5n6)

## Local setup
To install ReCSAI on your local machine you need a CUDA capable GPU, a [tensorflow](https://www.tensorflow.org/install/gpu) installation.
For the Wavelet prefiltering [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) is needed. The rest of the packages can be installed using the requirements.txt.
1. Open [tensorflow](https://www.tensorflow.org/install/gpu) and follow the instructions to install tensorflow on your machine
2. Clone [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) 
3. Open a command line window cd into the cloned repository and run:
```
python setup.py --install
```
4. Clone this repository
5. cd into the cloned folder and run:
```
pip install -r requirements.txt
```
6. Edit the main.py for your fitting purposes.

Custom trainings can be implemented by editing train_cs_net.py, creating a folder in datasets and defining this folder as the current dataset in facade.py.
## Paper