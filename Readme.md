

# Tf Wavelet Layers
is a SMLM (Single Molecule Localisation Microscopy) software, to fit the location sparse fluorescent emitters with subpixel accuracy. The current version is optimized to work with disrupted, nonlinear or varying PSF's occuring in confocal dSTORM measurements. The software currently implements the following features:
* Prefilter ROIs with potential localisations using a trainable wavelet filterbank
* Choose from several CS (Compressive Sensing) based network architectures
* Use the Emitter class to filter localisations, apply drift corrections, concatenate sets, save and read localisation files in multiple formats or compute metrics like the jaccard index
* Render your data in a visualization pipeline
* Ready to use Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GQI5KXUymahWzkJ_m4ZVx4LPRGPdVbQf#scrollTo=j6zaRBylyEpW)
* Train on your own data:

## Requirements
Additionally to installing the requirements.txt you need to get:
[tf-wavelets](https://github.com/UiO-CS/tf-wavelets).
## Usage
Tf Wavelet Layers can be used via [Colab](https://colab.research.google.com/drive/1GQI5KXUymahWzkJ_m4ZVx4LPRGPdVbQf?pli=1#scrollTo=j6zaRBylyEpW) or by cloning the git repository and editing the main.py.
Custom trainings can be implemented by editing train_cs_net.py, creating a folder in datasets and defining this folder as the current dataset in facade.py.
## Paper