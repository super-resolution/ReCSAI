# Tf Wavelet Layers
(integrations for colab zenodo)
is a Single Molecule Localisation Microscopy software, that combines the following components:
* Efficient prefiltering by learnable wavelet filters
(pics)
* A differentiable compressed sensing algorithm implemented into a neuronal network fro accurate localisation reconstructions
(update)
* A set of localisation filters
(emitter set)
* State of the art rendering of SMLM data
(more like rendering)

## Requirements
Additionally to installing the requirements.txt you need to get:
[tf-wavelets](https://github.com/UiO-CS/tf-wavelets).
## Usage
Tf Wavelet Layers can be used via [Colab](https://colab.research.google.com/drive/1GQI5KXUymahWzkJ_m4ZVx4LPRGPdVbQf?pli=1#scrollTo=j6zaRBylyEpW) or by cloning the git repository and editing the main.py.
Custom trainings can be implemented by editing train_cs_net.py, creating a folder in datasets and defining this folder as the current dataset in facade.py.
