# Scalable learning of interpretable rules for the microbiome time-series domain
We present a new differentiable model that learns human interpretable rules from microbiome time-series data for classifying the status of the human host.

# Installation (Python 3.6 and CUDA 10.0)
First install mditre package from pip via web or source. Then install pytorch.
### Install mditre from pip via web:
```
pip install mditre
```
### Install mditre from pip via source:
```
git clone https://github.com/gerberlab/mditre.git
cd mditre
pip install .
```
### Install pytorch from pip
```
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
```

# Usage
## Preparing a dataset
MDITRE requires a dataset stored as a python pickle file, for example [here](https://github.com/gerberlab/mditre/blob/master/mditre/datasets/david_agg_filtered.pickle). We show how to create this pickle file by considering the dataset from David et al. (2014).
1. Install MITRE in a separate python 2 environment using the steps listed [here](https://github.com/gerberlab/mitre#installation).
2. MITRE operation is controlled by a configuration file as shown [here](https://github.com/gerberlab/mitre#quick-start). We only need to use MITRE data pre-processing utilities so we provide a modified configuration file [here](https://github.com/gerberlab/mditre/tree/master/mditre/datasets/david_reference.cfg).
3. Run mitre with the given configuration file [here](https://github.com/gerberlab/mditre/tree/master/mditre/datasets/david_reference.cfg), which will generate a pickle object containing the MITRE preprocessed dataset given [here](https://github.com/gerberlab/mditre/tree/master/mditre/datasets/david_reference_dataset_object.cfg)
4. Run `python ./mditre/convert_mitre_dataset.py` which converts MITRE dataset object into another pickle file containing the dataset as a dictionary for ease of use as given [here](https://github.com/gerberlab/mditre/blob/master/mditre/datasets/david_agg_filtered.pickle).

## Configuration options
MDITRE operation requires a list of configuration options to be passed as arguments as explained [here](https://github.com/gerberlab/mditre/blob/master/mditre/config_doc.pdf). Default values of all options are given in the parse() function [here](https://github.com/gerberlab/mditre/blob/master/mditre/trainer_model.py).

## Running MDITRE on the prepared dataset
We included a jupyter notebook [here](https://github.com/gerberlab/mditre/blob/master/mditre/demo.ipynb) describing how to run the model on the prepared dataset from the previous step using default configuration options. Make sure to switch to the environment containing the mditre package before running the notebook.