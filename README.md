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
#### Linux or Windows (with NVIDIA GPU and CUDA 10.0)
```
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
```
#### Linux or Windows (CPU only)
```
pip install torch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
```
#### MacOS (CUDA not supported)
```
pip install torch==1.4.0 torchvision==0.5.0
```

# Usage
## Preparing a dataset
We provide a tutorial [here](./mditre/data_loading_tutorial.ipynb) that describes how to go from a raw dataset to a preprocessed data structured readily accepted by the MDITRE model code.

## Configuration options
MDITRE operation requires a list of configuration options to be passed as arguments as explained [here](https://github.com/gerberlab/mditre/blob/master/mditre/config_doc.pdf). Default values of all options are given in the parse() function [here](https://github.com/gerberlab/mditre/blob/master/mditre/trainer.py).

## Running MDITRE on the prepared dataset
We included a jupyter notebook [here](./mditre/model_run_tutorial.ipynb) describing how to run the model on the prepared dataset from the previous step using default configuration options.

## Rule visualization GUI
We included a jupyter notebook [here](./mditre/rule_visualization_tutorial.ipynb) describing how to use and navigate the GUI for gaining insights into the rules learned by the MDITRE model.