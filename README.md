# MDITRE: scalable and interpretable machine learning for predicting host status from temporal microbiome dynamics
We present a new differentiable model that learns human interpretable rules from microbiome time-series data for classifying the status of the human host.

# Installation (Python 3.6+ and CUDA 10.0)
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
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
#### MacOS (CUDA not supported)
```
pip install torch==1.4.0 torchvision==0.5.0
```

# Usage
## MDITRE workflow on 16S rRNA and shotgun metagenomics data 
We provide 2 tutorials, one for 16s-based data [here](https://github.com/gerberlab/mditre/blob/master/mditre/tutorials/Tutorial_Bokulich_16S_data.ipynb) and another for shotgun metagenomics (Metaphlan) based data [here](https://github.com/gerberlab/mditre/blob/master/mditre/tutorials/Tutorial_2_metaphlan_data.ipynb), which show how to use MDITRE for data loading and preprocessing, running the model code and using the GUI to interpret the learned rules for post-hoc analysis.

## Configuration options
MDITRE operation requires a list of configuration options to be passed as arguments as explained [here](https://github.com/gerberlab/mditre/blob/master/mditre/docs/config_doc.pdf).
