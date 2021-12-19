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
## Preparing a dataset
We provide a tutorial [here](./mditre/data_loading_tutorial.ipynb) that describes how to go from a raw dataset to a preprocessed data structured readily accepted by the MDITRE model code.

## Configuration options
MDITRE operation requires a list of configuration options to be passed as arguments as explained [here](https://github.com/gerberlab/mditre/blob/master/mditre/config_doc.pdf). Default values of all options are given in the parse() function [here](https://github.com/gerberlab/mditre/blob/master/mditre/trainer.py).

## Running MDITRE on the prepared dataset
After obtaining the pickle file with the preprocessed data, we are now ready to run the MDITRE model for analysis. The MDITRE model code after a successfull run outputs the following information.
- The predictive ability of the model using F1 score as the quantitative measure
- A serialized pickle object containing all the information needed to interpret and visualize the learned rule.

### Quick start with default options
As a quick start, we recommend running MDITRE on the preprocessed data using cross-validation to assess the generalization capability of the model. We included a python script [here](./mditre/tutorials/model_run_tutorial.py) to run the model on the prepared dataset from the previous step using default configuration options. We provide the following command that executes the script and runs a 5-fold cross-validation procedure using multiprocessing (1 process per fold) and returns logs pertaining to some training diagnostics (useful for debugging) and finally the F1 score. The code runs the fastest with the presence of a GPU. On a CPU, it should still take only a few minutes to run in multiprocessing mode. If the user has limited CPU cores, they can set the "--nproc_per_node" to a lower number. The full description of the command line arguments accepted by the model code is located in the documentation [here](./config_doc.pdf). We provide the short description of required options to be passed as arguments to the command.
- nproc_per_node: Number of parallel processes to be used for cross-validation procedure
- data: Lcoation of the preprocessed data pickle file from the previous step
- data_name: A string used to create output file directory
- is_16s: Required if using 16s based dataset (omit if using Metagenomics based data)

```
python -m torch.distributed.launch --nproc_per_node=5 \
    model_run_tutorial.py --data ./datasets/david_agg_filtered.pickle \
    --data_name david_test --is_16s
```

After executing the above command successfully, you can find the saved pickle file containing the rule information here ```logs/David/seed_42/rank_0/fold_0/rules_dump.pickle```, which will be used in the rule visualization tutorial.

## Rule visualization GUI
We included a jupyter notebook [here](./mditre/tutorials/rule_visualization_tutorial.ipynb) describing how to use and navigate the GUI for gaining insights into the rules learned by the MDITRE model.
