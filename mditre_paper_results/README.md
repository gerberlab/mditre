# Script for MDITRE manuscript results
We provide the required datasets and a script for reproducing MDITRE performance results.

## Usage
The script ```main.py``` runs the experiments 10 times on the specified dataset (located in ```datasets``` folder). Make sure to have MDITRE installed before running ```main.py```.

Due to the file size limit of github, we compressed the semi-synthetic datasets into 2 zip files (```case_1.zip``` and ```case_2.zip```), please make sure to unzip them in the ```datasets/semisyn``` directory.

Here is an example command to run on the dataset from David et al., 2014.
```
python -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 main.py --data ./datasets/david/david_agg_filtered.pickle --data_name david_testing --is_16s
```
This command runs 5-fold cross-validation procedure, using a separate process for each fold. If the hardware has limited cores or memory, setting ```--nproc_per_node=1``` will run the cross-validation serially in a single process.
