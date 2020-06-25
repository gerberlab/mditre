# Scalable learning of interpretable rules for the microbiome time-series domain
We present a new differentiable model that learns human interpretable rules from microbiome time-series data for classifying the status of the human host.

# Dependencies (Python 3.7.1)
  - ete3==3.1.1
  - matplotlib==3.1.0
  - numpy==1.17.3
  - scikit-learn==0.21.3
  - scipy==1.3.2
  - torch==1.4.0+cu100

# Datasets
Please download the datasets from this googlr drive [link](https://drive.google.com/drive/folders/1x2J7tSSjLhWNEmZDvIKG1Za6YTK13FV8). Makes ure the datasets folder is in the same directoy as the code.

# Training
Run the training code using the following commands for each dataset.
### David
```sh
python trainer.py \
	--data ./datasets/david_agg_filtered.pickle \
	--data_name david \
	--epochs 1000 -b 128 --lr_alpha 0.05  \
	--deterministic --seed 42 \
	--z_mean 0 --z_var 100 --z_r_mean 0 --z_r_var 100
```

### Vatanen
```sh
python trainer.py \
	--data ./datasets/knat_agg_filtered.pickle \
	--data_name knat \
	--epochs 1000 -b 128 --lr_alpha 0.05  \
	--deterministic --seed 42 \
	--cv_type kfold --kfolds 5 \
	--z_mean 0 --z_var 100 --z_r_mean 0 --z_r_var 100
```

### Bokulich
```sh
python trainer.py \
	--data ./datasets/bokulich_diet_agg_filtered.pickle \
	--data_name bokulich_diet \
	--epochs 1000 -b 128 --lr_alpha 0.005  \
	--deterministic --seed 42 \
	--z_mean 0 --z_var 100 --z_r_mean 0 --z_r_var 100
```

# This work is under double-blind peer review, please do not distribute!
