import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
import matplotlib.pyplot as plt
from ete3 import Tree

import torch
from torch.utils.data import Dataset, DataLoader


# Load a datset from a pickle file
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')

    return dataset


# Get OTU indexes from the tree
def get_otu_ids(phylo_tree, var_names):
    otu_names = phylo_tree.get_leaf_names()
    otu_idx = [i for i, n in enumerate(var_names) if n in otu_names]

    return otu_idx


# compute the distance matrix from the tree
def get_dist_matrix(phylo_tree, var_names, otu_idx):
    num_otus = len(otu_idx)
    dist_matrix = np.zeros((num_otus, num_otus), dtype=np.float32)
    for src in phylo_tree.get_leaves():
        i = var_names.index(src.name)
        i = otu_idx.index(i)
        for dst in phylo_tree.get_leaves():
            j = var_names.index(dst.name)
            j = otu_idx.index(j)
            dist_matrix[i, j] = src.get_distance(dst)
    return dist_matrix


# Compute the data matrix, consisting of abundances of OTUs only
# Also compute a mask that contains ones for time points with samples
# else zero
def get_data_matrix(num_otus, num_time, samples, times, otu_idx, scaling_factor=1):
    num_subjects = len(samples)
    X = np.zeros((num_subjects, num_otus, num_time), dtype=np.float32)
    X_mask = np.ones((num_subjects, num_time), dtype=np.float32)
    for i in range(num_subjects):
        x = samples[i]
        t = times[i].astype('int').tolist()
        for j in range(num_otus):
            for k in range(num_time):
                if k in t:
                    X[i, j, k] = x[otu_idx[j], t.index(k)] * scaling_factor
                else:
                    X_mask[i, k] = 0.

    return X, X_mask


# Get stratified kfold train/test splits for cross-val
def cv_kfold_splits(X, y, num_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

    train_ids = list()
    test_ids = list()
    for train_index, test_index in skf.split(X, y):
        train_ids.append(train_index)
        test_ids.append(test_index)

    return train_ids, test_ids


# Get leave-one-out train/test splits for cross-val
def cv_loo_splits(X, y):
    skf = LeaveOneOut()

    train_ids = list()
    test_ids = list()
    for train_index, test_index in skf.split(X, y):
        train_ids.append(train_index)
        test_ids.append(test_index)

    return train_ids, test_ids


# Get stratified train/test splits
def stratified_split(ids, y, test_size=0.1, seed=42):
    train_ids, test_ids = train_test_split(ids,
        test_size=test_size, stratify=y, random_state=seed)

    return train_ids, test_ids


# Get torch data loaders
def get_data_loaders(X, y, X_mask, batch_size, num_workers,
    shuffle=False, pin_memory=False):
    dataset = TrajectoryDataset(X, y, X_mask)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


class TrajectoryDataset(Dataset):
    """Custom torch dataset for abundances, labels and mask"""
    def __init__(self, X, y, mask=None):
        super(TrajectoryDataset, self).__init__()
        self.X = X
        self.y = y
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        traj, label = self.X[idx], self.y[idx]

        if self.mask is not None:
            X_mask = self.mask[idx]
            sample = {'data': traj, 'label': label, 'mask': X_mask}
        else:
            sample = {'data': traj, 'label': label}

        return sample