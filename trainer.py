# Import required libraries
import argparse
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import time
import datetime
import logging
from copy import deepcopy
import warnings
import random
import textwrap

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import scipy.stats as stats
from ete3 import Tree, TreeStyle, TextFace, NodeStyle, Face, ClusterTree, ProfileFace, add_face_to_node
from matplotlib.legend_handler import HandlerTuple
import matplotlib
matplotlib.use('agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from data import *
from models import MyModel, binary_concrete
from utils import AverageMeter

warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    parser.add_argument('--data', metavar='DIR',
                        default='./datasets/david_agg_filtered.pickle',
                        help='path to dataset')
    parser.add_argument('--data_name', default='David', type=str,
                        help='Name of the dataset, will be used for log dirname')
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr_kappa', default=0.1, type=float,
                        help='Initial learning rate for kappa.')
    parser.add_argument('--lr_time', default=0.1, type=float,
                        help='Initial learning rate for mu and sigma.')
    parser.add_argument('--lr_thresh', default=0.005, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--lr_alpha', default=0.05, type=float,
                        help='Initial learning rate for binary concrete logits on detectors.')
    parser.add_argument('--lr_beta', default=0.01, type=float,
                        help='Initial learning rate for binary concrete logits on rules.')
    parser.add_argument('--lr_fc', default=0.01, type=float,
                        help='Initial learning rate for linear classifier weights and bias.')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='Set random seed for reproducibility')
    parser.add_argument('--max_tau', default=2, type=float,
                        help='Max Temperature on binary concrete')
    parser.add_argument('--min_tau', default=0.01, type=float,
                        help='Min Temperature on binary concrete')
    parser.add_argument('--min_k_otu', default=10, type=float,
                        help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--max_k_otu', default=100, type=float,
                        help='Min Temperature on heavyside logistic for otu selection')
    parser.add_argument('--min_k_time', default=5, type=float,
                        help='Max Temperature on heavyside logistic for time selection')
    parser.add_argument('--max_k_time', default=10, type=float,
                        help='Min Temperature on heavyside logistic for time selection')
    parser.add_argument('--min_k_thresh', default=100, type=float,
                        help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--max_k_thresh', default=1000, type=float,
                        help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--max_k_bc', default=100, type=float,
                        help='Max Temperature on heavyside logistic for binary concretes')
    parser.add_argument('--cv_type', type=str, default='loo',
                        choices=['loo', 'kfold', 'None'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--scaling_factor', default=1, type=float,
                        help='Scale data by this factor')
    parser.add_argument('--z_mean', type=float, default=20,
                        help='NBD Mean active detectors per rule')
    parser.add_argument('--z_var', type=float, default=25,
                        help='NBD variance of active detectors per rule')
    parser.add_argument('--z_r_mean', type=float, default=20,
                        help='NBD Mean active rules')
    parser.add_argument('--z_r_var', type=float, default=25,
                        help='NBD variance of active rules')

    args = parser.parse_args()
    return args


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        # Create logger
        logging.basicConfig(format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.INFO)

        # print cmdline args
        self.logger.info(self.args)

        # Check for gpu availability
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            # Check for cudnn
            assert torch.backends.cudnn.enabled, \
                   "Need cudnn backend to be enabled!"
            cudnn.benchmark = True

        # Check for determinism in training, set random seed
        if self.args.deterministic:
            self.set_rng_seed(self.args.seed)

        # Check for gpu, assign device
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.logger.info('Using device: %s' % (self.device))

        self.logger.info('Trainer object initialized!')


    def load_data(self):
        # Load dataset
        data_dict = self.load_mitredata()

        self.logger.info('Loaded and preprocessed dataset!')


    def load_mitredata(self):
        # Load data from a pickle file
        dataset = load_from_pickle(self.args.data)

        # Number of variables (total nodes in phylo tree)
        num_vars = int(dataset['n_variables'])
        # Total experiment duration
        num_time = int(dataset['experiment_end'] - dataset['experiment_start'] + 1)
        # Total number of subjects in the dataset
        num_subjects = int(dataset['n_subjects'])
        # Abundance samples of each subject
        # list of numpy arrays, each array is abundance of variables over time
        samples = dataset['X']
        # Subject outcomes
        # Add 0. to convert to float
        y = dataset['y'] + 0.
        # Phylogenetic tree of variables
        phylo_tree = dataset['variable_tree']
        # List of numpy arrays timestamps of samples for each subject
        # Experiment start time could be negative, so convert to non-negative
        times = [t - dataset['experiment_start'] for t in dataset['T']]
        # List of variable names in the phylogenetic tree
        # For example [otu0001, otu0002, 12343,...]
        var_names = dataset['variable_names']
        # Dict of variable annotations in the phylogenetic tree
        # For example {'otu0001': 'OTU mapped to Bilophilia Wadsworthia',...}
        var_annot = dataset['variable_annotations']

        # Preprocess data
        self.preprocess_data(phylo_tree, var_names, num_time, samples, times)

        # save data as class variables for later use
        self.y = y
        self.times = times
        self.num_time = num_time
        self.phylo_tree = phylo_tree
        self.var_names = var_names
        self.var_annot = var_annot
        self.num_subjects = num_subjects
        if 'david' in self.args.data:
            self.label_0 = 'Animal diet'
            self.label_1 = 'Plant diet'
        elif 'knat' in self.args.data:
            self.label_0 = 'Finnish/Estonian'
            self.label_1 = 'Russian'
        elif 'bokulich_diet' in self.args.data:
            self.label_0 = 'Breast milk diet'
            self.label_1 = 'Formula diet'
        elif 'digiulio' in self.args.data:
            self.label_0 = 'Normal delivery'
            self.label_1 = 'Premature delivery'
        else:
            raise ValueError('Invalid dataset!')

        # print dataset info
        self.logger.info('Dataset: %s Variables: %d, Otus: %d,\
            Subjects: %d, Total samples: %d' % 
            (self.args.data_name, dataset['n_variables'],
            self.num_otus, dataset['n_subjects'],
            sum(map(len, dataset['T']))))
        self.logger.info('Outcomes: {}'.format(np.unique(y, return_counts=True)))
        self.logger.info('Exp start: {} Exp end: {}'.format(dataset['experiment_start'],
            dataset['experiment_end']))


    def preprocess_data(self, phylo_tree, var_names, num_time, samples, times):
        # Get the number of OTUS (leaf nodes in the tree)
        num_otus = len(phylo_tree)

        # Get the indices of OTUs by matching names of leaves
        # in the tree with their ids in var_names array
        # These ids are used to extract the abundances of
        # OTUs from the full data
        otu_idx = get_otu_ids(phylo_tree, var_names)

        # Get pairwise phylo. distance matrix between OTUs from the tree
        dist_matrix = get_dist_matrix(phylo_tree, var_names, otu_idx)

        # Get abundance and mask after some preprocessing
        # Data matrix is preprocessed as an numpy array
        # of size [num_subjects, num_otus, exp_duration]
        # Mask matrix is an indicator matrix of same size as data
        # Subjects are irregularly sampled, so mask indicates presence of samples
        # Scaling factor is used to scale the data, for example
        # relative abundance to percentange relative abudance
        X, X_mask = get_data_matrix(num_otus, num_time,
            samples, times, otu_idx,
            scaling_factor=self.args.scaling_factor)

        # save data as class variables for later use
        self.X = X
        self.X_mask = X_mask
        self.num_otus = num_otus
        self.dist_matrix = dist_matrix
        self.otu_idx = otu_idx


    def get_cv_splits(self):
        # Get train-test data splits for cross-val
        if self.args.cv_type == 'None':
            self.train_splits = [np.arange(self.num_subjects)]
            self.test_splits = self.train_splits
        elif self.args.cv_type == 'loo':
            self.train_splits, self.test_splits = cv_loo_splits(self.X, self.y)
        else:
            self.train_splits, self.test_splits = cv_kfold_splits(self.X, self.y,
                num_splits=self.args.kfolds, seed=self.args.seed)


    def set_model_hparams(self):
        # Number of rules
        self.num_rules = 10
        # Number of detectors per rule
        # For now each OTU is a detector
        self.num_detectors = self.num_otus

        ### Prior hyperparams ###
        # OTU aggregation bandwidth (kappa)
        # Get distribution of nth percentile distance
        dist_prior = 30
        dist_percentile = np.percentile(self.dist_matrix, dist_prior, axis=1)
        # Use median of the above distribution as the prior mean
        self.kappa_prior_mean = np.median(dist_percentile)
        # Scaled inv-chi-sq dof for initialization
        self.kappa_init_nu = 1000
        # Scaled inv-chi-sq dof for prior loss
        self.kappa_loss_nu = 5
        # Scaled inv-chi-sq tau-sq for initialization
        self.kappa_init_tausq = self.invschisq_tausq(self.kappa_init_nu, self.kappa_prior_mean)
        # Scaled inv-chi-sq tau-sq for prior loss
        self.kappa_loss_tausq = self.invschisq_tausq(self.kappa_loss_nu, self.kappa_prior_mean)

        # Time aggregation bandwith (sigma)
        # nth% of the total experiment duration as the mean
        time_prior = 0.3
        self.sigma_prior_mean = (time_prior * self.num_time)
        # dof for init
        self.sigma_init_nu = 1000
        # dof for loss
        self.sigma_loss_nu = 5
        # tau-sq for init
        self.sigma_init_tausq = self.invschisq_tausq(self.sigma_init_nu, self.sigma_prior_mean)
        # tau-sq for loss
        self.sigma_loss_tausq = self.invschisq_tausq(self.sigma_loss_nu, self.sigma_prior_mean)

        # Set mean and variance for gaussian prior on regression weights
        self.wts_mean = 0
        self.wts_var = 1e5

        # Set mean and variance for neg bin prior for detectors
        self.z_mean = self.args.z_mean
        self.z_var = self.args.z_var

        # Set mean and variance for neg bin prior for rules
        self.z_r_mean = self.args.z_r_mean
        self.z_r_var = self.args.z_r_var


    def init_kappa(self):
        kappa_init = stats.invgamma.rvs(self.kappa_init_nu / 2.,
            scale=(self.kappa_init_nu / 2.) * self.kappa_init_tausq,
            size=(self.num_rules, self.num_detectors))
        return kappa_init


    def init_sigma(self):
        sigma_init = stats.invgamma.rvs(self.sigma_init_nu / 2.,
            scale=(self.sigma_init_nu / 2.) * self.sigma_init_tausq,
            size=(self.num_rules, self.num_detectors))
        return sigma_init


    def get_detec_otu_ids(self):
        # For each detector, compute otu ids within kappa init distance
        # for threshold init
        detector_otuids = list()
        for i in range(self.num_otus):
            otu_incl = list()
            for j in range(self.num_otus):
                if self.dist_matrix[i, j] <= self.kappa_prior_mean:
                    otu_incl.append(j)
            detector_otuids.append(otu_incl)
        return detector_otuids


    def train_loop(self):
        # Variables for storing cross-val results
        k_fold_test_preds = list()
        k_fold_test_true = list()
        k_fold_test_prob = list()
        k_fold_val_preds = list()
        k_fold_val_true = list()
        k_fold_val_prob = list()
        best_models = list()

        # Init model hyperparams
        self.set_model_hparams()

        # Init kappa
        kappa_init = self.init_kappa()

        # Init sigma
        sigma_init = self.init_sigma()

        # Get detector otu ids for thresh init
        detector_otuids = self.get_detec_otu_ids()

        # Initialize model and optimizer (dummy)
        # Just init as a dummy model, later in training loop
        # re-initialize data dependent params
        init_model = MyModel(self.num_rules, self.num_detectors,
            self.num_time, self.dist_matrix)
        # Initiazation dict
        init_args = {
            'kappa_init': kappa_init,
            'mu_init': np.random.normal(0, 1, (self.num_rules, self.num_detectors)),
            'sigma_init': sigma_init,
            'thresh_init': np.random.normal(0, 1, (self.num_rules, self.num_detectors)),
        }
        init_model.init_params(init_args)
        self.init_model_state = init_model.state_dict()
        init_model.to(self.device)

        # Init cross-val data splits
        self.get_cv_splits()

        full_loader = get_data_loaders(self.X,
            self.y,
            self.X_mask, len(self.X), self.args.workers,
            shuffle=False, pin_memory=self.use_gpu)

        start = time.time()

        # Cross-val training loop
        for i, (train_ids, test_ids) in enumerate(zip(self.train_splits, self.test_splits)):
            # if self.y[test_ids]:
            test_loader = get_data_loaders(self.X[test_ids], self.y[test_ids],
                self.X_mask[test_ids],
                self.args.batch_size, self.args.workers,
                shuffle=False, pin_memory=self.use_gpu)

            train_nested_splits, val_splits = cv_kfold_splits(train_ids, self.y[train_ids],
                num_splits=5, seed=self.args.seed)

            best_nested_val_loss = np.inf
            best_model = None

            # Nested cross-val for model selection/hparam tuning
            for j, (t_ids, v_ids) in enumerate(zip(train_nested_splits, val_splits)):
                # Get nested train and val ids
                train_nested_ids = train_ids[t_ids]
                val_ids = train_ids[v_ids]

                # Init time filter centers on time points of samples
                # Centers are init on time points with atleast
                # median number of samples
                time_unique = np.unique(np.concatenate(np.array(self.times)[train_nested_ids]))
                # Init mu
                mu_init = np.random.choice(time_unique, (self.num_rules, self.num_detectors))

                # Mean abund for each OTU at every sample of train set
                # Do not consider contributions from subjects with no
                # samples at a time point
                X_train_sum = self.X[train_nested_ids].sum(axis=0)
                X_train_mean = np.zeros((self.num_otus, self.num_time), dtype=np.float32)
                X_train_mask_sum = self.X_mask[train_nested_ids].sum(axis=0)
                for l in time_unique.astype('int'):
                    X_train_mean[:, l] = X_train_sum[:, l] / X_train_mask_sum[l]

                # Init threshold for avg abun and slope
                thresh_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                self.thresh_init = thresh_init
                self.X_train_mean = torch.from_numpy(np.median(X_train_mean, axis=-1)).to(self.device)
                # Get mean abundance for threshold init at each center,
                # within prior time window
                # and for otus within prior mean distance
                window_start = np.clip(np.floor(mu_init - (self.sigma_prior_mean / 2.)), 0, self.num_time).astype('int')
                window_end = np.clip(np.ceil(mu_init + (self.sigma_prior_mean / 2.)), 0, self.num_time).astype('int')
                for l in range(self.num_rules):
                    for m in range(self.num_detectors):
                        thresh_init[l, m] = X_train_mean[detector_otuids[m],
                        window_start[l, m]:window_end[l, m]].mean()

                # Initiazation dict
                init_args = {
                    'kappa_init': kappa_init,
                    'mu_init': mu_init,
                    'sigma_init': sigma_init,
                    'thresh_init': thresh_init,
                }

                # train-val-test data loaders
                train_nested_loader = get_data_loaders(self.X[train_nested_ids],
                    self.y[train_nested_ids],
                    self.X_mask[train_nested_ids], self.args.batch_size, self.args.workers,
                    shuffle=True, pin_memory=self.use_gpu)
                val_loader = get_data_loaders(self.X[val_ids], self.y[val_ids],
                    self.X_mask[val_ids],
                    self.args.batch_size, self.args.workers,
                    shuffle=False, pin_memory=self.use_gpu)

                # Init model
                model = MyModel(self.num_rules, self.num_detectors,
                    self.num_time, self.dist_matrix)
                model.load_state_dict(self.init_model_state)
                model.init_params(init_args)
                model.to(self.device)

                # Inner loop training, store best model
                best_model_nested = self.train_model(model, train_nested_loader,
                    val_loader, test_loader, i)

                # Eval on val set to get f1
                # track the best model
                val_loss, val_f1 = self.eval(best_model_nested, val_loader)
                if val_loss < best_nested_val_loss:
                    best_model = deepcopy(best_model_nested)
                    probs, true = self.eval_with_preds(best_model_nested, val_loader)
                    best_val_preds = (np.array(probs) > 0.5).tolist()
                    best_val_true = true
                    best_nested_val_loss = val_loss


            # Eval on test set to get f1
            probs, true = self.eval_with_preds(best_model, test_loader)
            k_fold_test_prob.extend(probs)
            k_fold_test_preds.extend((np.array(probs) > 0.5).tolist())
            k_fold_test_true.extend(true)
            k_fold_val_preds.extend(best_val_preds)
            k_fold_val_true.extend(best_val_true)

            self.logger.info(k_fold_test_prob)
            self.logger.info(k_fold_test_preds)
            self.logger.info(k_fold_test_true)

            best_models.append(best_model)

            self.show_rules(best_model, full_loader, i)


        # Compute final cv f1 score
        cv_f1_val = f1_score(k_fold_val_true, k_fold_val_preds)
        cv_f1 = f1_score(k_fold_test_true, k_fold_test_preds)
        clf_report = classification_report(k_fold_test_true, k_fold_test_preds)
        tn, fp, fn, tp = confusion_matrix(k_fold_test_true, k_fold_test_preds).ravel()

        end = time.time()

        self.logger.info('CV F1 score: %.2f' % (cv_f1))
        self.logger.info('Val CV F1 score: %.2f' % (cv_f1_val))
        self.logger.info(k_fold_test_preds)
        self.logger.info(k_fold_test_true)
        self.logger.info(k_fold_test_prob)
        self.logger.info(clf_report)
        self.logger.info('FP: {} FN: {}'.format(fp, fn))
        self.logger.info('Total train time: %.2f hrs' % ((end - start) / 3600.))

        # Stats/viz for the best model over the whole dataset
        best_final_model = None
        best_final_f1 = 0.
        total_mean_rules = 0.
        total_mean_detecs = 0.

        for i, model in enumerate(best_models):
            model.eval()
            probs, true = self.eval_with_preds(model, full_loader)
            model_f1 = f1_score(true, (np.array(probs) > 0.5).tolist())
            mean_rules = 0.
            mean_detecs = 0.
            for j, r in enumerate(model.fc.z_r):
                if r > 0.5:
                    rec_rules = True
                    for k, d in enumerate(model.rules.z[j]):
                        if d > 0.5:
                            if rec_rules:
                                mean_rules += 1
                                rec_rules = False
                            mean_detecs += 1

            total_mean_rules += mean_rules
            total_mean_detecs += mean_detecs

            if model_f1 > best_final_f1:
                best_final_f1 = model_f1
                best_final_model = model

        total_mean_rules /= len(best_models)
        if total_mean_rules > 0 and total_mean_detecs > 0:
            total_mean_detecs /= total_mean_rules
            total_mean_detecs /= len(best_models)

        self.logger.info('Best f1 on full dataset: {}'.format(best_final_f1))
        self.logger.info('Mean rules over all kfolds: {}'.format(total_mean_rules))
        self.logger.info('Mean detectors over all kfolds: {}'.format(total_mean_detecs))

        self.show_rules(best_final_model, full_loader, 'final')

        return cv_f1_val


    def train_model(self, model, train_loader, val_loader, test_loader, outer_fold):
        best_val_loss = np.inf
        best_val_f1 = 0

        # Init optimizer and lr scheduler
        optimizer = optim.Adam([
            {'params': [model.spat_attn.kappa], 'lr': self.args.lr_kappa},
            {'params': [model.time_attn.mu, model.time_attn.sigma], 'lr': self.args.lr_time},
            {'params': [model.thresh_func.thresh], 'lr': self.args.lr_thresh},
            {'params': [model.rules.alpha], 'lr': self.args.lr_alpha},
            {'params': [model.fc.weight, model.fc.bias], 'lr': self.args.lr_fc},
            {'params': [model.fc.beta], 'lr': self.args.lr_beta},
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         self.args.epochs)

        # Training loop
        for epoch in range(self.args.epochs):
            train_loss_avg = AverageMeter()
            val_loss_avg = AverageMeter()
            train_f1_avg = AverageMeter()

            # Compute temperature for binary concrete
            # Linearly anneal from t_max to t_min every epoch
            t = self.linear_anneal(epoch, self.args.min_tau,
                self.args.max_tau, self.args.epochs,
                1, self.args.max_tau)
            approx_param_otu = self.linear_anneal(epoch, self.args.max_k_otu,
                self.args.min_k_otu, self.args.epochs,
                1, self.args.min_k_otu)
            approx_param_time = self.linear_anneal(epoch, self.args.max_k_time,
                self.args.min_k_time, self.args.epochs,
                1, self.args.min_k_time)
            approx_param_thresh = self.linear_anneal(epoch, self.args.max_k_thresh,
                self.args.min_k_thresh, self.args.epochs,
                1, self.args.min_k_thresh)

            # Switch model to train mode
            model.train()

            # Mini-batch training loop
            for i, batch in enumerate(train_loader):
                # Get data, labels, mask tensors
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                # Forward pass
                outputs, [spat_attn, time_attn, z, z_r] = model(
                    data, mask, t=t,
                    otu_k=approx_param_otu,
                    time_k=approx_param_time,
                    thresh_k=approx_param_thresh,
                    bc_k=self.args.max_k_bc,
                )

                # Zero previous grads
                optimizer.zero_grad()

                # Loss function on labels
                train_loss = self.ce_loss(outputs, labels)

                # Priors / regularizers
                negbin_zr_loss = self.negbin_loss(z_r.sum(dim=-1), self.z_r_mean, self.z_r_var)
                l2_wts_loss = self.l2_loss(model.fc.weight, self.wts_mean, self.wts_var)
                negbin_z_loss = self.negbin_loss(z.sum(dim=-1), self.z_mean, self.z_var)

                reg_loss = negbin_zr_loss + negbin_z_loss + l2_wts_loss

                # Backprop for computing grads
                loss = train_loss + reg_loss
                loss.backward()

                # Update parameters
                optimizer.step()

                # Clip mu within exp duration range
                model.time_attn.mu.data.clamp_(0, self.num_time - 1)

                # track avg loss, f1
                train_loss_avg.update(train_loss, 1)
                train_f1 = f1_score(labels.detach().cpu().numpy(),
                    outputs.argmax(dim=-1).detach().cpu().numpy() > 0.5)
                train_f1_avg.update(train_f1, 1)

            # Evaluate on val data
            val_loss, val_f1 = self.eval(model, val_loader)
            # Save stats and model for best val performance
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                best_model = deepcopy(model)

            # Track test data stats
            test_loss, test_f1 = self.eval(model, test_loader)

            # Anneal learning rate
            scheduler.step()

            # Print epoch stats
            kfold_epochs = 'Outer kfold: %d Epoch: %d '
            train_stats = 'TrainLoss: %.2f TrainF1: %.2f '
            val_stats = 'ValLoss: %.2f ValF1: %.2f '
            test_stats = 'TestLoss: %.2f TestF1: %.2f '
            log_str = kfold_epochs + train_stats + val_stats + test_stats
            self.logger.info(
                log_str % (outer_fold, epoch,
                    train_loss_avg.avg, train_f1_avg.avg,
                    val_loss, val_f1, test_loss, test_f1
                )
            )

        return best_model


    def eval(self, model, val_loader):
        val_preds = list()
        val_true = list()
        val_loss_avg = AverageMeter()

        # Evaluate model on val data
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                outputs, [spat_attn, time_attn, z, z_r] = model(data, mask,
                    t=self.args.min_tau,
                    otu_k=self.args.max_k_otu,
                    time_k=self.args.max_k_time,
                    thresh_k=self.args.max_k_thresh,
                    bc_k=self.args.max_k_bc)

                val_loss = self.ce_loss(outputs, labels)

                val_loss_avg.update(val_loss, 1)
                val_preds.extend(outputs.argmax(dim=-1).detach().cpu().numpy() > 0.5)
                val_true.extend(labels.detach().cpu().numpy())

        if len(val_true) > 1:
            val_f1 = f1_score(val_true, val_preds)
        else:
            val_f1 = float(labels.detach().cpu().numpy() == (outputs.argmax(dim=-1).detach().cpu().numpy() > 0.5))

        return val_loss_avg.avg, val_f1


    def eval_with_preds(self, model, val_loader):
        probs = list()
        true = list()
        model.eval()
        with torch.no_grad():
            for l, batch in enumerate(val_loader):
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                outputs, [spat_attn, time_attn, z, z_r] = model(data, mask,
                    t=self.args.min_tau,
                    otu_k=self.args.max_k_otu,
                    time_k=self.args.max_k_time,
                    thresh_k=self.args.max_k_thresh,
                    bc_k=self.args.max_k_bc)

                probs.extend(outputs.argmax(dim=-1).detach().cpu().numpy())
                true.extend(labels.detach().cpu().numpy())

        return probs, true


    # cross entropy loss
    def ce_loss(self, logits, labels):
        return F.cross_entropy(logits, labels.long(), reduction='sum')


    # Negative binomial loss
    def negbin_loss(self, x, mean, var):
        t3 = (mean ** 2 / (var - mean))
        t1 = (x + t3 + 1e-8).lgamma() - (x + 1 + 1e-8).lgamma()
        t2 = x * np.log(((var - mean) / var))
        loss = t1 + t2
        return -(loss.sum())


    # Gaussian loss
    def l2_loss(self, x, x_mean, x_var):
        return (1 / (2 * x_var)) * ((x - x_mean).norm(2).pow(2))


    # Laplace loss
    def l1_loss(self, x, x_mean, x_var):
        return (1 / x_var) * ((x - x_mean).norm(1))


    # Inverse-gamma loss
    def inv_gamma_loss(self, x, alpha, beta):
        return (((((alpha + 1) * torch.log(x)) + (beta / (x))).sum()))


    # Scaled inverse-chi-squared loss
    def invschisq_loss(self, x, nu, tausq):
        z_loss = (nu * tausq / (2 * (x))) + ((1 + (nu / 2)) * ((x).log()))
        return ((z_loss.sum()))


    # Function that computes tau-squared based on a given mean
    def invschisq_tausq(self, nu, mean):
        return mean * (1 - (2 / nu))


    # linear annealing
    def linear_anneal(self, x, y2, y1, x2, x1, c):
        return ((y2 - y1) / (x2 - x1)) * x + c


    # set random seed
    def set_rng_seed(self, seed=None):
        if seed is None:
            seed = int(time.time())

        if self.use_gpu:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.seed = seed


    def plot_abun(self):
        self.logger.info('Saving plots of abundances!')

        for i in range(self.num_subjects):
            for j in range(self.num_otus):
                plt.plot(self.times[i].astype('int'),
                    self.X[i][j][self.times[i].astype('int')],
                    marker='.')
            plt.xlabel('Time')
            plt.ylabel('Abundance')
            plt.title('SubjectID: {} Outcome: {}'.format(i, self.y[i]))
            plt.savefig('{}_abun_subject_{}_outcome_{}.pdf'.format(
                self.args.data_name, i, self.y[i]), bbox_inches='tight')
            plt.close()


    def show_rules(self, best_model, test_loader, fold, save_viz=True):
        best_model.eval()

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                gf_out, spat_attn = best_model.spat_attn(data, k=self.args.max_k_otu)
                time_out, time_attn = best_model.time_attn(gf_out, mask, k=self.args.max_k_time)
                x = best_model.thresh_func(time_out, t=self.args.min_tau,
                    thresh_k=self.args.max_k_thresh,
                    bc_k=self.args.max_k_bc)
                x, z = best_model.rules(x, t=self.args.min_tau, k=self.args.max_k_bc)
                outputs, z_r = best_model.fc(x, t=self.args.min_tau, k=self.args.max_k_bc)

        rules = z_r.detach().cpu().numpy()
        detectors = z.detach().cpu().numpy()
        time_wts = time_attn.detach().cpu().numpy()
        otu_wts = spat_attn.squeeze(-1).detach().cpu().numpy()
        agg_abun = gf_out.detach().cpu().numpy()
        time_abun = time_out.detach().cpu().numpy()
        threshs = best_model.thresh_func.thresh.pow(2).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        fc_wts = best_model.fc.weight.softmax(dim=0).detach().cpu().numpy()
        kappas = best_model.spat_attn.kappa.pow(2).detach().cpu().numpy()

        dirName = './{}_run_{}_{}'.format(self.args.data_name, fold, self.args.seed)

        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")


        torch.save(best_model.state_dict(), '{}/best_model.pth'.format(dirName))

        if save_viz:
            for i, r in enumerate(rules):
                if r > 0.5:
                    ds = detectors[i]
                    for j, d in enumerate(ds):
                        if d > 0.5:
                            thresh = threshs[i, j]
                            kappa = kappas[i, j]
                            t = time_wts.mean(axis=0)[i, j]
                            t_min = -1
                            t_max = self.num_time
                            for p in range(len(t)):
                                if t[t_min + 1] <= (1 / self.num_time):
                                    t_min += 1
                                if t[t_max - 1] <= (1 / self.num_time):
                                    t_max -= 1
                            o = otu_wts[i, j]
                            sel_otus = [self.var_names[self.otu_idx[l]] for l, ot in enumerate(o) if ot > 1 / self.num_otus]
                            otu_annot_str = ''
                            if self.var_annot == {}:
                                for n in sel_otus:
                                    otu_annot_str = otu_annot_str + '\n' + n
                            else:
                                sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                for n in sel_otu_annot:
                                    otu_annot_str = otu_annot_str + '\n' + n

                            f = open('{}/r{}d{}_otuwts_{}.txt'.format(
                                dirName, i, j, self.args.data_name), "w+")
                            f.write(otu_annot_str)
                            f.close()

                            tree = deepcopy(self.phylo_tree)
                            sel_nodes = list()
                            for n in tree.get_leaves():
                                if n.name in sel_otus:
                                    sel_nodes.append(n)

                            nstyle = NodeStyle()
                            for n in tree.get_leaves():
                                if n.name in sel_otus:
                                    fca = n.get_common_ancestor(sel_nodes)
                                    break
                            if sel_nodes == []:
                                pass

                            for n in fca.traverse():
                                if n.name in sel_otus:
                                    o_idx = self.otu_idx.index(self.var_names.index(n.name))
                                    nstyle = NodeStyle()
                                    nstyle["shape"] = "sphere"
                                    nstyle["fgcolor"] = "green"
                                    nstyle["size"] = 6
                                    nstyle["vt_line_color"] = "green"
                                    nstyle["hz_line_color"] = "green"
                                    nstyle["vt_line_width"] = 2
                                    nstyle["hz_line_width"] = 2
                                    nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                    nstyle["hz_line_type"] = 0
                                    n.set_style(nstyle)
                                    tw = textwrap.TextWrapper(width=30)
                                    if self.var_annot == {}:
                                        node_name = n.name
                                    else:
                                        node_name = self.var_annot.get(n.name, '(no annotation)')
                                        node_name = node_name.split(" ")
                                        remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                        new_node_name = ''
                                        for l in node_name:
                                            if not l in remove_list:
                                                new_node_name = new_node_name + l + ' '
                                        node_name = new_node_name
                                    text = TextFace(tw.fill(text=node_name), fsize=15)
                                    n.add_face(text, 0)
                                else:
                                    nstyle = NodeStyle()
                                    nstyle["size"] = 0
                                    n.set_style(nstyle)

                            ts = TreeStyle()
                            ts.show_leaf_name = False
                            ts.show_branch_length = False
                            ts.branch_vertical_margin = 10
                            ts.min_leaf_separation = 10
                            ts.show_scale = False
                            fca.render('{}/r{}d{}_subtree_{}.pdf'.format(
                                dirName, i, j, self.args.data_name),
                                tree_style=ts)

                            plt.bar(np.arange(self.num_otus), o)
                            plt.savefig('{}/r{}d{}_otuwts_{}.pdf'.format(
                                dirName, i, j, self.args.data_name))
                            plt.close()

                            plt.bar(np.arange(self.num_time), t)
                            plt.savefig('{}/r{}d{}_timewts_{}.pdf'.format(
                                dirName, i, j, self.args.data_name))
                            plt.close()

                            fig = plt.figure()
                            ax = fig.add_subplot()
                            for k in range(self.num_subjects):
                                abun = agg_abun[k, i, j]
                                if labels[k]:
                                    lines_1, = ax.plot(self.times[k].astype('int'),
                                        abun[self.times[k].astype('int')],
                                        marker='+', color='g',
                                        linewidth=1.5, markersize=8)
                                else:
                                    lines_0, = ax.plot(self.times[k].astype('int'),
                                        abun[self.times[k].astype('int')],
                                        marker='.', color='#FF8C00',
                                        linewidth=1.5, markersize=8)
                            ax.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                            sel_days = np.arange(t_min, t_max + 1)
                            line_thresh, = ax.plot(sel_days,
                                thresh * np.ones(len(sel_days)),
                                c='k',
                                linestyle='solid',
                                linewidth=5)
                            ax.set_xlabel('Days', fontsize=20)
                            ax.set_ylabel('Relative Abundance', fontsize=20)
                            ax.legend([lines_0, lines_1, line_thresh],
                                [self.label_0, self.label_1, 'Threshold'],
                                fontsize=10, loc='upper left',
                                handler_map={tuple: HandlerTuple(ndivide=None)})
                            plt.savefig('{}/r{}d{}_abun_{}.pdf'.format(
                                dirName, i, j, self.args.data_name),
                                bbox_inches='tight')
                            plt.close()

                            fig = plt.figure()
                            ax = fig.add_subplot()
                            for k in range(self.num_subjects):
                                t_abun = time_abun[k, i, j]
                                if labels[k]:
                                    lines_1, = ax.plot(k, t_abun,
                                        marker='+', color='r')
                                else:
                                    lines_0, = ax.plot(k, t_abun,
                                        marker='.', color='g')
                            line_thresh = ax.axhline(y=thresh,
                                c='m', linestyle='--', linewidth=5, alpha=0.5)
                            ax.set_title('thresh: %.4f time: [%d, %d]' % (
                                thresh,
                                t_min,
                                t_max), fontsize=10)
                            ax.set_ylabel('Aggregated Relative Abundance', fontsize=10)
                            ax.set_xlabel('Subjects', fontsize=10)
                            ax.legend([lines_0, lines_1, line_thresh],
                                [self.label_0, self.label_1, 'Threshold'],
                                fontsize=10, loc='upper right',
                                handler_map={tuple: HandlerTuple(ndivide=None)})
                            plt.savefig('{}/r{}d{}_aggabun_{}.pdf'.format(
                                dirName, i, j, self.args.data_name))
                            plt.close()

        return


if __name__ == '__main__':
    # Parse command line args
    args = parse()

    # Init trainer object
    trainer = Trainer(args)

    # Load data
    trainer.load_data()

    # run cv loop
    trainer.train_loop()
