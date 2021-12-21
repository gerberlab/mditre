## Import required libraries
import argparse
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import time
from copy import deepcopy
import random
import textwrap
import datetime
import pickle
import sys

import numpy as np
import numpy.ma as ma
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from ete3 import Tree, TreeStyle, TextFace, NodeStyle, Face, ClusterTree, ProfileFace, add_face_to_node
from matplotlib.legend_handler import HandlerTuple
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from itertools import product
from matplotlib.collections import LineCollection
from matplotlib import markers
from matplotlib.path import Path
import scipy
from scipy.special import logit, expit
import pandas as pd
import dendropy
import matplotlib.colors as mc
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import LineCollection
from matplotlib import markers
from matplotlib.path import Path
import matplotlib.image as mpimage
import matplotlib.gridspec as gridspec

from math import floor
from itertools import chain
from PIL import Image

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli

from mditre.data import *
from mditre.models import MDITRE, binary_concrete, transf_log, inv_transf_log, MDITREAbun
from mditre.utils import AverageMeter, get_logger
from mditre.visualize import *

import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--data_name', type=str,
                        help='Name of the dataset, will be used for log dirname')
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2048, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr_kappa', default=0.001, type=float,
                        help='Initial learning rate for kappa.')
    parser.add_argument('--lr_eta', default=0.001, type=float,
                        help='Initial learning rate for eta.')
    parser.add_argument('--lr_time', default=0.01, type=float,
                        help='Initial learning rate for sigma.')
    parser.add_argument('--lr_mu', default=0.01, type=float,
                        help='Initial learning rate for mu.')
    parser.add_argument('--lr_thresh', default=0.0001, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--lr_slope', default=0.00001, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--lr_alpha', default=0.005, type=float,
                        help='Initial learning rate for binary concrete logits on detectors.')
    parser.add_argument('--lr_beta', default=0.005, type=float,
                        help='Initial learning rate for binary concrete logits on rules.')
    parser.add_argument('--lr_fc', default=0.001, type=float,
                        help='Initial learning rate for linear classifier weights and bias.')
    parser.add_argument('--lr_bias', default=0.001, type=float,
                        help='Initial learning rate for linear classifier weights and bias.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Set random seed for reproducibility')
    parser.add_argument('--min_k_otu', default=100, type=float,
                        help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--max_k_otu', default=1000, type=float,
                        help='Min Temperature on heavyside logistic for otu selection')
    parser.add_argument('--min_k_time', default=1, type=float,
                        help='Max Temperature on heavyside logistic for time window')
    parser.add_argument('--max_k_time', default=10, type=float,
                        help='Min Temperature on heavyside logistic for time window')
    parser.add_argument('--min_k_thresh', default=100, type=float,
                        help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--max_k_thresh', default=1000, type=float,
                        help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--min_k_slope', default=1e3, type=float,
                        help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--max_k_slope', default=1e4, type=float,
                        help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--min_k_bc', default=1, type=float,
                        help='Min Temperature for binary concretes')
    parser.add_argument('--max_k_bc', default=10, type=float,
                        help='Max Temperature for binary concretes')
    parser.add_argument('--n_w', type=int, default=1,
                        help='Number of detectors (time)')
    parser.add_argument('--n_d', type=int, default=10,
                        help='Number of detectors (otus)')
    parser.add_argument('--cv_type', type=str, default='kfold',
                        choices=['loo', 'kfold', 'None'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--z_mean', type=float, default=1,
                        help='NBD Mean active detectors per rule')
    parser.add_argument('--z_var', type=float, default=5,
                        help='NBD variance of active detectors per rule')
    parser.add_argument('--z_r_mean', type=float, default=1,
                        help='NBD Mean active rules')
    parser.add_argument('--z_r_var', type=float, default=5,
                        help='NBD variance of active rules')
    parser.add_argument('--w_var', type=float, default=1e5,
                        help='Normal prior variance on weights.')
    parser.add_argument('--dist_prior', default=10, type=float,
                        help='phylo dist prior')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed multiprocess training')
    parser.add_argument('--debug', action='store_true',
                        help='Debugging')
    parser.add_argument('--save_as_csv', action='store_true',
                        help='Debugging')
    parser.add_argument('--inner_cv', action='store_true',
                        help='Do inner cross val')
    parser.add_argument('--is_16s', action='store_true',
                        help='16s dataset')
    parser.add_argument('--use_ref_kappa', action='store_true',
                        help='Use prior on kappa from ref tree')
    parser.add_argument('--evaluate', action='store_true',
                        help='Perform model evaluation')
    parser.add_argument('--model_ckpt_path', default='', type=str,
                        help='Path to pretrained model')
    parser.add_argument('--model_init_ckpt_path', default='', type=str,
                        help='Path to pretrained init model')
    parser.add_argument('--use_abun', action='store_true',
                        help='Use only abundance type detectors')
    parser.add_argument('--use_topo', action='store_true',
                        help='Use topological distance')

    args = parser.parse_args()
    return args


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        # Check for gpu availability
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            # Check for cudnn
            assert torch.backends.cudnn.enabled, \
                   "Need cudnn backend to be enabled!"
            cudnn.benchmark = True

        # Check for gpu, assign device
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl' if self.use_gpu else "gloo",
                                                 init_method='env://',
                                                 timeout=datetime.timedelta(days=7))
            self.args.world_size = torch.distributed.get_world_size()
            self.args.rank = torch.distributed.get_rank()

        if self.args.world_size > 1:
            self.args.distributed = True
        else:
            self.args.distributed = False

        # Set random seed
        self.set_rng_seed(self.args.seed)

        # Create logger
        log_path = self.create_log_path()
        self.logger = get_logger(os.path.join(log_path))

        # print cmdline args
        if not self.args.rank:
            self.logger.info(self.args)

        if not self.args.rank:
            self.logger.info('Using device: %s' % (self.device))
            self.logger.info('Trainer initialized!')

    def create_dir(self, filename):
        if not os.path.exists(filename):
            os.mkdir(filename)
            print("Directory " , filename ,  " Created ")
        else:    
            print("Directory " , filename ,  " already exists")
        return


    def create_log_path(self):
        # create top-level directory
        parent_log_dir = './logs/{}'.format(self.args.data_name)
        if not self.args.rank:
            self.create_dir(parent_log_dir)
        # create a dir for each seed
        seed_log_dir = '{}/seed_{}'.format(parent_log_dir, self.args.seed)
        if not self.args.rank:
            self.create_dir(seed_log_dir)
        # make sure all dirs are created
        if self.args.distributed:
            torch.distributed.barrier()
        # create a dir for each process
        self.log_dir = '{}/rank_{}'.format(seed_log_dir, self.args.rank)
        self.create_dir(self.log_dir)
        # path for logfile
        log_path = '{}/output.log'.format(self.log_dir)

        return log_path


    def load_data(self):
        # Load data from a pickle file
        dataset = load_from_pickle(self.args.data)

        # Number of variables (total nodes in phylo tree)
        self.num_vars = int(dataset['n_variables'])
        # Total experiment duration
        self.num_time = int(dataset['experiment_end'] - dataset['experiment_start'] + 1)
        # Abundance samples of each subject
        # list of numpy arrays, each array is abundance of variables over time
        samples = dataset['X']
        # Subject outcomes
        # Add 0. to convert to float
        self.y = dataset['y'] + 0.
        # Phylogenetic tree of variables
        self.phylo_tree = dataset['variable_tree']
        # List of numpy arrays timestamps of samples for each subject
        # Experiment start time could be negative, so convert to non-negative
        self.times = [t - dataset['experiment_start'] for t in dataset['T']]
        # List of variable names in the phylogenetic tree
        # For example [otu0001, otu0002, 12343,...]
        self.var_names = dataset['variable_names']
        # Dict of variable annotations in the phylogenetic tree
        # For example {'otu0001': 'OTU mapped to Bilophilia Wadsworthia',...}
        self.var_annot = dataset['variable_annotations']
        # Preprocess data
        self.preprocess_data(self.phylo_tree, self.var_names, self.num_time, samples, self.times)
        
        self.var_ids = self.phylo_tree.get_leaf_names()
        if self.var_annot != {}:
            self.var_names = [self.var_annot[n] for n in self.var_ids]

        if 'tree_distance' in dataset.keys() and not self.args.use_topo:
            self.dist_matrix = dataset['tree_distance']

        # Total number of subjects in the dataset
        self.num_subjects = self.X.shape[0]

        if self.args.batch_size > self.num_subjects:
            self.args.batch_size = self.num_subjects

        if 'david' in self.args.data.lower():
            self.label_0 = 'Animal diet'
            self.label_1 = 'Plant diet'
        elif 'knat' in self.args.data.lower():
            self.label_0 = 'Finnish/Estonian'
            self.label_1 = 'Russian'
        elif 'bokulich_diet' in self.args.data.lower():
            self.label_0 = 'Breast milk diet'
            self.label_1 = 'Formula diet'
        elif 'bokulich_delivery' in self.args.data.lower():
            self.label_0 = 'Normal delivery'
            self.label_1 = 'C-section'
        elif 'digiulio' in self.args.data.lower():
            self.label_0 = 'Normal delivery'
            self.label_1 = 'Premature delivery'
        elif 't1d' in self.args.data.lower():
            self.label_0 = 'No T1D'
            self.label_1 = 'T1D'
            self.args.lr_thresh = 1e-3
            self.args.lr_slope = 1e-6
            self.args.n_d = self.num_otus
        elif 'shao' in self.args.data.lower():
            self.label_0 = 'Normal delivery'
            self.label_1 = 'C-section'
        elif 'brooks' in self.args.data.lower():
            self.label_0 = 'Normal delivery'
            self.label_1 = 'C-section'
        else:
            self.label_0 = 'Outcome 0'
            self.label_1 = 'Outcome 1'

        if not self.args.rank:
            # print dataset info
            self.logger.info('Dataset: %s Variables: %d, Otus: %d,\
                Subjects: %d, Total samples: %d' % 
                (self.args.data_name, dataset['n_variables'],
                self.num_otus, self.num_subjects,
                sum(map(len, dataset['T']))))
            self.logger.info('Outcomes: {}'.format(np.unique(self.y, return_counts=True)))
            self.logger.info('Exp start: {} Exp end: {}'.format(dataset['experiment_start'],
                dataset['experiment_end']))

        if not self.args.rank:
            self.logger.info('Loaded and preprocessed dataset!')


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
        self.X, self.X_mask, self.y, self.times = get_data_matrix(num_otus, num_time,
            samples, times, otu_idx, self.y, use_abun=self.args.use_abun)

        # save data as class variables for later use
        self.dist_matrix = dist_matrix
        self.num_otus = num_otus
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

        if not self.args.rank:
            self.logger.info('Using cross-validation type: {}'.format(self.args.cv_type))


    def set_model_hparams(self):
        # Number of rules
        self.num_rules = 10
        # Number of otu centers per rule
        self.num_detectors = self.args.n_d
        dist = self.dist_matrix
        d = np.arange(1, self.num_otus + 1)
        for dd in d:
            self.emb_dim = int(dd)
            self.dist_emb = self.compute_dist_emb_mds().astype(np.float32)
            self.dist_matrix_embed = self.compute_dist()
            ks = scipy.stats.kstest(dist.reshape(-1),
                self.dist_matrix_embed.reshape(-1))
            if ks.pvalue <= 0.05:
                if dd == self.num_otus:
                    break
            else:
                self.emb_dim = int(dd)
                break
        if not self.args.rank:
            self.logger.info('Using embedding D = {}'.format(self.emb_dim))

        if self.use_gpu:
            mean = torch.tensor([0], dtype=torch.float32).cuda()
            std = torch.tensor(np.sqrt([1e5]), dtype=torch.float32).cuda()
        else:
            mean = 0
            std = np.sqrt(1e5)
        self.normal_rule = Normal(mean, std)
        if self.use_gpu:
            mean = torch.tensor([0], dtype=torch.float32).cuda()
            std = torch.tensor(np.sqrt([1e5]), dtype=torch.float32).cuda()
        else:
            mean = 0
            std = np.sqrt(1e5)
        self.normal_det = Normal(mean, std)

        # Set mean and variance for gaussian prior on regression weights
        self.wts_mean = 0
        self.wts_var = self.args.w_var
        mean = torch.ones(self.num_rules, dtype=torch.float32) * self.wts_mean
        cov = torch.eye(self.num_rules, dtype=torch.float32) * self.wts_var
        if self.use_gpu:
            mean = mean.cuda()
            cov = cov.cuda()
        self.normal_wts = MultivariateNormal(mean, cov)

        # Set mean and variance for neg bin prior for detectors
        self.z_mean = self.args.z_mean
        self.z_var = self.args.z_var
        if self.use_gpu:
            mean = torch.tensor([self.z_mean], dtype=torch.float32).cuda()
            var = torch.tensor([self.z_var], dtype=torch.float32).cuda()
        else:
            mean = self.z_mean
            var = self.z_var
        self.negbin_det = self.create_negbin(mean, var)

        # Set mean and variance for neg bin prior for rules
        self.z_r_mean = self.args.z_r_mean
        self.z_r_var = self.args.z_r_var
        if self.use_gpu:
            mean = torch.tensor([self.z_r_mean], dtype=torch.float32).cuda()
            var = torch.tensor([self.z_r_var], dtype=torch.float32).cuda()
        else:
            mean = self.z_r_mean
            var = self.z_r_var
        self.negbin_rules = self.create_negbin(mean, var)

        ### Prior hyperparams ###
        if self.args.is_16s:
            ref_median_dist = {
                'median_genus': -1.2057759,
                'median_family': -0.98787415,
                'median_phylum': -0.06123711,
                'median_full': 0.151053,
            }
            ref_median_var = {
                'median_genus': 0.7300124,
                'median_family': 0.7533012,
                'median_phylum': 0.3374289,
                'median_full': 0.46544546,
            }
        else:
            ref_median_dist = {
                'median_genus': -1.2410396213316708,
                'median_family': -1.1044373025931737,
                'median_phylum': -0.7362116801654603,
                'median_full': -0.40254638,
            }
            ref_median_var = {
                'median_genus': 0.41830161965506807,
                'median_family': 0.405738891096145,
                'median_phylum': 0.24929848049861536,
                'median_full': 0.30349404,
            }

        # if self.args.use_ref_kappa:
        # Use median of the above distribution as the prior mean
        self.kappa_prior_mean = ref_median_dist['median_family']
        self.kappa_prior_var = ref_median_var['median_family']
        # else:
        #     self.kappa_prior_mean = 0
        #     self.kappa_prior_var = 1e5
        if self.use_gpu:
            mean = torch.tensor([self.kappa_prior_mean], dtype=torch.float32).cuda()
            std = torch.tensor(np.sqrt([self.kappa_prior_var]), dtype=torch.float32).cuda()
        else:
            mean = self.kappa_prior_mean
            std = np.sqrt(self.kappa_prior_var)
        self.normal_kappa = Normal(mean, std)

        # Set mean and variance for gaussian prior on embeddings
        self.emb_mean = 0
        self.emb_var = 1e5
        mean = torch.ones(self.emb_dim, dtype=torch.float32) * self.emb_mean
        cov = torch.eye(self.emb_dim, dtype=torch.float32) * self.emb_var
        if self.use_gpu:
            mean = mean.cuda()
            cov = cov.cuda()
        self.normal_emb = MultivariateNormal(mean, cov)


        # Time aggregation bandwith (sigma)
        # nth% of the total experiment duration as the mean
        self.time_prior = 0.3
        self.sigma_prior_mean = (self.time_prior * self.num_time)
        self.sigma_prior_std = np.sqrt(1e5)
        if self.use_gpu:
            mean = torch.tensor([logit(self.time_prior)], dtype=torch.float32).cuda()
            std = torch.tensor([self.sigma_prior_std], dtype=torch.float32).cuda()
        else:
            mean = logit(self.time_prior)
            std = self.sigma_prior_std
        self.normal_time_abun_a = Normal(mean, std)
        self.normal_time_slope_a = Normal(mean, std)

        self.mu_prior_mean = logit(0.5)
        self.mu_prior_std = np.sqrt(1e5)
        if self.use_gpu:
            mean = torch.tensor([self.mu_prior_mean], dtype=torch.float32).cuda()
            std = torch.tensor([self.mu_prior_std], dtype=torch.float32).cuda()
        else:
            mean = self.mu_prior_mean
            std = self.mu_prior_std
        self.normal_time_abun_b = Normal(mean, std)
        self.normal_time_slope_b = Normal(mean, std)

        self.n_w = self.args.n_w

        if not self.args.rank:
            self.logger.info('Initialized priors!')
            self.logger.info('Rules: {} Detectors: {}'.format(self.num_rules, self.num_detectors))


    def compute_dist_emb_mds(self):
        mds = MDS(n_components=self.emb_dim, random_state=self.args.seed,
            n_jobs=-1, dissimilarity='precomputed')
        mds_transf = mds.fit_transform(self.dist_matrix)

        return mds_transf


    def compute_dist(self):
        dist = np.zeros((self.num_otus, self.num_otus), dtype=np.float32)
        for i in range(self.num_otus):
            for j in range(self.num_otus):
                dist[i, j] = np.linalg.norm(self.dist_emb[i] - self.dist_emb[j], axis=-1)

        return dist


    def create_negbin(self, mean, var):
        assert var != mean, 'NegBin Variance should not be = Mean!'
        p = float(var - mean) / float(var)
        r = float(mean ** 2) / float(var - mean)

        if self.use_gpu:
            p = torch.tensor([p], dtype=torch.float32).cuda()
            r = torch.tensor([r], dtype=torch.float32).cuda()

        return NegativeBinomial(r, probs=p)


    def train_loop(self):
        # Init model hyperparams
        self.set_model_hparams()

        # # Init otu centers
        detector_otuids = list()
        eta_init = np.zeros((self.num_rules, self.num_detectors, self.emb_dim), dtype=np.float32)
        kappa_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
        for i in range(self.num_rules):
            assigned_otus_det = list()
            kmeans = KMeans(n_clusters=self.num_detectors, random_state=self.args.seed + i).fit(self.dist_emb)
            for j in range(self.num_detectors):
                assigned_otus = list()
                eta_init[i, j] = kmeans.cluster_centers_[j]
                med_dist = list()
                for k in range(self.num_otus):
                    if kmeans.labels_[k] == j:
                        med_dist.append(np.linalg.norm(kmeans.cluster_centers_[j] - self.dist_emb[k], axis=-1))
                        cur_assig_otu = k
                if len(med_dist) > 1:
                    kappa_init[i, j] = np.mean(med_dist)
                else:
                    d = self.dist_matrix_embed[cur_assig_otu]
                    kappa_init[i, j] = min(d[np.nonzero(d)])
                for k in range(self.num_otus):
                    if kmeans.labels_[k] == j:
                        dist = np.linalg.norm(kmeans.cluster_centers_[j] - self.dist_emb[k], axis=-1)
                        if dist <= kappa_init[i, j]:
                            assigned_otus.append(k)
                assigned_otu_names = [self.var_names[k] for k in assigned_otus]
                assigned_otus_det.append(assigned_otus)
            detector_otuids.append(assigned_otus_det)


        # Compute minimum time-window lengths for each possible
        # time center
        self.time_unique = np.unique(np.concatenate(np.array(self.times))).astype('int')
        window_len = None
        acc_center_abun = list()
        acc_len_abun = list()
        acc_num_samples_abun = list()
        times = np.arange(self.num_time)
        for t in times:
            win_len = 1
            invalid_center = False
            while True:
                window_start = np.floor(t - (win_len / 2.)).astype('int')
                window_end = np.ceil(t + (win_len / 2.)).astype('int')
                if window_start >= 0 and window_end <= self.num_time:
                    win_mask = self.X_mask[:, window_start:window_end].sum(-1)
                    if np.all(win_mask >= 1):
                        window_len = win_len
                        break
                    else:
                        win_len += 1
                else:
                    invalid_center = True
                    break
            if not invalid_center:
                acc_center_abun.append(t)
                acc_len_abun.append(window_len)
                acc_num_samples_abun.append(self.X_mask[:, window_start:window_end].sum())

        p_acc_num_samples_abun = np.array(acc_num_samples_abun) / sum(acc_num_samples_abun)
        N_cur = min(3, len(acc_center_abun))

        if not self.args.use_abun:
            window_len_slope = None
            acc_center_slope = list()
            acc_len_slope = list()
            acc_num_samples_slope = list()
            for t in times:
                win_len = 1
                invalid_center = False
                while True:
                    window_start = np.floor(t - win_len / 2.).astype('int')
                    window_end = np.ceil(t + win_len / 2.).astype('int')
                    if window_start >= 0 and window_end <= self.num_time:
                        win_mask = self.X_mask[:, window_start:window_end].sum(-1)
                        if np.all(win_mask >= 2):
                            window_len_slope = win_len
                            break
                        else:
                            win_len += 1
                    else:
                        invalid_center = True
                        break
                if not invalid_center:
                    acc_center_slope.append(t)
                    acc_len_slope.append(window_len_slope)
                    acc_num_samples_slope.append(self.X_mask[:, window_start:window_end].sum())

            p_acc_num_samples_slope = np.array(acc_num_samples_slope) / sum(acc_num_samples_slope)
            N_cur_slope = min(3, len(acc_center_slope))

        mu_idx = np.random.choice(np.arange(len(acc_center_abun)), (self.num_rules, self.num_detectors),
            p=p_acc_num_samples_abun)
        mu_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
        sigma_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
        if not self.args.use_abun:
            mu_slope_idx = np.random.choice(np.arange(len(acc_center_slope)), (self.num_rules, self.num_detectors),
                p=p_acc_num_samples_slope)
            mu_slope_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
            sigma_slope_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
        for l in range(self.num_rules):
            for m in range(self.num_detectors):
                mu_init[l, m] = acc_center_abun[mu_idx[l, m]]
                sigma_init[l, m] = acc_len_abun[mu_idx[l, m]]
                if not self.args.use_abun:
                    mu_slope_init[l, m] = acc_center_slope[mu_slope_idx[l, m]]
                    sigma_slope_init[l, m] = acc_len_slope[mu_slope_idx[l, m]]



        # Init logsitic regression weights
        alpha_init = np.random.normal(0, 1, (self.num_rules, self.num_detectors)) * 1e-3
        beta_init = np.random.normal(0, 1, (self.num_rules)) * 1e-3
        w_init = np.random.normal(0, 1, (1, self.num_rules))
        bias_init = np.zeros((1))

        # Init cross-val data splits
        self.get_cv_splits()

        full_loader = get_data_loaders(self.X, self.y,
            self.X_mask,
            len(self.X),
            self.args.workers,
            shuffle=False, pin_memory=self.use_gpu)

        if self.args.evaluate:
            model_ckpt = torch.load(self.args.model_ckpt_path)
            if self.args.model_init_ckpt_path:
                model_init_ckpt = torch.load(self.args.model_init_ckpt_path)
                if not self.args.use_abun:
                    model_init = MDITRE(self.num_rules, self.num_otus, self.num_detectors,
                        self.num_time, self.args.n_w, self.dist_emb, self.emb_dim)
                else:
                    model_init = MDITREAbun(self.num_rules, self.num_otus, self.num_detectors,
                        self.num_time, self.args.n_w, self.dist_emb, self.emb_dim)
                model_init.load_state_dict(model_init_ckpt)
                model_init.to(self.device)
            else:
                model_init = None
            if not self.args.use_abun:
                model = MDITRE(self.num_rules, self.num_otus, self.num_detectors,
                    self.num_time, self.args.n_w, self.dist_emb, self.emb_dim)
            else:
                model = MDITREAbun(self.num_rules, self.num_otus, self.num_detectors,
                    self.num_time, self.args.n_w, self.dist_emb, self.emb_dim)
            model.load_state_dict(model_ckpt)
            model.to(self.device)
            val_loss, val_f1 = self.eval(model, full_loader)
            self.sub_log_odds = model.fc.sub_log_odds.detach().cpu().numpy()
            self.subjects_log_odds = model.fc.log_odds.detach().cpu().numpy()
            self.sub_0_log_odds = list()
            self.sub_1_log_odds = list()
            self.subjects_0_log_odds = list()
            self.subjects_1_log_odds = list()
            for i in range(self.num_subjects):
                if self.y[i]:
                    self.sub_1_log_odds.append(self.sub_log_odds[i, :])
                    self.subjects_1_log_odds.append(self.subjects_log_odds[i])
                else:
                    self.sub_0_log_odds.append(self.sub_log_odds[i, :])
                    self.subjects_0_log_odds.append(self.subjects_log_odds[i])
            
            self.logger.info('Evaluated F1 score: {:.2f}'.format(val_f1))
            if not self.args.use_abun:
                self.show_rules(model, full_loader, 0, save_viz=True, best_model_init=model_init)
            else:
                self.show_rules_abun(model, full_loader, 0, save_viz=True, best_model_init=model_init)
        else:
            if self.args.world_size > 1:
                if self.args.world_size > len(self.train_splits):
                    raise ValueError('Num. processes should not exceed num kfolds!')
                # Split kfolds equally between the procs
                proc_kfolds = np.array_split(np.arange((len(self.train_splits))), self.args.world_size)
            else:
                proc_kfolds = [np.arange((len(self.train_splits)))]

            k_fold_test_preds = np.zeros((len(self.X)), dtype=np.float32)
            k_fold_test_true = np.zeros((len(self.X)), dtype=np.float32)
            k_fold_test_prob = np.zeros((len(self.X)), dtype=np.float32)

            k_fold_test_f1 = np.zeros(len(self.train_splits))

            start = time.time()

            # Cross-val training loop
            for i, (train_ids, test_ids) in enumerate(zip(self.train_splits, self.test_splits)):
                if i in proc_kfolds[self.args.rank]:
                    if self.args.inner_cv:
                        # val data for inner cv
                        train_ids, val_ids = train_test_split(train_ids,
                            test_size=0.1, stratify=self.y[train_ids],
                            random_state=self.args.seed)
                    else:
                        val_ids = train_ids

                    # Init time windows
                    thresh_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                    if not self.args.use_abun:
                        slope_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                    thresh_mean = list()
                    slope_mean = list()

                    if not self.args.rank:
                        self.logger.info('Initializing model!')

                    for l in range(self.num_rules):
                        for m in range(self.num_detectors):
                            # mu_abun = mu_init[l, m]
                            # sigma_abun = sigma_init[l, m]
                            # window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                            # window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                            # x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start_abun:window_end_abun]
                            # x_mask = self.X_mask[train_ids, :][:, window_start_abun:window_end_abun]
                            # X = x.sum(1).sum(-1) / x_mask.sum(-1)
                            # thresh_init[l, m] = X.mean()


                            x_t = np.zeros((len(train_ids), len(acc_center_abun)))
                            for n in range(len(acc_center_abun)):
                                mu_abun = acc_center_abun[n]
                                sigma_abun = acc_len_abun[n]
                                window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                                window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                                x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start_abun:window_end_abun]
                                x_mask = self.X_mask[train_ids, :][:, window_start_abun:window_end_abun]
                                X = x.sum(1).sum(-1) / x_mask.sum(-1)
                                x_t[:, n] = X - X.mean()

                            x_marg_0 = x_t[self.y[train_ids] == 0, :].mean(axis=0)
                            x_marg_1 = x_t[self.y[train_ids] == 1, :].mean(axis=0)
                            x_marg = np.absolute(x_marg_0 - x_marg_1)
                            best_t_id_abun = np.argsort(x_marg)[::-1][0]
                            mu_abun = acc_center_abun[best_t_id_abun]
                            sigma_abun = acc_len_abun[best_t_id_abun]
                            window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                            window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                            x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start_abun:window_end_abun]
                            x_mask = self.X_mask[train_ids, :][:, window_start_abun:window_end_abun]
                            X = x.sum(1).sum(-1) / x_mask.sum(-1)
                            thresh_init[l, m] = X.mean()
                            mu_init[l, m] = mu_abun
                            sigma_init[l, m] = sigma_abun
                            thresh_mean.append(x_marg[best_t_id_abun])


                            if not self.args.use_abun:
                                # mu_slope = mu_slope_init[l, m]
                                # sigma_slope = sigma_slope_init[l, m]
                                # window_start_slope = np.floor(mu_slope - (sigma_slope / 2.)).astype('int')
                                # window_end_slope = np.ceil(mu_slope + (sigma_slope / 2.)).astype('int')
                                # x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start_slope:window_end_slope]
                                # x_mask = self.X_mask[train_ids, :][:, window_start_slope:window_end_slope]
                                # tau = np.arange(window_start_slope, window_end_slope) - mu_slope
                                # X_slope = np.array([np.polyfit(tau, x[s].sum(0), 1, w=x_mask[s])[0] for s in range(len(train_ids))])
                                # slope_init[l, m] = X_slope.mean()

                                x_t = np.zeros((len(train_ids), len(acc_center_slope)))
                                for n in range(len(acc_center_slope)):
                                    mu_abun = acc_center_slope[n]
                                    sigma_abun = acc_len_slope[n]
                                    window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                                    window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                                    x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start_abun:window_end_abun]
                                    x_mask = self.X_mask[train_ids, :][:, window_start_abun:window_end_abun]
                                    tau = np.arange(window_start_abun, window_end_abun) - mu_abun
                                    X = np.array([np.polyfit(tau, x[s].sum(0), 1, w=x_mask[s])[0] for s in range(len(train_ids))])
                                    x_t[:, n] = X - X.mean()

                                x_marg_0 = x_t[self.y[train_ids] == 0, :].mean(axis=0)
                                x_marg_1 = x_t[self.y[train_ids] == 1, :].mean(axis=0)
                                x_marg = np.absolute(x_marg_0 - x_marg_1)
                                best_t_id_abun = np.argsort(x_marg)[::-1][0]
                                mu_abun = acc_center_slope[best_t_id_abun]
                                sigma_abun = acc_len_slope[best_t_id_abun]
                                window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                                window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                                x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start_abun:window_end_abun]
                                x_mask = self.X_mask[train_ids, :][:, window_start_abun:window_end_abun]
                                tau = np.arange(window_start_abun, window_end_abun) - mu_abun
                                X = np.array([np.polyfit(tau, x[s].sum(0), 1, w=x_mask[s])[0] for s in range(len(train_ids))])
                                slope_init[l, m] = X.mean()
                                mu_slope_init[l, m] = mu_abun
                                sigma_slope_init[l, m] = sigma_abun
                                slope_mean.append(x_marg[best_t_id_abun])

                    abun_a_init = sigma_init / self.num_time
                    abun_a_init = np.clip(abun_a_init, 1e-2, 1 - 1e-2)
                    abun_b_init = (mu_init - (self.num_time * abun_a_init / 2.)) / ((1 - abun_a_init) * self.num_time)
                    abun_b_init = np.clip(abun_b_init, 1e-2, 1 - 1e-2)

                    if not self.args.use_abun:
                        slope_a_init = sigma_slope_init / self.num_time
                        slope_a_init = np.clip(slope_a_init, 1e-2, 1 - 1e-2)
                        slope_b_init = (mu_slope_init - (self.num_time * slope_a_init / 2.)) / ((1 - slope_a_init) * self.num_time)
                        slope_b_init = np.clip(slope_b_init, 1e-2, 1 - 1e-2)

                    # Initiazation dict
                    if not self.args.use_abun:
                        init_args = {
                            'kappa_init': kappa_init,
                            'eta_init': eta_init,
                            'abun_a_init': abun_a_init,
                            'abun_b_init': abun_b_init,
                            'slope_a_init': slope_a_init,
                            'slope_b_init': slope_b_init,
                            'thresh_init': thresh_init,
                            'w_init': w_init,
                            'bias_init': bias_init,
                            'alpha_init': alpha_init,
                            'beta_init': beta_init,
                            'slope_init': slope_init,
                        }
                    else:
                        init_args = {
                            'kappa_init': kappa_init,
                            'eta_init': eta_init,
                            'abun_a_init': abun_a_init,
                            'abun_b_init': abun_b_init,
                            'thresh_init': thresh_init,
                            'w_init': w_init,
                            'bias_init': bias_init,
                            'alpha_init': alpha_init,
                            'beta_init': beta_init,
                        }


                    # train-val-test data loaders
                    train_loader = get_data_loaders(self.X[train_ids],
                        self.y[train_ids],
                        self.X_mask[train_ids],
                        self.args.batch_size, self.args.workers,
                        shuffle=True, pin_memory=self.use_gpu)
                    val_loader = get_data_loaders(self.X[val_ids],
                        self.y[val_ids],
                        self.X_mask[val_ids],
                        self.args.batch_size, self.args.workers,
                        shuffle=False, pin_memory=self.use_gpu)
                    test_loader = get_data_loaders(self.X[test_ids],
                        self.y[test_ids],
                        self.X_mask[test_ids],
                        self.args.batch_size, self.args.workers,
                        shuffle=False, pin_memory=self.use_gpu)

                    # Init model
                    if not self.args.use_abun:
                        model = MDITRE(self.num_rules, self.num_otus, self.num_detectors,
                            self.num_time, self.args.n_w, self.dist_emb, self.emb_dim)
                    else:
                        model = MDITREAbun(self.num_rules, self.num_otus, self.num_detectors,
                            self.num_time, self.args.n_w, self.dist_emb, self.emb_dim)
                    model.init_params(init_args)
                    model.to(self.device)
                    model_init = deepcopy(model)

                    # thresh_mean = 10 ** (-floor(-np.log(np.mean(thresh_mean))))
                    # if not self.args.use_abun:
                    #     slope_mean = 10 ** (-floor(-np.log(np.mean(np.absolute(slope_mean)))))
                    # # thresh_mean = (np.mean(thresh_init.reshape(-1)))
                    # # if not self.args.use_abun:
                    # #     slope_mean = np.mean(np.absolute(slope_init.reshape(-1)))

                    # # self.args.lr_thresh = max(0.1 * thresh_mean, 1e-3)
                    # # self.args.lr_thresh = (0.1 * thresh_mean)
                    # self.args.lr_thresh = 0.1 * (thresh_mean)
                    # # self.args.lr_thresh = 1e-3
                    # if not self.args.use_abun:
                    #     # self.args.lr_slope = max(0.1 * slope_mean, 1e-4)
                    #     self.args.lr_slope = (0.1 * slope_mean)
                    #     # self.args.lr_slope = 1e-4
                    #     # self.args.lr_slope = 1e-6
                    # # self.args.min_k_thresh = (1 / thresh_mean)
                    # # self.args.max_k_thresh = 10 * self.args.min_k_thresh
                    # self.args.min_k_thresh = 0.1 * (1 / (thresh_mean))
                    # self.args.max_k_thresh = 10 * self.args.min_k_thresh
                    # # self.args.min_k_thresh = 1e3
                    # # self.args.max_k_thresh = 1e4
                    # print(self.args.lr_thresh, self.args.min_k_thresh, self.args.max_k_thresh)
                    # if not self.args.use_abun:
                    #     self.args.min_k_slope = 0.1 * (1 / slope_mean)
                    #     self.args.max_k_slope = 10 * self.args.min_k_slope
                    #     # self.args.min_k_slope = 1e4
                    #     # self.args.max_k_slope = 1e5
                    #     # self.args.min_k_slope = 1e6
                    #     # self.args.max_k_slope = 1e7
                    #     print(self.args.lr_slope, self.args.min_k_slope, self.args.max_k_slope)

                    if not self.args.rank:
                        for k, v in model.named_parameters():
                            self.logger.info(k)
                            self.logger.info(v)

                    # Inner loop training, store best model
                    best_model = self.train_model(model, train_loader,
                        val_loader, test_loader, i)

                    # Eval on test set to get f1
                    probs, true = self.eval_with_preds(best_model, test_loader)
                    k_fold_test_prob[test_ids] = probs
                    k_fold_test_preds[test_ids] = (np.array(probs) > 0.5).astype(np.float32)
                    k_fold_test_true[test_ids] = true

                    k_fold_test_f1[i] = f1_score(true, (np.array(probs) > 0.5).astype(np.float32))

                    self.eval_with_preds(best_model, full_loader)
                    self.sub_log_odds = best_model.fc.sub_log_odds.detach().cpu().numpy()
                    self.subjects_log_odds = best_model.fc.log_odds.detach().cpu().numpy()
                    self.sub_0_log_odds = list()
                    self.sub_1_log_odds = list()
                    self.subjects_0_log_odds = list()
                    self.subjects_1_log_odds = list()
                    for jj in range(self.num_subjects):
                        if self.y[jj]:
                            self.sub_1_log_odds.append(self.sub_log_odds[jj, :])
                            self.subjects_1_log_odds.append(self.subjects_log_odds[jj])
                        else:
                            self.sub_0_log_odds.append(self.sub_log_odds[jj, :])
                            self.subjects_0_log_odds.append(self.subjects_log_odds[jj])

                    if not self.args.rank:
                        for k, v in best_model.named_parameters():
                            self.logger.info(k)
                            self.logger.info(v)

                    if not self.args.use_abun:
                        self.show_rules(best_model, full_loader, i, save_viz=True, best_model_init=None)
                    else:
                        self.show_rules_abun(best_model, full_loader, i, save_viz=True, best_model_init=None)


            if self.args.world_size > 1:
                k_fold_test_true_tens = torch.from_numpy(k_fold_test_true).to(self.device)
                k_fold_test_preds_tens = torch.from_numpy(k_fold_test_preds).to(self.device)
                k_fold_test_f1_tens = torch.from_numpy(k_fold_test_f1).to(self.device)

                torch.distributed.all_reduce(k_fold_test_true_tens, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(k_fold_test_preds_tens, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(k_fold_test_f1_tens, op=torch.distributed.ReduceOp.SUM)

                k_fold_test_true = k_fold_test_true_tens.detach().cpu().numpy()
                k_fold_test_preds = k_fold_test_preds_tens.detach().cpu().numpy()
                k_fold_test_f1 = k_fold_test_f1_tens.detach().cpu().numpy()


            if not self.args.rank:
                # Compute final cv f1 score
                cv_f1 = f1_score(k_fold_test_true, k_fold_test_preds)
                cv_auc = roc_auc_score(k_fold_test_true, k_fold_test_preds)
                clf_report = classification_report(k_fold_test_true, k_fold_test_preds)
                tn, fp, fn, tp = confusion_matrix(k_fold_test_true, k_fold_test_preds).ravel()

                end = time.time()

                self.logger.info('F1 score: %.2f' % (cv_f1))
                # self.logger.info('AUC score: %.2f' % (cv_auc))
                # self.logger.info('Preds: {}'.format(k_fold_test_preds))
                # self.logger.info('labels: {}'.format(k_fold_test_true))
                # self.logger.info(clf_report)
                # self.logger.info('FP: {} FN: {}'.format(fp, fn))
                # for i in range(len(k_fold_test_f1)):
                #     self.logger.info('Fold {}: {}'.format(i, k_fold_test_f1[i]))
                # self.logger.info('Mean F1 score across folds: {}'.format(np.mean(k_fold_test_f1)))
                # self.logger.info('Median F1 score across folds: {}'.format(np.median(k_fold_test_f1)))
                # self.logger.info('Total train time: %.2f hrs' % ((end - start) / 3600.))

                if self.args.save_as_csv:
                    column_names = ['F1 score', 'AUC', 'True -', 'False +', 'False -', 'True +', 'Total running time (hours)']
                    csv_df = pd.DataFrame(columns=column_names)
                    csv_df.loc[len(csv_df.index)] = [cv_f1, cv_auc, tn, fp, fn, tp, (end - start) / 3600.]
                    csv_df.to_csv('{}/perf_dump.csv'.format(self.log_dir), index=False)

        return


    def train_model(self, model, train_loader, val_loader, test_loader, outer_fold):
        best_val_loss = np.inf
        best_val_f1 = 0.

        # Init optimizer and lr scheduler
        if not self.args.use_abun:
            optimizer_0 = optim.RMSprop([
                {'params': [model.spat_attn.kappa], 'lr': self.args.lr_kappa},
                {'params': [model.spat_attn.eta], 'lr': self.args.lr_eta},
                {'params': [model.time_attn.abun_a, model.time_attn.slope_a], 'lr': self.args.lr_time},
                {'params': [model.time_attn.abun_b, model.time_attn.slope_b], 'lr': self.args.lr_mu},
                {'params': [model.thresh_func.thresh], 'lr': self.args.lr_thresh},
                {'params': [model.slope_func.slope], 'lr': self.args.lr_slope},
                {'params': [model.rules.alpha, model.rules_slope.alpha], 'lr': self.args.lr_alpha},
                {'params': [model.fc.weight], 'lr': self.args.lr_fc},
                {'params': [model.fc.bias], 'lr': self.args.lr_bias},
                {'params': [model.fc.beta], 'lr': self.args.lr_beta},
            ])
        else:
            optimizer_0 = optim.RMSprop([
                {'params': [model.spat_attn.kappa], 'lr': self.args.lr_kappa},
                {'params': [model.spat_attn.eta], 'lr': self.args.lr_eta},
                {'params': [model.time_attn.abun_a], 'lr': self.args.lr_time},
                {'params': [model.time_attn.abun_b], 'lr': self.args.lr_mu},
                {'params': [model.thresh_func.thresh], 'lr': self.args.lr_thresh},
                {'params': [model.rules.alpha], 'lr': self.args.lr_alpha},
                {'params': [model.fc.weight], 'lr': self.args.lr_fc},
                {'params': [model.fc.bias], 'lr': self.args.lr_bias},
                {'params': [model.fc.beta], 'lr': self.args.lr_beta},
            ])
        scheduler_0 = optim.lr_scheduler.CosineAnnealingLR(optimizer_0, self.args.epochs)

        dirName = '{}/fold_{}'.format(self.log_dir, outer_fold)
        self.create_dir(dirName)

        if self.args.save_as_csv:
            column_names = ['Epoch', 'Total loss', 'Train CE loss', 'Val CE loss', 'Test CE loss', 'NegBin rule loss', 'NegBin detector loss',\
            'L2 weights loss', 'Time window abundance loss', 'Time window slope loss']
            losses_csv = pd.DataFrame(columns=column_names)

        ls = list()
        val_ls = list()
        test_ls = list()
        cels = list()
        zls = list()
        zrls = list()
        wls = list()
        tls = list()
        tsls = list()
        tlsa = list()
        tlsb = list()
        tslsa = list()
        tslsb = list()
        embls = list()
        kls = list()
        rnls = list()
        dnls = list()

        # Training loop
        for epoch in range(self.args.epochs):
            train_loss_avg = AverageMeter()
            val_loss_avg = AverageMeter()
            train_f1_avg = AverageMeter()

            # Compute temperature for binary concrete
            # Linearly anneal from t_max to t_min every epoch
            k_alpha = self.linear_anneal(epoch, self.args.max_k_bc,
                self.args.min_k_bc, self.args.epochs,
                1, self.args.min_k_bc)
            k_beta = self.linear_anneal(epoch, self.args.max_k_bc,
                self.args.min_k_bc, self.args.epochs,
                1, self.args.min_k_bc)
            k_otu = self.linear_anneal(epoch, self.args.max_k_otu,
                self.args.min_k_otu, self.args.epochs,
                1, self.args.min_k_otu)
            k_time = self.linear_anneal(epoch, self.args.max_k_time,
                self.args.min_k_time, self.args.epochs,
                1, self.args.min_k_time)
            k_thresh = self.linear_anneal(epoch, self.args.max_k_thresh,
                self.args.min_k_thresh, self.args.epochs,
                1, self.args.min_k_thresh)
            k_slope = self.linear_anneal(epoch, self.args.max_k_slope,
                self.args.min_k_slope, self.args.epochs,
                1, self.args.min_k_slope)

            # Switch model to train mode
            model.train()

            # Mini-batch training loop
            for i, batch in enumerate(train_loader):
                # Get data, labels, mask tensors
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                # Forward pass
                outputs = model(data, mask=mask,
                    k_alpha=k_alpha,
                    k_beta=k_beta,
                    k_otu=k_otu,
                    k_time=k_time,
                    k_thresh=k_thresh,
                    k_slope=k_slope,
                    use_noise=False,
                    hard=False,
                )

                # Zero previous grads
                optimizer_0.zero_grad()

                # Loss function on labels
                train_loss = self.ce_loss(outputs, labels)

                # Priors / regularizers
                rules = model.fc.z
                detectors = model.rules.z
                if not self.args.use_abun:
                    detectors_slope = model.rules_slope.z
                negbin_zr_loss = self.negbin_loss(rules.sum(), self.z_r_mean, self.z_r_var)
                if not self.args.use_abun:
                    negbin_z_loss = self.negbin_loss(detectors.sum(dim=-1) + detectors_slope.sum(dim=-1),
                        self.z_mean, self.z_var).sum()
                else:
                    negbin_z_loss = self.negbin_loss(detectors.sum(dim=-1),
                        self.z_mean, self.z_var).sum()
                l2_wts_loss = -self.normal_wts.log_prob(model.fc.weight).sum()
                time_wts_abun = model.time_attn.wts.sum(dim=-1)
                if not self.args.use_abun:
                    time_wts_slope = model.time_attn.wts_slope.sum(dim=-1)
                time_loss = -(torch.sigmoid((time_wts_abun - 1.) * 10).prod(dim=0)).sum()
                if not self.args.use_abun:
                    time_slope_loss = -(torch.sigmoid((time_wts_slope - 2.) * 10).prod(dim=0)).sum()
                emb_normal_loss = -self.normal_emb.log_prob(model.spat_attn.eta).sum()
                time_abun_a_normal_loss = -self.normal_time_abun_a.log_prob(model.time_attn.abun_a).sum()
                if not self.args.use_abun:
                    time_slope_a_normal_loss = -self.normal_time_slope_a.log_prob(model.time_attn.slope_a).sum()
                time_abun_b_normal_loss = -self.normal_time_abun_b.log_prob(model.time_attn.abun_b).sum()
                if not self.args.use_abun:
                    time_slope_b_normal_loss = -self.normal_time_slope_b.log_prob(model.time_attn.slope_b).sum()
                rule_normal_loss = -self.normal_rule.log_prob(model.fc.beta).sum()
                if not self.args.use_abun:
                    det_normal_loss = -(self.normal_det.log_prob(model.rules.alpha).sum() + \
                        self.normal_det.log_prob(model.rules_slope.alpha).sum())
                else:
                    det_normal_loss = -(self.normal_det.log_prob(model.rules.alpha).sum())
                kappa_normal_loss = -self.normal_kappa.log_prob(model.spat_attn.kappa).sum()
                
                rule_bc_loss = self.binary_concrete_loss(1/k_beta, 1, rules + 1e-5).sum()
                det_abun_bc_loss = self.binary_concrete_loss(1/k_alpha, 1, detectors + 1e-5).sum()
                if not self.args.use_abun:
                    det_slope_bc_loss = self.binary_concrete_loss(1/k_alpha, 1, detectors_slope + 1e-5).sum()

                if not self.args.use_abun:
                    reg_loss = negbin_zr_loss + negbin_z_loss + l2_wts_loss + \
                        emb_normal_loss + time_loss + time_slope_loss + \
                        time_abun_a_normal_loss + time_slope_a_normal_loss + \
                        time_abun_b_normal_loss + time_slope_b_normal_loss + \
                        kappa_normal_loss
                else:
                    reg_loss = negbin_zr_loss + negbin_z_loss + l2_wts_loss + \
                        emb_normal_loss + time_loss + \
                        time_abun_a_normal_loss + \
                        time_abun_b_normal_loss + \
                        kappa_normal_loss

                # Backprop for computing grads
                loss = train_loss + reg_loss
                loss.backward()

                if self.args.debug:
                    grad_dict = {k: v.grad for k, v in model.named_parameters()}
                    model.grad_dict = grad_dict
                    torch.save(model, '{}/model_epoch_{}.pth'.format(dirName, epoch))

                # Update parameters
                optimizer_0.step()

                # Uniform prior implemented avia clipping
                model.thresh_func.thresh.data.clamp_(0, 1)
                if not self.args.use_abun:
                    model.slope_func.slope.data.clamp_(-1, 1)

                # track avg loss, f1
                train_loss_avg.update(train_loss, 1)
                train_f1 = f1_score(labels.detach().cpu().numpy(),
                    outputs.sigmoid().detach().cpu().numpy() > 0.5)
                train_f1_avg.update(train_f1, 1)

            # Evaluate on val data
            val_loss, val_f1 = self.eval(model, val_loader)
            # Save stats and model for best val performance
            # if val_loss <= best_val_loss:
            if val_f1 >= best_val_f1:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                best_model = deepcopy(model)

            # Track test data stats
            test_loss, test_f1 = self.eval(model, test_loader)

            # ls.append(loss.item())
            # val_ls.append(val_loss)
            # test_ls.append(test_loss)
            # cels.append(train_loss.item())
            # zls.append(negbin_z_loss.item())
            # zrls.append(negbin_zr_loss.item())
            # wls.append(l2_wts_loss.item())
            # tls.append(time_loss.item())
            # tsls.append(time_slope_loss.item())
            # tlsa.append(time_abun_a_normal_loss.item())
            # tlsb.append(time_abun_b_normal_loss.item())
            # tslsa.append(time_slope_a_normal_loss.item())
            # tslsb.append(time_slope_b_normal_loss.item())
            # embls.append(emb_normal_loss.item())
            # kls.append(kappa_normal_loss.item())
            # rnls.append(rule_normal_loss.item())
            # dnls.append(det_normal_loss.item())

            if self.args.save_as_csv:
                losses_csv.loc[len(losses_csv.index)] = [epoch, loss.item(), train_loss.item(),\
                val_loss, test_loss, negbin_zr_loss.item(),\
                negbin_z_loss.item(), l2_wts_loss.item(), time_loss.item(), time_slope_loss.item()]

            # Print epoch stats
            kfold_epochs = 'Outer kfold: %d Epoch: %d '
            train_stats = 'TrainLoss: %.2f TrainF1: %.2f '
            val_stats = 'ValLoss: %.2f ValF1: %.2f '
            test_stats = 'TestLoss: %.2f TestF1: %.2f '
            time_stats = 'zrloss: %.2f '
            log_str = kfold_epochs + train_stats + val_stats + test_stats + time_stats
            if not self.args.rank:
                self.logger.info(
                    log_str % (outer_fold, epoch,
                        train_loss_avg.avg, train_f1_avg.avg,
                        val_loss, val_f1, test_loss, test_f1,
                        negbin_zr_loss.item(),
                    )
                )

            scheduler_0.step()


        if self.args.save_as_csv:
            losses_csv.to_csv('{}/losses_dump.csv'.format(dirName), index=False)

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

                outputs = model(data, mask=mask,
                    k_alpha=self.args.max_k_bc,
                    k_beta=self.args.max_k_bc,
                    k_otu=self.args.max_k_otu,
                    k_time=self.args.max_k_time,
                    k_thresh=self.args.max_k_thresh,
                    k_slope=self.args.max_k_slope,
                    hard=False, use_noise=False)

                val_loss = self.ce_loss(outputs, labels)

                val_loss_avg.update(val_loss, 1)
                val_preds.extend(outputs.sigmoid().detach().cpu().numpy() > 0.5)
                val_true.extend(labels.detach().cpu().numpy())

        if len(val_true) > 1:
            val_f1 = f1_score(val_true, val_preds)
        else:
            val_f1 = float(labels.detach().cpu().numpy() == (outputs.sigmoid().detach().cpu().numpy() > 0.5))

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

                outputs = model(data, mask=mask,
                    k_alpha=self.args.max_k_bc,
                    k_beta=self.args.max_k_bc,
                    k_otu=self.args.max_k_otu,
                    k_time=self.args.max_k_time,
                    k_thresh=self.args.max_k_thresh,
                    k_slope=self.args.max_k_slope,
                    hard=False, use_noise=False)

                probs.extend(outputs.sigmoid().detach().cpu().numpy())
                true.extend(labels.detach().cpu().numpy())

        return probs, true


    # cross entropy loss
    def ce_loss(self, logits, labels):
        if labels.size(0) > 1:
            if labels.sum() > 0:
                pos_weight = (labels.size()[0] / labels.sum()) - 1
            else:
                pos_weight = None
        else:
            pos_weight = None

        loss = F.binary_cross_entropy_with_logits(logits, labels,
            reduction='sum', pos_weight=None)

        return loss


    def negbin_loss(self, x, mean, var):
        r = mean ** 2 / (var - mean)
        p = (var - mean) / var
        loss_1 = -torch.lgamma(r + x + 1e-5)
        loss_2 = torch.lgamma(x + 1)
        loss_3 = -(x * np.log(p))
        loss = loss_1 + loss_2 + loss_3
        return loss


    def binary_concrete_loss(self, temp, alpha, x):
        try:
            loss_1 = (temp + 1) * (torch.log(x * (1 - x) + 1e-5))
            loss_2 = 2 * (torch.log((alpha / ((x ** temp) + 1e-5)) + (1 / ((1 - x) ** temp) + 1e-5) + 1e-5))
        except Exception as e:
            print(loss_1, loss_2)
            print(x)
            raise e
        
        return loss_1 + loss_2


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


    def get_rule(self, w, log_odds, r, d, t_min, t_max, thresh, metric='slope'):
        if w > 0.:
            out = self.label_1
        else:
            out = self.label_0
        rule = "Rule {} Detector {}: TRUE for {} if the average {} of selected taxa\nbetween days {} to {} is greater than {:.4f}. (Log-odds: {:.2f})".format(
            r, d, out, metric, t_min, t_max, thresh, log_odds)
        return rule


    def heatmap_rule_viz(self, x_0, x_1, x_mean_0, x_mean_1,
        mask_0, mask_1, x_ticks,
        tree, thresh, det_type, win_start, win_end,
        rule_eng, rule_path, t_min, t_max, pdf=None, view_type=None):
        plt.rcParams["font.family"] = 'sans-serif'

        fig = plt.figure(figsize=(40, 40))
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        gs = fig.add_gridspec(3, 3,
            width_ratios=[x_0.shape[1], 0.05 * x_0.shape[1], 0.05 * x_0.shape[1]],
            height_ratios=[x_0.shape[0], x_1.shape[0], 3 * max(x_0.shape[0], x_1.shape[0])])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        cbar_ax = fig.add_subplot(gs[0:2, 2])

        x_0_ma = np.ma.masked_array(x_0, mask=np.logical_not(mask_0))
        x_1_ma = np.ma.masked_array(x_1, mask=np.logical_not(mask_1))
        x_mean_0_ma = np.ma.masked_array(x_mean_0, mask=np.zeros_like(x_mean_0))
        x_mean_1_ma = np.ma.masked_array(x_mean_1, mask=np.zeros_like(x_mean_1))

        x_ma_con = np.ma.concatenate((x_0_ma, x_1_ma), axis=0)
        x_mean_ma_con = np.ma.concatenate((x_mean_0_ma, x_mean_1_ma), axis=0)
        x_con = np.ma.concatenate((x_ma_con, x_mean_ma_con[:, np.newaxis]), axis=1)
        x_min = np.ma.min(x_con)
        x_max = np.ma.max(x_con)

        # x_0_inter = pd.DataFrame(x_0).interpolate(method='linear', limit_direction ='both', axis=1).to_numpy()
        # x_1_inter = pd.DataFrame(x_1).interpolate(method='linear', limit_direction ='both', axis=1).to_numpy()
        # alphas_0 = mask_0 + np.logical_not(mask_0).astype('int') * 0.3
        # alphas_1 = mask_1 + np.logical_not(mask_1).astype('int') * 0.3
        # x_min = min(np.min(x_0_inter), np.min(x_1_inter))
        # x_max = max(np.max(x_0_inter), np.max(x_1_inter))
        # x_mean_min = min(np.min(x_mean_0), np.min(x_mean_1))
        # x_mean_max = max(np.max(x_mean_0), np.max(x_mean_1))
        # x_mean_min = min(x_min, x_mean_min)
        # x_mean_max = max(x_max, x_mean_max)

        # colors_undersea = plt.cm.terrain(np.linspace(0, thresh + x_mean_min, 256))
        # colors_land = plt.cm.terrain(np.linspace(thresh + x_mean_min, x_mean_min + x_mean_max, 256))
        # all_colors = np.vstack((colors_undersea, colors_land))
        # terrain_map = mc.LinearSegmentedColormap.from_list(
        #     'terrain_map', all_colors)
        # cmap = mc.ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
        # cmap_colors = ['#5d74a5', '#b0cbe7', '#fff0b4', '#eba07e', '#a45851']
        # cmap = mc.LinearSegmentedColormap.from_list("", cmap_colors)
        # Create a normalizer that goes from minimum to maximum temperature
        # norm = mc.Normalize(x_mean_min, x_mean_max)
        # norm = mc.TwoSlopeNorm(vmin=x_mean_min, vcenter=thresh, vmax=x_mean_max)

        class MidpointNormalize(mc.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                mc.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        # cmap_colors = ['blue', 'green', 'yellow', 'red']
        # cmap_0 = mc.LinearSegmentedColormap.from_list("", cmap_colors[:2])
        # cmap_1 = mc.LinearSegmentedColormap.from_list("", cmap_colors[2:])
        cmap_0 = plt.get_cmap('cool_r', 256)
        cmap_1 = plt.get_cmap('autumn_r', 256)
        newcolors = np.vstack((cmap_0(np.linspace(0, 1, 256)),
                               cmap_1(np.linspace(0, 1, 256))))
        cmap = mc.ListedColormap(newcolors)
        divnorm = mc.TwoSlopeNorm(vmin=x_min, vcenter=thresh, vmax=x_max)
        # divnorm = MidpointNormalize(vmin=x_min, midpoint=thresh, vmax=x_max)
        # divnorm_0 = MidpointNormalize(vmin=np.min(x_0), midpoint=thresh, vmax=np.max(x_0))
        # divnorm_1 = MidpointNormalize(vmin=np.min(x_1), midpoint=thresh, vmax=np.max(x_1))
        # divnorm_mean_0 = MidpointNormalize(vmin=np.min(x_mean_0), midpoint=thresh, vmax=np.max(x_mean_0))
        # divnorm_mean_1 = MidpointNormalize(vmin=np.min(x_mean_0), midpoint=thresh, vmax=np.max(x_mean_0))
        # divnorm_0 = MidpointNormalize(vmin=x_mean_min, midpoint=thresh, vmax=x_mean_max)
        # divnorm_1 = MidpointNormalize(vmin=x_mean_min, midpoint=thresh, vmax=x_mean_max)
        # divnorm_mean_0 = MidpointNormalize(vmin=x_mean_min, midpoint=thresh, vmax=x_mean_max)
        # divnorm_mean_1 = MidpointNormalize(vmin=x_mean_min, midpoint=thresh, vmax=x_mean_max)

        h1 = sns.heatmap(x_0, ax=ax0, cmap=cmap,
            xticklabels=x_ticks, yticklabels=False,
            cbar=False,
            # robust=True,
            # vmin=x_min,
            # vmax=x_max,
            # alpha=alphas_0,
            # shading='auto',
            # center=thresh,
            norm=divnorm, rasterized=True,
            mask=np.logical_not(mask_0))
        for i in range(x_0.shape[0]):
            for j in range(x_0.shape[1]):
                if mask_0[i, j]:
                    h1.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='k', lw=0.5))
        h2 = sns.heatmap(x_1, ax=ax2, cmap=cmap,
            xticklabels=x_ticks, yticklabels=False,
            # robust=True,
            # vmin=x_min,
            # vmax=x_max,
            cbar=False,
            # alpha=alphas_1,
            # shading='auto',
            # center=thresh,
            norm=divnorm, rasterized=True,
            mask=np.logical_not(mask_1))
        for i in range(x_1.shape[0]):
            for j in range(x_1.shape[1]):
                if mask_1[i, j]:
                    h2.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='k', lw=0.5))
        h3 = sns.heatmap(x_mean_0[:, np.newaxis], ax=ax1, cmap=cmap,
                    xticklabels=False, yticklabels=False,
                   cbar=False,
                   # vmin=x_min,
                   # vmax=x_max,
                   # shading='auto',
                   # center=thresh,
                   norm=divnorm, rasterized=True,
                   )
        h4 = sns.heatmap(x_mean_1[:, np.newaxis], ax=ax3, cmap=cmap,
                    xticklabels=False, yticklabels=False,
                   # cbar=False,
                   # vmin=x_min,
                   # vmax=x_max,
                   # shading='auto',
                   # center=thresh,
                   norm=divnorm,
                   cbar_ax=cbar_ax,
                   cbar_kws={
                       'orientation': "vertical",
                   }, rasterized=True,
                   )

        # ax0_pc = ax0.pcolormesh(x_0_ma, cmap=cmap, norm=divnorm,
        #     shading='auto', rasterized=True)
        # ax1_pc = ax1.pcolormesh(x_mean_0_ma[:, np.newaxis], cmap=cmap, norm=divnorm,
        #     shading='auto', rasterized=True)
        # ax2_pc = ax2.pcolormesh(x_1_ma, cmap=cmap, norm=divnorm,
        #     shading='auto', rasterized=True)
        # ax3_pc = ax3.pcolormesh(x_mean_1_ma[:, np.newaxis], cmap=cmap, norm=divnorm,
        #     shading='auto', rasterized=True)

        # for i in range(x_0.shape[0]):
        #     for j in range(x_0.shape[1]):
        #         if mask_0[i, j]:
        #             ax0.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='k', lw=1))
        # for i in range(x_1.shape[0]):
        #     for j in range(x_1.shape[1]):
        #         if mask_1[i, j]:
        #             ax2.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='k', lw=1))

        # # make frame visible
        for _, spine in h1.spines.items():
            spine.set_visible(True)
        for _, spine in h2.spines.items():
            spine.set_visible(True)
        for _, spine in h3.spines.items():
            spine.set_visible(True)
        for _, spine in h4.spines.items():
            spine.set_visible(True)

        win_start_idx = x_ticks.tolist().index(max([t for t in x_ticks if t <= win_start]))
        win_end_idx = x_ticks.tolist().index(min([t for t in x_ticks if t >= win_end]))
        # win_mid = (ax2.get_xticks()[win_start_idx] + ax2.get_xticks()[win_end_idx]) // 2
        # ax0.axvline(win_start_idx, linewidth=8, alpha=0.7, color='black')
        # ax0.axvline(win_end_idx + 1, linewidth=8, alpha=0.7, color='black')
        ax0.add_patch(Rectangle((win_start_idx, 0), win_end_idx - win_start_idx + 1, x_0.shape[0],
            alpha=0.7, fill=False, edgecolor='k', lw=6))
        ax0.text(ax2.get_xticks()[win_start_idx], -.05, str(t_min), color='black', transform=ax0.get_xaxis_transform(),
            ha='center', va='top', fontsize=20)
        ax0.text(ax2.get_xticks()[win_end_idx], -.05, str(t_max), color='black', transform=ax0.get_xaxis_transform(),
            ha='center', va='top', fontsize=20)
        # ax2.axvline(win_start_idx, linewidth=8, alpha=0.7, color='black')
        # ax2.axvline(win_end_idx + 1, linewidth=8, alpha=0.7, color='black')
        ax2.add_patch(Rectangle((win_start_idx, 0), win_end_idx - win_start_idx + 1, x_0.shape[0],
            alpha=0.7, fill=False, edgecolor='k', lw=6))
        ax2.text(ax2.get_xticks()[win_start_idx], -.05, str(t_min), color='black', transform=ax2.get_xaxis_transform(),
            ha='center', va='top', fontsize=20)
        ax2.text(ax2.get_xticks()[win_end_idx], -.05, str(t_max), color='black', transform=ax2.get_xaxis_transform(),
            ha='center', va='top', fontsize=20)
        # ax0.text(x=win_mid, y=-0.2, s='Selected time window',
        #         horizontalalignment='center', fontsize=12)
        ax2.text(np.median(ax2.get_xticks()), -.15, s='Time (days)',
                horizontalalignment='center', fontsize=20, transform=ax2.get_xaxis_transform())
        ax0.set_ylabel('{}'.format(self.label_0), fontsize=20)
        ax2.set_ylabel('{}'.format(self.label_1), fontsize=20)
        # ax2.set_xlabel('Time (days)', fontsize=12)
        # ax2.xaxis.set_label_position('top')
        if det_type == 'Slope' and view_type == 'Slope':
            ax0.set_title('Average slope over time', pad=30, fontsize=20)
        else:
            ax0.set_title('Abundances over time', pad=30, fontsize=20)
        ax0.set_xticks([])
        ax2.set_xticks([])

        cbar_ax.axhline(thresh, linewidth=6, color='black', alpha=0.7)
        y_pos = (thresh - x_min) / (x_max - x_min)
        # cb.ax.annotate('{:.4f}'.format(thresh),
        #                xy=(1.1, y_pos),
        #                xycoords='axes fraction',
        #               bbox=dict(facecolor='none'),
        #               fontsize='smaller')
        ax1.set_title('Average {} over\ndays {} to {}'.format(det_type, t_min, t_max),
            wrap=True, pad=30, fontsize=20)

        # tree_ax = fig.add_subplot(gs[2, 1])
        tree_grid = gridspec.GridSpecFromSubplotSpec(1,2, gs[2, :], width_ratios=[1, 2])
        tree_ax = fig.add_subplot(tree_grid[0, 0])
        t = plot_tree(tree, axe=tree_ax, font_size=16)
        tree_ax.set_axis_off()
        tree_ax.set_title('Selected sub-tree of taxa', fontsize=20, fontweight='bold')

        taxa_to_star = {
            'species\n': '*',
            'genus\n': '**',
            'family\n': '***',
            'order\n': '****',
            'class\n': '*****',
            'phylum': '******',
        }
        taxa_str = 'LEGEND\nOTU mapped to:\n'
        for k, v in taxa_to_star.items():
            taxa_str = ' ' + taxa_str + v + ': ' + k
        taxa_ax = fig.add_subplot(tree_grid[0, 1])
        taxa_ax.annotate(taxa_str,
            xy=(1, 0),
            xycoords='axes fraction',
            bbox=dict(facecolor='none'),
            fontsize=14,
            ha='center')
        taxa_ax.set_axis_off()

        plt.suptitle(rule_eng, wrap=True, fontsize=24, fontweight='bold', y=0.98)
        plt.savefig(rule_path, bbox_inches='tight')
        if pdf is not None:
            pdf.savefig(bbox_inches='tight')
        plt.close()


    def show_rules_abun(self, best_model, test_loader, fold, save_viz=True, best_model_init=None):
        dirName = '{}/fold_{}'.format(self.log_dir, fold)
        self.create_dir(dirName)
        torch.save(best_model.state_dict(), '{}/best_model.pth'.format(dirName))

        taxa_to_star = {
            'species': '*',
            'genus': '**',
            'family': '***',
            'order': '****',
            'class': '*****',
            'phylum': '******',
        }

        if self.args.save_as_csv:
            column_names = ['rule_id', 'detector_id', 'detector_type', 'csv_filepath', 'significant_taxa', 'otu_radius_kappa', 'time_window_low',\
            'time_window_high', 'abundance_threshold', 'slope_threshold', 'median_thresh_0', 'median_thresh_1', 'weight', 'bias']
            csv_df = pd.DataFrame(columns=column_names)

        if save_viz:
            best_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    data = batch['data'].to(self.device)
                    labels = batch['label'].to(self.device)
                    mask = batch['mask'].to(self.device)

                    x_spat = best_model.spat_attn(data, k=self.args.max_k_otu)
                    x_time = best_model.time_attn(x_spat, mask=mask, k=self.args.max_k_time)
                    x_thresh = best_model.thresh_func(x_time, k=self.args.max_k_thresh)
                    x = best_model.rules(x_thresh, k=self.args.max_k_bc, hard=False)
                    outputs = best_model.fc(x, k=self.args.max_k_bc, hard=False)

            x_spat = x_spat.detach().cpu().numpy()
            x_time = x_time.detach().cpu().numpy()
            otu_wts = best_model.spat_attn.wts.detach().cpu().numpy()
            time_mu = (best_model.time_attn.m).detach().cpu().numpy()
            time_sigma = (best_model.time_attn.s_abun).detach().cpu().numpy()
            time_wts = best_model.time_attn.wts.detach().cpu().numpy()
            rules = best_model.fc.z.detach().cpu().numpy()
            detectors = best_model.rules.z.detach().cpu().numpy()
            threshs = best_model.thresh_func.thresh.detach().cpu().numpy()
            # threshs = x_time.mean(axis=0)
            # slopes = x_time_slope.mean(axis=0)
            labels = labels.detach().cpu().numpy()
            fc_wts = best_model.fc.weight.view(-1).detach().cpu().numpy()
            fc_bias = best_model.fc.bias.detach().cpu().item()
            kappas = (best_model.spat_attn.kappas).detach().cpu().numpy()
            etas = (best_model.spat_attn.eta).detach().cpu().numpy()
            dist_emb = (best_model.spat_attn.emb_dist).detach().cpu().numpy()
            x_thresh = x_thresh.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            model_prob = torch.sigmoid(outputs).detach().cpu().numpy()
            rule_log_odds = fc_wts * rules + fc_bias

            rules_dict = {
                'subjects_0_log_odds': self.subjects_0_log_odds,
                'subjects_1_log_odds': self.subjects_1_log_odds,
                'sub_0_log_odds': self.sub_0_log_odds,
                'sub_1_log_odds': self.sub_1_log_odds,
                'log_odds': rule_log_odds,
                'taxa_tree': [],
                'full_tree': [],
                'exp_duration': self.num_time,
                'time_unique': self.time_unique,
                'x_0': [],
                'x_1': [],
                'x_mask_0': [],
                'x_mask_1': [],
                'x_avg_0': [],
                'x_avg_1': [],
                'det_type': [],
                'outcome_0': self.label_0,
                'outcome_1': self.label_1,
                'thresh': [],
                't_min': [],
                't_max': [],
                'new_t_min': [],
                'new_t_max': [],
            }

            if best_model_init is not None:
                torch.save(best_model_init.state_dict(), '{}/best_model_init.pth'.format(dirName))
                best_model_init.eval()
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        data_init = batch['data'].to(self.device)
                        labels_init = batch['label'].to(self.device)
                        mask_init = batch['mask'].to(self.device)

                        x_spat_init = best_model_init.spat_attn(data_init, k=self.args.max_k_otu)
                        x_time_init, x_time_slope_init = best_model_init.time_attn(x_spat_init, mask=mask_init, k=self.args.max_k_time)
                        x_thresh_init = best_model_init.thresh_func(x_time_init, k=self.args.max_k_thresh)
                        x_init = best_model_init.rules(x_thresh_init, k=self.args.max_k_bc, hard=False)
                        x_slope_init = best_model_init.slope_func(x_time_slope_init, k=self.args.max_k_slope)
                        x_slope_init = best_model_init.rules(x_slope_init, k=self.args.max_k_bc, hard=False)
                        outputs_init = best_model_init.fc(x_init, x_slope_init, k=self.args.max_k_bc, hard=False)

                x_spat_init = x_spat_init.detach().cpu().numpy()
                x_time_init = x_time_init.detach().cpu().numpy()
                x_time_slope_init = x_time_slope_init.detach().cpu().numpy()
                otu_wts_init = best_model_init.spat_attn.wts.detach().cpu().numpy()
                time_mu_init = (best_model_init.time_attn.m).detach().cpu().numpy()
                time_mu_slope_init = (best_model_init.time_attn.m_slope).detach().cpu().numpy()
                time_wts_init = best_model_init.time_attn.wts.detach().cpu().numpy()
                time_wts_slope_init = best_model_init.time_attn.wts_slope.detach().cpu().numpy()
                threshs_init = best_model_init.thresh_func.thresh.detach().cpu().numpy()
                slopes_init = best_model_init.slope_func.slope.detach().cpu().numpy()
                labels_init = labels_init.detach().cpu().numpy()
                fc_wts_init = best_model_init.fc.weight.view(-1).detach().cpu().numpy()
                fc_bias_init = best_model_init.fc.bias.detach().cpu().item()
                kappas_init = (best_model_init.spat_attn.kappas).detach().cpu().numpy()
                etas_init = (best_model_init.spat_attn.eta).detach().cpu().numpy()
                dist_emb_init = (best_model_init.spat_attn.emb_dist).detach().cpu().numpy()


            # Create the PdfPages object to which we will save the pages:
            # The with statement makes sure that the PdfPages object is closed properly at
            # the end of the block, even if an Exception occurs.
            with PdfPages('{}/rules.pdf'.format(dirName)) as pdf:
                num_rules = 0
                num_det_per_rule = 0
                for i, r in enumerate(rules):
                    has_det = False
                    if r >= 0.9:
                        for p in range(self.num_detectors):
                            d = detectors[i, p]
                            if d >= 0.9:
                                has_det = True
                            if d >= 0.9:
                                num_det_per_rule += 1
                        if has_det:
                            num_rules += 1

                if num_rules == 0:
                    num_rules = 1
                num_det_per_rule = num_det_per_rule / num_rules
                fig = plt.figure(figsize=(20, 18))
                ax = fig.add_subplot()
                rule_str = 'Number of Active rules (selection wt > 0.9): {}'.format(num_rules)
                det_str = 'Number of Active detectors per active rule (selection wt > 0.9): {:.2f}'.format(num_det_per_rule)
                full_str = rule_str + '\n' + det_str
                ax.text(0, 1, full_str,
                    horizontalalignment="left",
                    verticalalignment="top", fontsize=12,)
                ax.set_axis_off()
                pdf.savefig(fig, dpi=1200)
                plt.close()

                for i, r in enumerate(rules):
                    if r >= 0.:
                        taxa_tree = list()
                        full_tree = list()
                        x_abun_slope_0 = list()
                        x_avg_abun_slope_0 = list()
                        x_abun_slope_1 = list()
                        x_avg_abun_slope_1 = list()
                        x_mask_0 = list()
                        x_mask_1 = list()
                        x_det_type = list()
                        x_threshold = list()
                        x_new_t_min = list()
                        x_new_t_max = list()
                        x_t_min = list()
                        x_t_max = list()
                        for p in range(self.num_detectors):
                            d = detectors[i, p]
                            if d >= 0.9:
                                kappa = kappas[i, p]
                                eta = etas[i, p]
                                d_e = dist_emb[i, p]
                                thresh = threshs[i, p]
                                mu = time_mu[i, p]
                                t = time_wts.mean(axis=0)[i, p]
                                t_min = int(np.floor(time_mu[i, p] - (time_sigma[i, p] // 2)))
                                t_max = int(np.ceil(time_mu[i, p] + (time_sigma[i, p] // 2)))
                                o = otu_wts[i, p]
                                sel_otu_ids = [l for l, ot in enumerate(o) if ot >= 0.9]
                                sel_dist = np.zeros((len(sel_otu_ids), len(sel_otu_ids)))
                                for ii in range(len(sel_otu_ids)):
                                    for jj in range(len(sel_otu_ids)):
                                        sel_dist[ii, jj] = self.dist_matrix_embed[sel_otu_ids[ii], sel_otu_ids[jj]]
                                sel_dist_emb = np.zeros((len(sel_otu_ids)))
                                for ii in range(len(sel_otu_ids)):
                                    sel_dist_emb[ii] = d_e[sel_otu_ids[ii]]
                                sel_otus = [self.var_ids[l] for l, ot in enumerate(o) if ot >= 0.9]
                                otu_annot_str = ''
                                if self.var_annot == {}:
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + '\n' + n
                                else:
                                    # sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + ',' + n

                                tree = deepcopy(self.phylo_tree)
                                sel_nodes = list()
                                for n in tree.traverse():
                                    if n.name in sel_otus:
                                        sel_nodes.append(n)
                                if sel_nodes == []:
                                    print(np.sort(o))
                                    print(kappa)
                                else:
                                    nstyle = NodeStyle()
                                    fca = None
                                    for n in tree.traverse():
                                        if n.name in sel_otus:
                                            fca = n.get_common_ancestor(sel_nodes)
                                            break
                                    # for n in fca.traverse():
                                    #     nstyle = NodeStyle()
                                    #     nstyle["size"] = 0
                                    #     if n.name in sel_otus:
                                    #         nstyle["vt_line_color"] = "red"
                                    #         nstyle["hz_line_color"] = "red"
                                    #         nstyle["vt_line_width"] = 2
                                    #         nstyle["hz_line_width"] = 2
                                    #         nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                    #         nstyle["hz_line_type"] = 0
                                    #     n.set_style(nstyle)

                                    # ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    # ts.show_branch_length = False
                                    # ts.branch_vertical_margin = 4
                                    # ts.min_leaf_separation = 4
                                    # ts.show_scale = False
                                    # fca.render('{}/r{}d{}_tree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)
                                    fca_full_copy = deepcopy(fca)
                                    fca_copy = deepcopy(fca)
                                    
                                    for n in fca.get_leaves():
                                        if n.name in sel_otus:
                                            # tw = textwrap.TextWrapper(width=30)
                                            # if self.var_annot == {}:
                                            #     nn_split = n.name.split('|')[-1]
                                            #     nn_split = nn_split.split('__')
                                            #     for k, v in taxa_to_star.items():
                                            #         if k.lower()[0] == nn_split[0].lower():
                                            #             taxa_name = v
                                            #     node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                            # else:
                                            #     node_name = self.var_annot.get(n.name, '(no annotation)')
                                            #     node_name = node_name.split(" ")
                                            #     remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                            #     new_node_name = ''
                                            #     for l in node_name:
                                            #         if not l in remove_list:
                                            #             new_node_name = new_node_name + l + ' '
                                            #     node_name = new_node_name
                                            #     if ',including' in node_name:
                                            #         node_name = node_name.replace(',including', '').split(" ")[:2]
                                            #         node_name = ' '.join(map(str, node_name))
                                            #     node_name = '{}'.format(node_name)
                                            #     n_split = node_name.split(' ')
                                            #     new_name = ''
                                            #     taxa_name = ''
                                            #     for nn in n_split:
                                            #         if nn.lower() in taxa_to_star.keys():
                                            #             taxa_name = taxa_to_star[nn.lower()]
                                            #         else:
                                            #             if '/' in nn:
                                            #                 nn = nn.split('/')[0]
                                            #             new_name = new_name + nn + ' '
                                            #     if taxa_name == '':
                                            #         taxa_name = taxa_to_star['species']
                                            #     node_name = '{} {}'.format(taxa_name, new_name)
                                            # text = TextFace(tw.fill(text=node_name), fsize=24)
                                            # n.add_face(text, 0)

                                            n_idx = sel_otu_ids[sel_otus.index(n.name)]
                                            text = 'OTU{}'.format(n_idx)
                                            n.name = text
                                        else:
                                            n.delete()

                                    ts = TreeStyle()
                                    ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 10
                                    ts.min_leaf_separation = 10
                                    ts.show_scale = False
                                    fca.prune(sel_nodes)
                                    # fca.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)

                                    # for n in fca_copy.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # for ll, n in enumerate(fca_copy.get_leaves()):
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # fca_copy.write(outfile="{}/r{}d{}_subtree_abun.newick".format(
                                    #     dirName, i, p))
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)

                                    while True:
                                        if len(fca_full_copy.get_leaves()) > len(sel_otus):
                                            break
                                        else:
                                            fca_full_copy = fca_full_copy.up

                                    tree_copy = deepcopy(self.phylo_tree)
                                    for n in tree_copy.traverse():
                                        nstyle = NodeStyle()
                                        nstyle["size"] = 0
                                        if n.name in sel_otus:
                                            nstyle["vt_line_color"] = "red"
                                            nstyle["hz_line_color"] = "red"
                                            nstyle["vt_line_width"] = 2
                                            nstyle["hz_line_width"] = 2
                                            nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                            nstyle["hz_line_type"] = 0
                                        n.set_style(nstyle)

                                        if n.is_leaf():
                                            n_idx = self.var_ids.index(n.name)
                                            if self.var_annot == {}:
                                                nn_split = n.name.split('|')[-1]
                                                nn_split = nn_split.split('__')
                                                for k, v in taxa_to_star.items():
                                                    if k.lower()[0] == nn_split[0].lower():
                                                        taxa_name = v
                                                node_name = 'OTU{} {} {}'.format(n_idx, taxa_name, ' '.join(nn_split[1].split('_')))
                                            else:
                                                node_name = self.var_annot.get(n.name, '(no annotation)')
                                                node_name = node_name.split(" ")
                                                remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                                new_node_name = ''
                                                for l in node_name:
                                                    if not l in remove_list:
                                                        new_node_name = new_node_name + l + ' '
                                                node_name = new_node_name
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{}'.format(node_name)
                                                n_split = node_name.split(' ')
                                                new_name = ''
                                                taxa_name = ''
                                                for nn in n_split:
                                                    if nn.lower() in taxa_to_star.keys():
                                                        taxa_name = taxa_to_star[nn.lower()]
                                                    else:
                                                        if '/' in nn:
                                                            nn = nn.split('/')[0]
                                                        new_name = new_name + nn + ' '
                                                if taxa_name == '':
                                                    taxa_name = taxa_to_star['species']
                                                node_name = 'OTU{} {} {}'.format(n_idx, taxa_name, new_name)
                                            n.name = node_name

                                    for n in fca_full_copy.traverse():
                                        nstyle = NodeStyle()
                                        nstyle["size"] = 0
                                        if n.name in sel_otus:
                                            nstyle["vt_line_color"] = "red"
                                            nstyle["hz_line_color"] = "red"
                                            nstyle["vt_line_width"] = 2
                                            nstyle["hz_line_width"] = 2
                                            nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                            nstyle["hz_line_type"] = 0
                                        n.set_style(nstyle)

                                        if n.is_leaf():
                                            if self.var_annot == {}:
                                                nn_split = n.name.split('|')[-1]
                                                nn_split = nn_split.split('__')
                                                for k, v in taxa_to_star.items():
                                                    if k.lower()[0] == nn_split[0].lower():
                                                        taxa_name = v
                                                node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                            else:
                                                node_name = self.var_annot.get(n.name, '(no annotation)')
                                                node_name = node_name.split(" ")
                                                remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                                new_node_name = ''
                                                for l in node_name:
                                                    if not l in remove_list:
                                                        new_node_name = new_node_name + l + ' '
                                                node_name = new_node_name
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{}'.format(node_name)
                                                n_split = node_name.split(' ')
                                                new_name = ''
                                                taxa_name = ''
                                                for nn in n_split:
                                                    if nn.lower() in taxa_to_star.keys():
                                                        taxa_name = taxa_to_star[nn.lower()]
                                                    else:
                                                        if '/' in nn:
                                                            nn = nn.split('/')[0]
                                                        new_name = new_name + nn + ' '
                                                if taxa_name == '':
                                                    taxa_name = taxa_to_star['species']
                                                node_name = '{} {}'.format(taxa_name, new_name)
                                            n.name = node_name

                                    ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 1
                                    ts.min_leaf_separation = 4
                                    ts.show_scale = False
                                    # fca_full_copy.render('{}/r{}d{}_tree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)

                                    sorted_prob_ids = np.argsort(model_prob)
                                    labels_new = labels[sorted_prob_ids]
                                    x_time_new = x_time[sorted_prob_ids, :, :]
                                    x_spat_new = x_spat[sorted_prob_ids, :, :]

                                    # fig = plt.figure()
                                    # fig.subplots_adjust(top=0.8)
                                    # ax = fig.add_subplot(121)
                                    abun_0 = list()
                                    abun_1 = list()
                                    for k in range(self.num_subjects):
                                        t_abun = x_time_new[k, i, p]
                                        if labels_new[k]:
                                            # lines_1, = ax.plot(k, t_abun,
                                            #     marker='+', color='g')
                                            abun_1.append(t_abun)
                                        else:
                                            # lines_0, = ax.plot(k, t_abun,
                                            #     marker='.', color='#FF8C00')
                                            abun_0.append(t_abun)
                                    # line_thresh = ax.axhline(y=thresh,
                                    #     c='k', linestyle='--')
                                    # ax.set_xlabel('Subjects')
                                    # ax.set_ylabel('Abundance')
                                    # ax.legend([lines_0, lines_1, line_thresh],
                                    #     [self.label_0, self.label_1, 'Threshold'],
                                    #     loc='upper left',
                                    #     fancybox=True, framealpha=0.5,
                                    #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                    log_odds = fc_wts[i] * r + fc_bias
                                    rule_eng = self.get_rule(fc_wts[i], log_odds,
                                        i, p, t_min, t_max, thresh, metric='abundance')
                                    # plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    # tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name))
                                    # ax1 = fig.add_subplot(122)
                                    # imgplot = plt.imshow(tree_img)
                                    # ax1.set_axis_off()
                                    # plt.savefig('{}/r{}d{}_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200)
                                    # plt.close()

                                    if self.args.save_as_csv:
                                        column_names = ['Subject', 'Label', *['day_{}'.format(day) for day in np.arange(self.num_time)]]
                                        csv_det_df = pd.DataFrame(columns=column_names)
                                    # fig = plt.figure(figsize=(20, 18))
                                    # fig.subplots_adjust(top=0.8)
                                    # gs = fig.add_gridspec(2, 2)
                                    # f_ax1 = fig.add_subplot(gs[0, 0])
                                    # f_ax2 = fig.add_subplot(gs[1, 0], sharex=f_ax1)
                                    abundances_0 = list()
                                    abundances_1 = list()
                                    mask_0 = list()
                                    mask_1 = list()
                                    for k in range(self.num_subjects):
                                        abun = x_spat_new[k, i, p]
                                        if labels_new[k]:
                                            # lines_1, = f_ax1.plot(np.array(self.times)[sorted_prob_ids][k].astype('int'),
                                            #     abun[np.array(self.times)[sorted_prob_ids][k].astype('int')],
                                            #     marker='.', color='g')
                                            for day in range(self.num_time):
                                                if not day in np.array(self.times)[sorted_prob_ids][k].astype('int'):
                                                    abun[day] = -1.
                                            abundances_1.append(abun)
                                            mask_1.append(mask[sorted_prob_ids][k])
                                        else:
                                            # lines_0, = f_ax2.plot(np.array(self.times)[sorted_prob_ids][k].astype('int'),
                                            #     abun[np.array(self.times)[sorted_prob_ids][k].astype('int')],
                                            #     marker='.', color='#FF8C00')
                                            for day in range(self.num_time):
                                                if not day in np.array(self.times)[sorted_prob_ids][k].astype('int'):
                                                    abun[day] = -1.
                                            abundances_0.append(abun)
                                            mask_0.append(mask[sorted_prob_ids][k])
                                        if self.args.save_as_csv:
                                            for day in range(self.num_time):
                                                if not day in np.array(self.times)[sorted_prob_ids][k].astype('int'):
                                                    abun[day] = -1.
                                            csv_det_df.loc[len(csv_det_df.index)] = [k, labels[k], *abun]
                                    # f_ax1.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    sel_days = np.arange(t_min, t_max + 1)
                                    # line_thresh = f_ax1.axhline(y=thresh,
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='k', linestyle='--', linewidth=3)
                                    # line_thresh_1 = f_ax1.axhline(y=np.median(abun_1),
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='r', linestyle='--', linewidth=3)
                                    # f_ax1.set_ylabel('Relative Abundance')
                                    # f_ax1.set_title(self.label_1)
                                    # plt.setp(f_ax1.get_xticklabels(), visible=False)
                                    # f_ax2.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    # line_thresh = f_ax2.axhline(y=thresh,
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='k', linestyle='--', linewidth=3)
                                    # line_thresh_0 = f_ax2.axhline(y=np.median(abun_0),
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='r', linestyle='--', linewidth=3)
                                    # f_ax2.set_xlabel('Days')
                                    # f_ax2.set_ylabel('Relative Abundance')
                                    # f_ax2.set_title(self.label_0)
                                    log_odds = fc_wts[i] * r + fc_bias
                                    rule_eng = self.get_rule(fc_wts[i], log_odds, i,
                                        p, t_min, t_max, thresh, metric='abundance')
                                    # plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    # tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name))
                                    # f_ax3 = fig.add_subplot(gs[:, 1])
                                    # imgplot = plt.imshow(tree_img)
                                    # f_ax3.set_axis_off()
                                    # pdf.savefig(bbox_inches='tight', dpi=1200)
                                    # plt.savefig('{}/r{}d{}_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200)
                                    # plt.close()

                                    labels_0 = np.array([int(lab) for lab in labels_new if lab == 0])
                                    labels_1 = np.array([int(lab) for lab in labels_new if lab == 1])
                                    new_t_min = min([t for t in self.time_unique if t >= t_min])
                                    new_t_max = max([t for t in self.time_unique if t <= t_max])
                                    rule_path = '{}/r{}d{}_heatmap_abun_{}.pdf'.format(
                                        dirName, i, p, self.args.data_name)
                                    new_t = np.array([t for t in self.time_unique if t >= t_min and t <= t_max])
                                    x_0 = np.array(abundances_0)[:, self.time_unique]
                                    x_1 = np.array(abundances_1)[:, self.time_unique]
                                    x_0[x_0 == -1] = np.nan
                                    x_1[x_1 == -1] = np.nan
                                    # self.heatmap_rule_viz(x_0, x_1,
                                    #     np.array(abun_0), np.array(abun_1), np.array(mask_0)[:, self.time_unique],
                                    #     np.array(mask_1)[:, self.time_unique], self.time_unique,
                                    #     fca_full_copy, thresh, 'Abun', new_t_min, new_t_max,
                                    #     rule_eng, rule_path, t_min, t_max, pdf=pdf)

                                    x_abun_slope_0.append(x_0)
                                    x_abun_slope_1.append(x_1)
                                    x_avg_abun_slope_0.append(np.array(abun_0))
                                    x_avg_abun_slope_1.append(np.array(abun_1))
                                    x_mask_0.append(np.array(mask_0)[:, self.time_unique])
                                    x_mask_1.append(np.array(mask_1)[:, self.time_unique])
                                    x_threshold.append(thresh)
                                    x_det_type.append('Abundance')
                                    x_new_t_min.append(new_t_min)
                                    x_new_t_max.append(new_t_max)
                                    x_t_min.append(new_t_min)
                                    x_t_max.append(new_t_max)
                                    taxa_tree.append(fca)
                                    full_tree.append(tree_copy)


                                    model_info = {
                                        'kappa': kappa,
                                        'eta': eta,
                                        'dist_from_eta': sel_dist_emb,
                                        'thresh': thresh,
                                        'dist_between_otus': sel_dist,
                                    }

                                    if self.args.save_as_csv:
                                        rule_path = '{}/rule_{}_detector_{}_abun.csv'.format(dirName, i, p)
                                        csv_df.loc[len(csv_df.index)] = [i, p, 'abundance', rule_path, otu_annot_str, kappa, t_min, t_max,\
                                        thresh, 'N/A', np.median(abun_0), np.median(abun_1), fc_wts[i], fc_bias]
                                        csv_det_df.to_csv(rule_path, index=False)

                                    if self.args.debug:
                                        kapps = list()
                                        mus = list()
                                        sigs = list()
                                        thres = list()
                                        zs = list()
                                        zrs = list()
                                        ws = list()
                                        bs = list()

                                        #grads
                                        kapps_grad = list()
                                        abun_a_grad = list()
                                        abun_b_grad = list()
                                        slope_a_grad = list()
                                        slope_b_grad = list()
                                        sigs_grad = list()
                                        thres_grad = list()
                                        zs_grad = list()
                                        zrs_grad = list()
                                        ws_grad = list()
                                        bs_grad = list()
                                        for epoch in range(self.args.epochs):
                                            best_model_init = torch.load('{}/model_epoch_{}.pth'.format(dirName, epoch))
                                            kapp = (best_model_init.spat_attn.kappas).detach().cpu().numpy()[i, p]
                                            mu = (best_model_init.time_attn.m).detach().cpu().numpy()[i, p]
                                            sig = (best_model_init.time_attn.s_abun).detach().cpu().numpy()[i, p]
                                            thre = best_model_init.thresh_func.thresh.detach().cpu().numpy()[i, p]
                                            zz = torch.sigmoid(best_model_init.rules.alpha * self.args.max_k_bc).detach().cpu().numpy()[i, p]
                                            zr = torch.sigmoid(best_model_init.fc.beta * self.args.max_k_bc).detach().cpu().numpy()[i]
                                            w = best_model_init.fc.weight.view(-1).detach().cpu().numpy()[i]
                                            b = best_model_init.fc.bias.detach().cpu().numpy()
                                            kapps.append(kapp)
                                            mus.append(mu)
                                            sigs.append(sig)
                                            thres.append(thre)
                                            zs.append(zz)
                                            zrs.append(zr)
                                            ws.append(w)
                                            bs.append(b)

                                            kapp = best_model_init.grad_dict['spat_attn.kappa'][i, p]
                                            abun_a = best_model_init.grad_dict['time_attn.abun_a'][i, p]
                                            abun_b = best_model_init.grad_dict['time_attn.abun_b'][i, p]
                                            thre = best_model_init.grad_dict['thresh_func.thresh'][i, p]
                                            zz = best_model_init.grad_dict['rules.alpha'][i, p]
                                            zr = best_model_init.grad_dict['fc.beta'][i]
                                            w = best_model_init.grad_dict['fc.weight'].view(-1)[i]
                                            b = best_model_init.grad_dict['fc.bias']
                                            kapps_grad.append(kapp)
                                            abun_a_grad.append(abun_a)
                                            abun_b_grad.append(abun_b)
                                            thres_grad.append(thre)
                                            zs_grad.append(zz)
                                            zrs_grad.append(zr)
                                            ws_grad.append(w)
                                            bs_grad.append(b)
                                        plt.figure()
                                        plt.plot(kapps)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Kappa')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(mus)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Mu')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(sigs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Sigma')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(thres)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Threshold')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Detector Selector')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zrs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Rule Selector')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(ws)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Weight')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(bs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Bias')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()

                                        plt.figure()
                                        plt.plot(kapps_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Kappa Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(abun_a_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('epsilon (abun) Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(abun_b_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Delta (abun) Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(thres_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Threshold Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zs_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Detector Selector Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zrs_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Rule Selector Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(ws_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Weight Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(bs_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Bias Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()

                                    if best_model_init is not None:
                                        kappa = kappas_init[i, p]
                                        eta = etas_init[i, p]
                                        d_e = dist_emb_init[i, p]
                                        thresh = threshs_init[i, p]
                                        mu = time_mu_init[i, p]
                                        t = time_wts_init.mean(axis=0)[i, p]
                                        t_min = -1
                                        t_max = self.num_time
                                        for b in range(len(t)):
                                            if t[t_min + 1] <= (1 / self.num_time):
                                                t_min += 1
                                            if t[t_max - 1] <= (1 / self.num_time):
                                                t_max -= 1
                                        o = otu_wts_init[i, p]
                                        sel_otu_ids = [l for l, ot in enumerate(o) if ot >= 0.9]
                                        sel_dist = np.zeros((len(sel_otu_ids), len(sel_otu_ids)))
                                        for ii in range(len(sel_otu_ids)):
                                            for jj in range(len(sel_otu_ids)):
                                                sel_dist[ii, jj] = np.linalg.norm(self.dist_emb[ii] - self.dist_emb[jj], axis=-1)
                                        sel_dist_emb = np.zeros((len(sel_otu_ids)))
                                        for ii in range(len(sel_otu_ids)):
                                            sel_dist_emb[ii] = d_e[sel_otu_ids[ii]]
                                        sel_otus = [self.var_names[self.otu_idx[l]] for l, ot in enumerate(o) if ot >= 0.5]
                                        otu_annot_str = ''
                                        if self.var_annot == {}:
                                            for n in sel_otus:
                                                otu_annot_str = otu_annot_str + '\n' + n
                                        else:
                                            sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                            for n in sel_otu_annot:
                                                otu_annot_str = otu_annot_str + '\n' + n

                                        tree = deepcopy(self.phylo_tree)
                                        sel_nodes = list()
                                        for n in tree.traverse():
                                            if n.name in sel_otus:
                                                sel_nodes.append(n)

                                        nstyle = NodeStyle()
                                        for n in tree.traverse():
                                            if n.name in sel_otus:
                                                fca = n.get_common_ancestor(sel_nodes)
                                                break
                                        if sel_nodes == []:
                                            pass

                                        for n in fca.traverse():
                                            if n.name in sel_otus:
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
                                                node_name = '{} {}'.format(n.name, node_name)
                                                text = TextFace(tw.fill(text=node_name), fsize=20)
                                                n.add_face(text, 0)
                                            else:
                                                nstyle = NodeStyle()
                                                nstyle["size"] = 0
                                                n.set_style(nstyle)

                                        ts = TreeStyle()
                                        ts.show_leaf_name = False
                                        ts.show_branch_length = True
                                        ts.branch_vertical_margin = 10
                                        ts.min_leaf_separation = 10
                                        ts.show_scale = False
                                        # fca.render('{}/r{}d{}_subtree_init_abun_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     tree_style=ts)

                                        # fig = plt.figure()
                                        # ax = fig.add_subplot()
                                        # for k in range(self.num_subjects):
                                        #     abun = x_spat_init[k, i, p]
                                        #     if labels_init[k]:
                                        #         lines_1, = ax.plot(self.times[k].astype('int'),
                                        #             abun[self.times[k].astype('int')],
                                        #             marker='+', color='g',
                                        #             linewidth=1.5, markersize=8)
                                        #     else:
                                        #         lines_0, = ax.plot(self.times[k].astype('int'),
                                        #             abun[self.times[k].astype('int')],
                                        #             marker='.', color='#FF8C00',
                                        #             linewidth=1.5, markersize=8)
                                        # ax.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                        # ax.axvline(mu)
                                        # line_thresh = ax.axhline(y=thresh,
                                        #     xmin=((t_min) / self.num_time),
                                        #     xmax=((t_max + 1) / self.num_time),
                                        #     c='k', linestyle='solid', linewidth=5)
                                        # ax.set_xlabel('Days', fontsize=20)
                                        # ax.set_ylabel('Abundance', fontsize=20)
                                        # ax.set_title('wt: %.2f bias: %.2f r: %.2f d: %.2f kappa: %.2f' % (
                                        #     fc_wts_init[i],
                                        #     fc_bias_init, r, d_slope, kappa), fontsize=10)
                                        # ax.legend([lines_0, lines_1, line_thresh],
                                        #     [self.label_0, self.label_1, 'Abundance Threshold'],
                                        #     fontsize=10, loc='upper left',
                                        #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                        # plt.savefig('{}/r{}d{}_abun_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     bbox_inches='tight')
                                        # plt.close()

                                        # fig = plt.figure()
                                        # ax = fig.add_subplot()
                                        # for k in range(self.num_subjects):
                                        #     t_abun = x_time_init[k, i, p]
                                        #     if labels_init[k]:
                                        #         lines_1, = ax.plot(k, t_abun,
                                        #             marker='+', color='r')
                                        #     else:
                                        #         lines_0, = ax.plot(k, t_abun,
                                        #             marker='.', color='g')
                                        # line_thresh = ax.axhline(y=thresh,
                                        #     c='k', linestyle='--', linewidth=5, alpha=0.5)
                                        # ax.set_title('thresh: %.4f wt: %.2f bias: %.2f r: %.2f d: %.2f' % (
                                        #     thresh,
                                        #     fc_wts_init[i],
                                        #     fc_bias_init, r, d_slope), fontsize=10)
                                        # ax.set_xlabel('Subjects', fontsize=10)
                                        # ax.legend([lines_0, lines_1, line_thresh],
                                        #     [self.label_0, self.label_1, 'Threshold'],
                                        #     fontsize=10, loc='upper left',
                                        #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                        # plt.savefig('{}/r{}d{}_abunnn_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     bbox_inches='tight')
                                        # plt.close()

                                        model_init_info = {
                                            'kappa': kappa,
                                            'eta': eta,
                                            'dist_from_eta': sel_dist_emb,
                                            'thresh': thresh,
                                        }

                                    # fig = plt.figure(figsize=(20, 18))
                                    # ax = fig.add_subplot()
                                    full_str = ''
                                    for k, v in model_info.items():
                                        if best_model_init is not None:
                                            model_init_v = model_init_info[k]
                                            model_init_info_str = 'INIT {}: {}\n'.format(k, model_init_v)
                                        else:
                                            model_init_info_str = ''
                                        model_final_info_str = 'FINAL {}: {}\n'.format(k, v)
                                        full_str = full_str + model_init_info_str + model_final_info_str + '\n'
                                    # ax.text(0, 1, full_str,
                                    #     horizontalalignment="left",
                                    #     verticalalignment="top", fontsize=12,)
                                    # ax.set_axis_off()
                                    # pdf.savefig(fig, dpi=1200)
                                    # plt.close()


                        rules_dict['taxa_tree'].append(taxa_tree)
                        rules_dict['x_0'].append(x_abun_slope_0)
                        rules_dict['x_1'].append(x_abun_slope_1)
                        rules_dict['x_mask_0'].append(x_mask_0)
                        rules_dict['x_mask_1'].append(x_mask_1)
                        rules_dict['x_avg_0'].append(x_avg_abun_slope_0)
                        rules_dict['x_avg_1'].append(x_avg_abun_slope_1)
                        rules_dict['det_type'].append(x_det_type)
                        rules_dict['thresh'].append(x_threshold)
                        rules_dict['t_min'].append(x_t_min)
                        rules_dict['t_max'].append(x_t_max)
                        rules_dict['new_t_min'].append(x_new_t_min)
                        rules_dict['new_t_max'].append(x_new_t_max)
                        rules_dict['full_tree'].append(full_tree)


            if self.args.debug:
                for epoch in range(self.args.epochs):
                    os.remove('{}/model_epoch_{}.pth'.format(dirName, epoch))

            if self.args.save_as_csv:
                csv_df.to_csv('{}/rules_dump.csv'.format(dirName), index=False)

            with open('{}/rules_dump.pickle'.format(dirName), 'wb') as f:
                pickle.dump(rules_dict, f)


    def show_rules(self, best_model, test_loader, fold, save_viz=True, best_model_init=None):
        dirName = '{}/fold_{}'.format(self.log_dir, fold)
        self.create_dir(dirName)
        torch.save(best_model.state_dict(), '{}/best_model.pth'.format(dirName))

        taxa_to_star = {
            'species': '*',
            'genus': '**',
            'family': '***',
            'order': '****',
            'class': '*****',
            'phylum': '******',
        }

        if self.args.save_as_csv:
            column_names = ['rule_id', 'detector_id', 'detector_type', 'csv_filepath', 'significant_taxa', 'otu_radius_kappa', 'time_window_low',\
            'time_window_high', 'abundance_threshold', 'slope_threshold', 'median_thresh_0', 'median_thresh_1', 'weight', 'bias']
            csv_df = pd.DataFrame(columns=column_names)

        if save_viz:
            best_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    data = batch['data'].to(self.device)
                    labels = batch['label'].to(self.device)
                    mask = batch['mask'].to(self.device)

                    x_spat = best_model.spat_attn(data, k=self.args.max_k_otu)
                    x_time, x_time_slope = best_model.time_attn(x_spat, mask=mask, k=self.args.max_k_time)
                    x_thresh = best_model.thresh_func(x_time, k=self.args.max_k_thresh)
                    x = best_model.rules(x_thresh, k=self.args.max_k_bc, hard=False)
                    x_slope = best_model.slope_func(x_time_slope, k=self.args.max_k_slope)
                    x_s = best_model.rules(x_slope, k=self.args.max_k_bc, hard=False)
                    outputs = best_model.fc(x, x_s, k=self.args.max_k_bc, hard=False)

            x_spat = x_spat.detach().cpu().numpy()
            x_time = x_time.detach().cpu().numpy()
            x_time_slope = x_time_slope.detach().cpu().numpy()
            otu_wts = best_model.spat_attn.wts.detach().cpu().numpy()
            time_mu = (best_model.time_attn.m).detach().cpu().numpy()
            time_mu_slope = (best_model.time_attn.m_slope).detach().cpu().numpy()
            time_sigma = (best_model.time_attn.s_abun).detach().cpu().numpy()
            time_sigma_slope = (best_model.time_attn.s_slope).detach().cpu().numpy()
            time_wts = best_model.time_attn.wts.detach().cpu().numpy()
            time_wts_slope = best_model.time_attn.wts_slope.detach().cpu().numpy()
            rules = best_model.fc.z.detach().cpu().numpy()
            detectors = best_model.rules.z.detach().cpu().numpy()
            detectors_slope = best_model.rules_slope.z.detach().cpu().numpy()
            threshs = best_model.thresh_func.thresh.detach().cpu().numpy()
            slopes = best_model.slope_func.slope.detach().cpu().numpy()
            # threshs = x_time.mean(axis=0)
            # slopes = x_time_slope.mean(axis=0)
            labels = labels.detach().cpu().numpy()
            fc_wts = best_model.fc.weight.view(-1).detach().cpu().numpy()
            fc_bias = best_model.fc.bias.detach().cpu().item()
            kappas = (best_model.spat_attn.kappas).detach().cpu().numpy()
            etas = (best_model.spat_attn.eta).detach().cpu().numpy()
            dist_emb = (best_model.spat_attn.emb_dist).detach().cpu().numpy()
            x_thresh = x_thresh.detach().cpu().numpy()
            x_slope = x_slope.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            rule_log_odds = fc_wts * rules + fc_bias

            rules_dict = {
                'subjects_0_log_odds': self.subjects_0_log_odds,
                'subjects_1_log_odds': self.subjects_1_log_odds,
                'sub_0_log_odds': self.sub_0_log_odds,
                'sub_1_log_odds': self.sub_1_log_odds,
                'log_odds': rule_log_odds,
                'taxa_tree': [],
                'full_tree': [],
                'exp_duration': self.num_time,
                'time_unique': self.time_unique,
                'x_0': [],
                'x_1': [],
                'x_mask_0': [],
                'x_mask_1': [],
                'x_avg_0': [],
                'x_avg_1': [],
                'det_type': [],
                'outcome_0': self.label_0,
                'outcome_1': self.label_1,
                'thresh': [],
                't_min': [],
                't_max': [],
                'new_t_min': [],
                'new_t_max': [],
            }

            if best_model_init is not None:
                torch.save(best_model_init.state_dict(), '{}/best_model_init.pth'.format(dirName))
                best_model_init.eval()
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        data_init = batch['data'].to(self.device)
                        labels_init = batch['label'].to(self.device)
                        mask_init = batch['mask'].to(self.device)

                        x_spat_init = best_model_init.spat_attn(data_init, k=self.args.max_k_otu)
                        x_time_init, x_time_slope_init = best_model_init.time_attn(x_spat_init, mask=mask_init, k=self.args.max_k_time)
                        x_thresh_init = best_model_init.thresh_func(x_time_init, k=self.args.max_k_thresh)
                        x_init = best_model_init.rules(x_thresh_init, k=self.args.max_k_bc, hard=False)
                        x_slope_init = best_model_init.slope_func(x_time_slope_init, k=self.args.max_k_slope)
                        x_slope_init = best_model_init.rules(x_slope_init, k=self.args.max_k_bc, hard=False)
                        outputs_init = best_model_init.fc(x_init, x_slope_init, k=self.args.max_k_bc, hard=False)

                x_spat_init = x_spat_init.detach().cpu().numpy()
                x_time_init = x_time_init.detach().cpu().numpy()
                x_time_slope_init = x_time_slope_init.detach().cpu().numpy()
                otu_wts_init = best_model_init.spat_attn.wts.detach().cpu().numpy()
                time_mu_init = (best_model_init.time_attn.m).detach().cpu().numpy()
                time_mu_slope_init = (best_model_init.time_attn.m_slope).detach().cpu().numpy()
                time_wts_init = best_model_init.time_attn.wts.detach().cpu().numpy()
                time_wts_slope_init = best_model_init.time_attn.wts_slope.detach().cpu().numpy()
                threshs_init = best_model_init.thresh_func.thresh.detach().cpu().numpy()
                slopes_init = best_model_init.slope_func.slope.detach().cpu().numpy()
                labels_init = labels_init.detach().cpu().numpy()
                fc_wts_init = best_model_init.fc.weight.view(-1).detach().cpu().numpy()
                fc_bias_init = best_model_init.fc.bias.detach().cpu().item()
                kappas_init = (best_model_init.spat_attn.kappas).detach().cpu().numpy()
                etas_init = (best_model_init.spat_attn.eta).detach().cpu().numpy()
                dist_emb_init = (best_model_init.spat_attn.emb_dist).detach().cpu().numpy()


            # Create the PdfPages object to which we will save the pages:
            # The with statement makes sure that the PdfPages object is closed properly at
            # the end of the block, even if an Exception occurs.
            with PdfPages('{}/rules.pdf'.format(dirName)) as pdf:
                num_rules = 0
                num_det_per_rule = 0
                for i, r in enumerate(rules):
                    has_det = False
                    if r >= 0.9:
                        for p in range(self.num_detectors):
                            d = detectors[i, p]
                            d_slope = detectors_slope[i, p]
                            if d >= 0.9 or d_slope >=0.9:
                                has_det = True
                            if d >= 0.9:
                                num_det_per_rule += 1
                            if d_slope >= 0.9:
                                num_det_per_rule += 1
                        if has_det:
                            num_rules += 1

                if num_rules == 0:
                    num_rules = 1
                num_det_per_rule = num_det_per_rule / num_rules
                # fig = plt.figure(figsize=(20, 18))
                # ax = fig.add_subplot()
                # rule_str = 'Number of Active rules (selection wt > 0.9): {}'.format(num_rules)
                # det_str = 'Number of Active detectors per active rule (selection wt > 0.9): {:.2f}'.format(num_det_per_rule)
                # full_str = rule_str + '\n' + det_str
                # ax.text(0, 1, full_str,
                #     horizontalalignment="left",
                #     verticalalignment="top", fontsize=12,)
                # ax.set_axis_off()
                # pdf.savefig(fig, dpi=1200)
                # plt.close()

                for i, r in enumerate(rules):
                    if r >= 0.:
                        taxa_tree = list()
                        full_tree = list()
                        x_abun_slope_0 = list()
                        x_avg_abun_slope_0 = list()
                        x_abun_slope_1 = list()
                        x_avg_abun_slope_1 = list()
                        x_mask_0 = list()
                        x_mask_1 = list()
                        x_det_type = list()
                        x_threshold = list()
                        x_new_t_min = list()
                        x_new_t_max = list()
                        x_t_min = list()
                        x_t_max = list()
                        for p in range(self.num_detectors):
                            d = detectors[i, p]
                            d_slope = detectors_slope[i, p]
                            if d >= 0.9:
                                kappa = kappas[i, p]
                                eta = etas[i, p]
                                d_e = dist_emb[i, p]
                                thresh = threshs[i, p]
                                mu = time_mu[i, p]
                                t = time_wts.mean(axis=0)[i, p]
                                t_min = int(np.floor(time_mu[i, p] - (time_sigma[i, p] // 2)))
                                t_max = int(np.ceil(time_mu[i, p] + (time_sigma[i, p] // 2)))
                                o = otu_wts[i, p]
                                sel_otu_ids = [l for l, ot in enumerate(o) if ot >= 0.9]
                                sel_dist = np.zeros((len(sel_otu_ids), len(sel_otu_ids)))
                                for ii in range(len(sel_otu_ids)):
                                    for jj in range(len(sel_otu_ids)):
                                        sel_dist[ii, jj] = self.dist_matrix_embed[sel_otu_ids[ii], sel_otu_ids[jj]]
                                sel_dist_emb = np.zeros((len(sel_otu_ids)))
                                for ii in range(len(sel_otu_ids)):
                                    sel_dist_emb[ii] = d_e[sel_otu_ids[ii]]
                                sel_otus = [self.var_ids[l] for l, ot in enumerate(o) if ot >= 0.9]
                                otu_annot_str = ''
                                if self.var_annot == {}:
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + '\n' + n
                                else:
                                    # sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + ',' + n

                                tree = deepcopy(self.phylo_tree)
                                sel_nodes = list()
                                for n in tree.traverse():
                                    if n.name in sel_otus:
                                        sel_nodes.append(n)
                                if sel_nodes == []:
                                    print(np.sort(o))
                                    print(kappa)
                                else:
                                    nstyle = NodeStyle()
                                    fca = None
                                    for n in tree.traverse():
                                        if n.name in sel_otus:
                                            fca = n.get_common_ancestor(sel_nodes)
                                            break
                                    # for n in fca.traverse():
                                    #     nstyle = NodeStyle()
                                    #     nstyle["size"] = 0
                                    #     if n.name in sel_otus:
                                    #         nstyle["vt_line_color"] = "red"
                                    #         nstyle["hz_line_color"] = "red"
                                    #         nstyle["vt_line_width"] = 2
                                    #         nstyle["hz_line_width"] = 2
                                    #         nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                    #         nstyle["hz_line_type"] = 0
                                    #     n.set_style(nstyle)

                                    # ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    # ts.show_branch_length = False
                                    # ts.branch_vertical_margin = 4
                                    # ts.min_leaf_separation = 4
                                    # ts.show_scale = False
                                    # fca.render('{}/r{}d{}_tree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)
                                    # fca_full_copy = deepcopy(fca).up
                                    
                                    # for n in fca.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         tw = textwrap.TextWrapper(width=30)
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         text = TextFace(tw.fill(text=node_name), fsize=24)
                                    #         n.add_face(text, 0)
                                    #     else:
                                    #         n.delete()

                                    # ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    # ts.show_branch_length = False
                                    # ts.branch_vertical_margin = 10
                                    # ts.min_leaf_separation = 10
                                    # ts.show_scale = False
                                    # fca.prune(sel_nodes)
                                    # # fca.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    # #     dirName, i, p, self.args.data_name), dpi=1200,
                                    # #     tree_style=ts)

                                    # fca_copy = deepcopy(fca)
                                    # for n in fca_copy.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # fca_copy.write(outfile="{}/r{}d{}_subtree_abun.newick".format(
                                    #     dirName, i, p))
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)

                                    # for n in fca_full_copy.traverse():
                                    #     nstyle = NodeStyle()
                                    #     nstyle["size"] = 0
                                    #     if n.name in sel_otus:
                                    #         nstyle["vt_line_color"] = "red"
                                    #         nstyle["hz_line_color"] = "red"
                                    #         nstyle["vt_line_width"] = 2
                                    #         nstyle["hz_line_width"] = 2
                                    #         nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                    #         nstyle["hz_line_type"] = 0
                                    #     n.set_style(nstyle)

                                    #     if n.is_leaf():
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name

                                    fca_full_copy = deepcopy(fca)
                                    fca_copy = deepcopy(fca)
                                    
                                    for n in fca.get_leaves():
                                        if n.name in sel_otus:
                                            # tw = textwrap.TextWrapper(width=30)
                                            # if self.var_annot == {}:
                                            #     nn_split = n.name.split('|')[-1]
                                            #     nn_split = nn_split.split('__')
                                            #     for k, v in taxa_to_star.items():
                                            #         if k.lower()[0] == nn_split[0].lower():
                                            #             taxa_name = v
                                            #     node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                            # else:
                                            #     node_name = self.var_annot.get(n.name, '(no annotation)')
                                            #     node_name = node_name.split(" ")
                                            #     remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                            #     new_node_name = ''
                                            #     for l in node_name:
                                            #         if not l in remove_list:
                                            #             new_node_name = new_node_name + l + ' '
                                            #     node_name = new_node_name
                                            #     if ',including' in node_name:
                                            #         node_name = node_name.replace(',including', '').split(" ")[:2]
                                            #         node_name = ' '.join(map(str, node_name))
                                            #     node_name = '{}'.format(node_name)
                                            #     n_split = node_name.split(' ')
                                            #     new_name = ''
                                            #     taxa_name = ''
                                            #     for nn in n_split:
                                            #         if nn.lower() in taxa_to_star.keys():
                                            #             taxa_name = taxa_to_star[nn.lower()]
                                            #         else:
                                            #             if '/' in nn:
                                            #                 nn = nn.split('/')[0]
                                            #             new_name = new_name + nn + ' '
                                            #     if taxa_name == '':
                                            #         taxa_name = taxa_to_star['species']
                                            #     node_name = '{} {}'.format(taxa_name, new_name)
                                            # text = TextFace(tw.fill(text=node_name), fsize=24)
                                            # n.add_face(text, 0)

                                            n_idx = sel_otu_ids[sel_otus.index(n.name)]
                                            text = 'OTU{}'.format(n_idx)
                                            n.name = text
                                        else:
                                            n.delete()

                                    ts = TreeStyle()
                                    ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 10
                                    ts.min_leaf_separation = 10
                                    ts.show_scale = False
                                    fca.prune(sel_nodes)
                                    # fca.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)

                                    # for n in fca_copy.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # for ll, n in enumerate(fca_copy.get_leaves()):
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # fca_copy.write(outfile="{}/r{}d{}_subtree_abun.newick".format(
                                    #     dirName, i, p))
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)

                                    while True:
                                        if len(fca_full_copy.get_leaves()) > len(sel_otus):
                                            break
                                        else:
                                            fca_full_copy = fca_full_copy.up

                                    tree_copy = deepcopy(self.phylo_tree)
                                    for n in tree_copy.traverse():
                                        nstyle = NodeStyle()
                                        nstyle["size"] = 0
                                        if n.name in sel_otus:
                                            nstyle["vt_line_color"] = "red"
                                            nstyle["hz_line_color"] = "red"
                                            nstyle["vt_line_width"] = 2
                                            nstyle["hz_line_width"] = 2
                                            nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                            nstyle["hz_line_type"] = 0
                                        n.set_style(nstyle)

                                        if n.is_leaf():
                                            n_idx = self.var_ids.index(n.name)
                                            if self.var_annot == {}:
                                                nn_split = n.name.split('|')[-1]
                                                nn_split = nn_split.split('__')
                                                for k, v in taxa_to_star.items():
                                                    if k.lower()[0] == nn_split[0].lower():
                                                        taxa_name = v
                                                node_name = 'OTU{} {} {}'.format(n_idx, taxa_name, ' '.join(nn_split[1].split('_')))
                                            else:
                                                node_name = self.var_annot.get(n.name, '(no annotation)')
                                                node_name = node_name.split(" ")
                                                remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                                new_node_name = ''
                                                for l in node_name:
                                                    if not l in remove_list:
                                                        new_node_name = new_node_name + l + ' '
                                                node_name = new_node_name
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{}'.format(node_name)
                                                n_split = node_name.split(' ')
                                                new_name = ''
                                                taxa_name = ''
                                                for nn in n_split:
                                                    if nn.lower() in taxa_to_star.keys():
                                                        taxa_name = taxa_to_star[nn.lower()]
                                                    else:
                                                        if '/' in nn:
                                                            nn = nn.split('/')[0]
                                                        new_name = new_name + nn + ' '
                                                if taxa_name == '':
                                                    taxa_name = taxa_to_star['species']
                                                node_name = 'OTU{} {} {}'.format(n_idx, taxa_name, new_name)
                                            n.name = node_name

                                    for n in fca_full_copy.traverse():
                                        nstyle = NodeStyle()
                                        nstyle["size"] = 0
                                        if n.name in sel_otus:
                                            nstyle["vt_line_color"] = "red"
                                            nstyle["hz_line_color"] = "red"
                                            nstyle["vt_line_width"] = 2
                                            nstyle["hz_line_width"] = 2
                                            nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                            nstyle["hz_line_type"] = 0
                                        n.set_style(nstyle)

                                        if n.is_leaf():
                                            if self.var_annot == {}:
                                                nn_split = n.name.split('|')[-1]
                                                nn_split = nn_split.split('__')
                                                for k, v in taxa_to_star.items():
                                                    if k.lower()[0] == nn_split[0].lower():
                                                        taxa_name = v
                                                node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                            else:
                                                node_name = self.var_annot.get(n.name, '(no annotation)')
                                                node_name = node_name.split(" ")
                                                remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                                new_node_name = ''
                                                for l in node_name:
                                                    if not l in remove_list:
                                                        new_node_name = new_node_name + l + ' '
                                                node_name = new_node_name
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{}'.format(node_name)
                                                n_split = node_name.split(' ')
                                                new_name = ''
                                                taxa_name = ''
                                                for nn in n_split:
                                                    if nn.lower() in taxa_to_star.keys():
                                                        taxa_name = taxa_to_star[nn.lower()]
                                                    else:
                                                        if '/' in nn:
                                                            nn = nn.split('/')[0]
                                                        new_name = new_name + nn + ' '
                                                if taxa_name == '':
                                                    taxa_name = taxa_to_star['species']
                                                node_name = '{} {}'.format(taxa_name, new_name)
                                            n.name = node_name

                                    ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 1
                                    ts.min_leaf_separation = 4
                                    ts.show_scale = False
                                    # fca_full_copy.render('{}/r{}d{}_tree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)
                                    


                                    # fig = plt.figure()
                                    # fig.subplots_adjust(top=0.8)
                                    # ax = fig.add_subplot(121)
                                    abun_0 = list()
                                    abun_1 = list()
                                    for k in range(self.num_subjects):
                                        t_abun = x_time[k, i, p]
                                        if labels[k]:
                                            # lines_1, = ax.plot(k, t_abun,
                                            #     marker='+', color='g')
                                            abun_1.append(t_abun)
                                        else:
                                            # lines_0, = ax.plot(k, t_abun,
                                            #     marker='.', color='#FF8C00')
                                            abun_0.append(t_abun)
                                    # line_thresh = ax.axhline(y=thresh,
                                    #     c='k', linestyle='--')
                                    # ax.set_xlabel('Subjects')
                                    # ax.set_ylabel('Abundance')
                                    # ax.legend([lines_0, lines_1, line_thresh],
                                    #     [self.label_0, self.label_1, 'Threshold'],
                                    #     loc='upper left',
                                    #     fancybox=True, framealpha=0.5,
                                    #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                    log_odds = fc_wts[i] * r + fc_bias
                                    rule_eng = self.get_rule(fc_wts[i], log_odds,
                                        i, p, t_min, t_max, thresh, metric='abundance')
                                    # plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    # tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name))
                                    # ax1 = fig.add_subplot(122)
                                    # imgplot = plt.imshow(tree_img)
                                    # ax1.set_axis_off()
                                    # plt.savefig('{}/r{}d{}_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200)
                                    # plt.close()

                                    if self.args.save_as_csv:
                                        column_names = ['Subject', 'Label', *['day_{}'.format(day) for day in np.arange(self.num_time)]]
                                        csv_det_df = pd.DataFrame(columns=column_names)
                                    # fig = plt.figure(figsize=(20, 18))
                                    # fig.subplots_adjust(top=0.8)
                                    # gs = fig.add_gridspec(2, 2)
                                    # f_ax1 = fig.add_subplot(gs[0, 0])
                                    # f_ax2 = fig.add_subplot(gs[1, 0], sharex=f_ax1)
                                    abundances_0 = list()
                                    abundances_1 = list()
                                    mask_0 = list()
                                    mask_1 = list()
                                    for k in range(self.num_subjects):
                                        abun = x_spat[k, i, p]
                                        if labels[k]:
                                            # lines_1, = f_ax1.plot(self.times[k].astype('int'),
                                            #     abun[self.times[k].astype('int')],
                                            #     marker='.', color='g')
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            abundances_1.append(abun)
                                            mask_1.append(mask[k])
                                        else:
                                            # lines_0, = f_ax2.plot(self.times[k].astype('int'),
                                            #     abun[self.times[k].astype('int')],
                                            #     marker='.', color='#FF8C00')
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            abundances_0.append(abun)
                                            mask_0.append(mask[k])
                                        if self.args.save_as_csv:
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            csv_det_df.loc[len(csv_det_df.index)] = [k, labels[k], *abun]
                                    # f_ax1.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    # sel_days = np.arange(t_min, t_max + 1)
                                    # line_thresh = f_ax1.axhline(y=thresh,
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='k', linestyle='--', linewidth=3)
                                    # line_thresh_1 = f_ax1.axhline(y=np.median(abun_1),
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='r', linestyle='--', linewidth=3)
                                    # f_ax1.set_ylabel('Relative Abundance')
                                    # f_ax1.set_title(self.label_1)
                                    # plt.setp(f_ax1.get_xticklabels(), visible=False)
                                    # f_ax2.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    # line_thresh = f_ax2.axhline(y=thresh,
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='k', linestyle='--', linewidth=3)
                                    # line_thresh_0 = f_ax2.axhline(y=np.median(abun_0),
                                    #     xmin=((t_min) / self.num_time),
                                    #     xmax=((t_max + 1) / self.num_time),
                                    #     c='r', linestyle='--', linewidth=3)
                                    # f_ax2.set_xlabel('Days')
                                    # f_ax2.set_ylabel('Relative Abundance')
                                    # f_ax2.set_title(self.label_0)
                                    log_odds = fc_wts[i] * r + fc_bias
                                    rule_eng = self.get_rule(fc_wts[i], log_odds, i,
                                        p, t_min, t_max, thresh, metric='abundance')
                                    # plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    # tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name))
                                    # f_ax3 = fig.add_subplot(gs[:, 1])
                                    # imgplot = plt.imshow(tree_img)
                                    # f_ax3.set_axis_off()
                                    # pdf.savefig(bbox_inches='tight', dpi=1200)
                                    # plt.savefig('{}/r{}d{}_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200)
                                    # plt.close()

                                    labels_0 = np.array([int(lab) for lab in labels if lab == 0])
                                    labels_1 = np.array([int(lab) for lab in labels if lab == 1])
                                    new_t_min = min([t for t in self.time_unique if t >= t_min])
                                    new_t_max = max([t for t in self.time_unique if t <= t_max])
                                    rule_path = '{}/r{}d{}_heatmap_abun_{}.pdf'.format(
                                        dirName, i, p, self.args.data_name)
                                    new_t = np.array([t for t in self.time_unique if t >= t_min and t <= t_max])
                                    x_0 = np.array(abundances_0)[:, self.time_unique]
                                    x_1 = np.array(abundances_1)[:, self.time_unique]
                                    x_0[x_0 == -1] = np.nan
                                    x_1[x_1 == -1] = np.nan
                                    # self.heatmap_rule_viz(x_0, x_1,
                                    #     np.array(abun_0), np.array(abun_1), np.array(mask_0)[:, self.time_unique],
                                    #     np.array(mask_1)[:, self.time_unique], self.time_unique,
                                    #     fca_full_copy, thresh, 'Abun', new_t_min, new_t_max,
                                    #     rule_eng, rule_path, t_min, t_max, pdf=pdf)

                                    x_abun_slope_0.append(x_0)
                                    x_abun_slope_1.append(x_1)
                                    x_avg_abun_slope_0.append(np.array(abun_0))
                                    x_avg_abun_slope_1.append(np.array(abun_1))
                                    x_mask_0.append(np.array(mask_0)[:, self.time_unique])
                                    x_mask_1.append(np.array(mask_1)[:, self.time_unique])
                                    x_threshold.append(thresh)
                                    x_det_type.append('Abundance')
                                    x_new_t_min.append(new_t_min)
                                    x_new_t_max.append(new_t_max)
                                    x_t_min.append(new_t_min)
                                    x_t_max.append(new_t_max)
                                    taxa_tree.append(fca)
                                    full_tree.append(tree_copy)

                                    model_info = {
                                        'kappa': kappa,
                                        'eta': eta,
                                        'dist_from_eta': sel_dist_emb,
                                        'thresh': thresh,
                                        'dist_between_otus': sel_dist,
                                    }

                                    if self.args.save_as_csv:
                                        rule_path = '{}/rule_{}_detector_{}_abun.csv'.format(dirName, i, p)
                                        csv_df.loc[len(csv_df.index)] = [i, p, 'abundance', rule_path, otu_annot_str, kappa, t_min, t_max,\
                                        thresh, 'N/A', np.median(abun_0), np.median(abun_1), fc_wts[i], fc_bias]
                                        csv_det_df.to_csv(rule_path, index=False)

                                    if self.args.debug:
                                        kapps = list()
                                        mus = list()
                                        sigs = list()
                                        thres = list()
                                        zs = list()
                                        zrs = list()
                                        ws = list()
                                        bs = list()

                                        #grads
                                        kapps_grad = list()
                                        abun_a_grad = list()
                                        abun_b_grad = list()
                                        slope_a_grad = list()
                                        slope_b_grad = list()
                                        sigs_grad = list()
                                        thres_grad = list()
                                        zs_grad = list()
                                        zrs_grad = list()
                                        ws_grad = list()
                                        bs_grad = list()
                                        for epoch in range(self.args.epochs):
                                            best_model_init = torch.load('{}/model_epoch_{}.pth'.format(dirName, epoch))
                                            kapp = (best_model_init.spat_attn.kappas).detach().cpu().numpy()[i, p]
                                            mu = (best_model_init.time_attn.m).detach().cpu().numpy()[i, p]
                                            sig = (best_model_init.time_attn.s_abun).detach().cpu().numpy()[i, p]
                                            thre = best_model_init.thresh_func.thresh.detach().cpu().numpy()[i, p]
                                            zz = torch.sigmoid(best_model_init.rules.alpha * self.args.max_k_bc).detach().cpu().numpy()[i, p]
                                            zr = torch.sigmoid(best_model_init.fc.beta * self.args.max_k_bc).detach().cpu().numpy()[i]
                                            w = best_model_init.fc.weight.view(-1).detach().cpu().numpy()[i]
                                            b = best_model_init.fc.bias.detach().cpu().numpy()
                                            kapps.append(kapp)
                                            mus.append(mu)
                                            sigs.append(sig)
                                            thres.append(thre)
                                            zs.append(zz)
                                            zrs.append(zr)
                                            ws.append(w)
                                            bs.append(b)

                                            kapp = best_model_init.grad_dict['spat_attn.kappa'][i, p]
                                            abun_a = best_model_init.grad_dict['time_attn.abun_a'][i, p]
                                            abun_b = best_model_init.grad_dict['time_attn.abun_b'][i, p]
                                            thre = best_model_init.grad_dict['thresh_func.thresh'][i, p]
                                            zz = best_model_init.grad_dict['rules.alpha'][i, p]
                                            zr = best_model_init.grad_dict['fc.beta'][i]
                                            w = best_model_init.grad_dict['fc.weight'].view(-1)[i]
                                            b = best_model_init.grad_dict['fc.bias']
                                            kapps_grad.append(kapp)
                                            abun_a_grad.append(abun_a)
                                            abun_b_grad.append(abun_b)
                                            thres_grad.append(thre)
                                            zs_grad.append(zz)
                                            zrs_grad.append(zr)
                                            ws_grad.append(w)
                                            bs_grad.append(b)
                                        plt.figure()
                                        plt.plot(kapps)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Kappa')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(mus)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Mu')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(sigs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Sigma')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(thres)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Threshold')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Detector Selector')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zrs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Rule Selector')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(ws)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Weight')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(bs)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Bias')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()

                                        plt.figure()
                                        plt.plot(kapps_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Kappa Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(abun_a_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('epsilon (abun) Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(abun_b_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Delta (abun) Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(thres_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Threshold Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zs_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Detector Selector Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(zrs_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Rule Selector Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(ws_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Weight Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        plt.figure()
                                        plt.plot(bs_grad)
                                        plt.xlabel('Training Epochs')
                                        plt.ylabel('Bias Gradients')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()

                                    if best_model_init is not None:
                                        kappa = kappas_init[i, p]
                                        eta = etas_init[i, p]
                                        d_e = dist_emb_init[i, p]
                                        thresh = threshs_init[i, p]
                                        mu = time_mu_init[i, p]
                                        t = time_wts_init.mean(axis=0)[i, p]
                                        t_min = -1
                                        t_max = self.num_time
                                        for b in range(len(t)):
                                            if t[t_min + 1] <= (1 / self.num_time):
                                                t_min += 1
                                            if t[t_max - 1] <= (1 / self.num_time):
                                                t_max -= 1
                                        o = otu_wts_init[i, p]
                                        sel_otu_ids = [l for l, ot in enumerate(o) if ot >= 0.9]
                                        sel_dist = np.zeros((len(sel_otu_ids), len(sel_otu_ids)))
                                        for ii in range(len(sel_otu_ids)):
                                            for jj in range(len(sel_otu_ids)):
                                                sel_dist[ii, jj] = np.linalg.norm(self.dist_emb[ii] - self.dist_emb[jj], axis=-1)
                                        sel_dist_emb = np.zeros((len(sel_otu_ids)))
                                        for ii in range(len(sel_otu_ids)):
                                            sel_dist_emb[ii] = d_e[sel_otu_ids[ii]]
                                        sel_otus = [self.var_names[self.otu_idx[l]] for l, ot in enumerate(o) if ot >= 0.5]
                                        otu_annot_str = ''
                                        if self.var_annot == {}:
                                            for n in sel_otus:
                                                otu_annot_str = otu_annot_str + '\n' + n
                                        else:
                                            sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                            for n in sel_otu_annot:
                                                otu_annot_str = otu_annot_str + '\n' + n

                                        tree = deepcopy(self.phylo_tree)
                                        sel_nodes = list()
                                        for n in tree.traverse():
                                            if n.name in sel_otus:
                                                sel_nodes.append(n)

                                        nstyle = NodeStyle()
                                        for n in tree.traverse():
                                            if n.name in sel_otus:
                                                fca = n.get_common_ancestor(sel_nodes)
                                                break
                                        if sel_nodes == []:
                                            pass

                                        for n in fca.traverse():
                                            if n.name in sel_otus:
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
                                                node_name = '{} {}'.format(n.name, node_name)
                                                text = TextFace(tw.fill(text=node_name), fsize=20)
                                                n.add_face(text, 0)
                                            else:
                                                nstyle = NodeStyle()
                                                nstyle["size"] = 0
                                                n.set_style(nstyle)

                                        ts = TreeStyle()
                                        ts.show_leaf_name = False
                                        ts.show_branch_length = True
                                        ts.branch_vertical_margin = 10
                                        ts.min_leaf_separation = 10
                                        ts.show_scale = False
                                        # fca.render('{}/r{}d{}_subtree_init_abun_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     tree_style=ts)

                                        # fig = plt.figure()
                                        # ax = fig.add_subplot()
                                        # for k in range(self.num_subjects):
                                        #     abun = x_spat_init[k, i, p]
                                        #     if labels_init[k]:
                                        #         lines_1, = ax.plot(self.times[k].astype('int'),
                                        #             abun[self.times[k].astype('int')],
                                        #             marker='+', color='g',
                                        #             linewidth=1.5, markersize=8)
                                        #     else:
                                        #         lines_0, = ax.plot(self.times[k].astype('int'),
                                        #             abun[self.times[k].astype('int')],
                                        #             marker='.', color='#FF8C00',
                                        #             linewidth=1.5, markersize=8)
                                        # ax.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                        # ax.axvline(mu)
                                        # line_thresh = ax.axhline(y=thresh,
                                        #     xmin=((t_min) / self.num_time),
                                        #     xmax=((t_max + 1) / self.num_time),
                                        #     c='k', linestyle='solid', linewidth=5)
                                        # ax.set_xlabel('Days', fontsize=20)
                                        # ax.set_ylabel('Abundance', fontsize=20)
                                        # ax.set_title('wt: %.2f bias: %.2f r: %.2f d: %.2f kappa: %.2f' % (
                                        #     fc_wts_init[i],
                                        #     fc_bias_init, r, d_slope, kappa), fontsize=10)
                                        # ax.legend([lines_0, lines_1, line_thresh],
                                        #     [self.label_0, self.label_1, 'Abundance Threshold'],
                                        #     fontsize=10, loc='upper left',
                                        #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                        # plt.savefig('{}/r{}d{}_abun_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     bbox_inches='tight')
                                        # plt.close()

                                        # fig = plt.figure()
                                        # ax = fig.add_subplot()
                                        # for k in range(self.num_subjects):
                                        #     t_abun = x_time_init[k, i, p]
                                        #     if labels_init[k]:
                                        #         lines_1, = ax.plot(k, t_abun,
                                        #             marker='+', color='r')
                                        #     else:
                                        #         lines_0, = ax.plot(k, t_abun,
                                        #             marker='.', color='g')
                                        # line_thresh = ax.axhline(y=thresh,
                                        #     c='k', linestyle='--', linewidth=5, alpha=0.5)
                                        # ax.set_title('thresh: %.4f wt: %.2f bias: %.2f r: %.2f d: %.2f' % (
                                        #     thresh,
                                        #     fc_wts_init[i],
                                        #     fc_bias_init, r, d_slope), fontsize=10)
                                        # ax.set_xlabel('Subjects', fontsize=10)
                                        # ax.legend([lines_0, lines_1, line_thresh],
                                        #     [self.label_0, self.label_1, 'Threshold'],
                                        #     fontsize=10, loc='upper left',
                                        #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                        # plt.savefig('{}/r{}d{}_abunnn_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     bbox_inches='tight')
                                        # plt.close()

                                        model_init_info = {
                                            'kappa': kappa,
                                            'eta': eta,
                                            'dist_from_eta': sel_dist_emb,
                                            'thresh': thresh,
                                        }

                                    # fig = plt.figure(figsize=(20, 18))
                                    # ax = fig.add_subplot()
                                    full_str = ''
                                    for k, v in model_info.items():
                                        if best_model_init is not None:
                                            model_init_v = model_init_info[k]
                                            model_init_info_str = 'INIT {}: {}\n'.format(k, model_init_v)
                                        else:
                                            model_init_info_str = ''
                                        model_final_info_str = 'FINAL {}: {}\n'.format(k, v)
                                        full_str = full_str + model_init_info_str + model_final_info_str + '\n'
                                    # ax.text(0, 1, full_str,
                                    #     horizontalalignment="left",
                                    #     verticalalignment="top", fontsize=12,)
                                    # ax.set_axis_off()
                                    # pdf.savefig(fig, dpi=1200)
                                    # plt.close()


                            if d_slope >= 0.9:
                                kappa = kappas[i, p]
                                eta = etas[i, p]
                                d_e = dist_emb[i, p]
                                thresh = slopes[i, p]
                                mu = time_mu_slope[i, p]
                                t = time_wts_slope.mean(axis=0)[i, p]
                                t_min = int(np.floor(time_mu[i, p] - (time_sigma[i, p] // 2)))
                                t_max = int(np.ceil(time_mu[i, p] + (time_sigma[i, p] // 2)))
                                o = otu_wts[i, p]
                                sel_otu_ids = [l for l, ot in enumerate(o) if ot >= 0.9]
                                sel_dist = np.zeros((len(sel_otu_ids), len(sel_otu_ids)))
                                for ii in range(len(sel_otu_ids)):
                                    for jj in range(len(sel_otu_ids)):
                                        sel_dist[ii, jj] = self.dist_matrix_embed[sel_otu_ids[ii], sel_otu_ids[jj]]
                                sel_dist_emb = np.zeros((len(sel_otu_ids)))
                                for ii in range(len(sel_otu_ids)):
                                    sel_dist_emb[ii] = d_e[sel_otu_ids[ii]]
                                sel_otus = [self.var_ids[l] for l, ot in enumerate(o) if ot >= 0.9]
                                otu_annot_str = ''
                                if self.var_annot == {}:
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + '\n' + n
                                else:
                                    # sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + '\n' + n

                                tree = deepcopy(self.phylo_tree)
                                sel_nodes = list()
                                for n in tree.traverse():
                                    if n.name in sel_otus:
                                        sel_nodes.append(n)
                                if sel_nodes == []:
                                    print(np.sort(o))
                                    print(kappa)
                                else:
                                    nstyle = NodeStyle()
                                    fca = None
                                    for n in tree.traverse():
                                        if n.name in sel_otus:
                                            fca = n.get_common_ancestor(sel_nodes)
                                            break
                                    # # for n in fca.traverse():
                                    # #     nstyle = NodeStyle()
                                    # #     nstyle["size"] = 0
                                    # #     if n.name in sel_otus:
                                    # #         nstyle["vt_line_color"] = "red"
                                    # #         nstyle["hz_line_color"] = "red"
                                    # #         nstyle["vt_line_width"] = 2
                                    # #         nstyle["hz_line_width"] = 2
                                    # #         nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                    # #         nstyle["hz_line_type"] = 0
                                    # #     n.set_style(nstyle)

                                    # # ts = TreeStyle()
                                    # # ts.show_leaf_name = False
                                    # # ts.show_branch_length = False
                                    # # ts.branch_vertical_margin = 4
                                    # # ts.min_leaf_separation = 4
                                    # # ts.show_scale = False
                                    # # fca.render('{}/r{}d{}_tree_ete3_abun_{}.pdf'.format(
                                    # #     dirName, i, p, self.args.data_name),
                                    # #     tree_style=ts)
                                    # fca_full_copy = deepcopy(fca)
                                    
                                    # for n in fca.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         tw = textwrap.TextWrapper(width=30)
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         text = TextFace(tw.fill(text=node_name), fsize=24)
                                    #         n.add_face(text, 0)
                                    #     else:
                                    #         n.delete()

                                    # ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    # ts.show_branch_length = False
                                    # ts.branch_vertical_margin = 10
                                    # ts.min_leaf_separation = 10
                                    # ts.show_scale = False
                                    # fca.prune(sel_nodes)
                                    # # fca.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    # #     dirName, i, p, self.args.data_name), dpi=1200,
                                    # #     tree_style=ts)

                                    # fca_copy = deepcopy(fca)
                                    # for n in fca_copy.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # fca_copy.write(outfile="{}/r{}d{}_subtree_slope.newick".format(
                                    #     dirName, i, p))
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_slope_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_slope_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)

                                    # for n in fca_full_copy.traverse():
                                    #     nstyle = NodeStyle()
                                    #     nstyle["size"] = 0
                                    #     if n.name in sel_otus:
                                    #         nstyle["vt_line_color"] = "red"
                                    #         nstyle["hz_line_color"] = "red"
                                    #         nstyle["vt_line_width"] = 2
                                    #         nstyle["hz_line_width"] = 2
                                    #         nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                    #         nstyle["hz_line_type"] = 0
                                    #     n.set_style(nstyle)

                                    #     if n.is_leaf():
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name

                                    fca_full_copy = deepcopy(fca)
                                    fca_copy = deepcopy(fca)
                                    
                                    for n in fca.get_leaves():
                                        if n.name in sel_otus:
                                            # tw = textwrap.TextWrapper(width=30)
                                            # if self.var_annot == {}:
                                            #     nn_split = n.name.split('|')[-1]
                                            #     nn_split = nn_split.split('__')
                                            #     for k, v in taxa_to_star.items():
                                            #         if k.lower()[0] == nn_split[0].lower():
                                            #             taxa_name = v
                                            #     node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                            # else:
                                            #     node_name = self.var_annot.get(n.name, '(no annotation)')
                                            #     node_name = node_name.split(" ")
                                            #     remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                            #     new_node_name = ''
                                            #     for l in node_name:
                                            #         if not l in remove_list:
                                            #             new_node_name = new_node_name + l + ' '
                                            #     node_name = new_node_name
                                            #     if ',including' in node_name:
                                            #         node_name = node_name.replace(',including', '').split(" ")[:2]
                                            #         node_name = ' '.join(map(str, node_name))
                                            #     node_name = '{}'.format(node_name)
                                            #     n_split = node_name.split(' ')
                                            #     new_name = ''
                                            #     taxa_name = ''
                                            #     for nn in n_split:
                                            #         if nn.lower() in taxa_to_star.keys():
                                            #             taxa_name = taxa_to_star[nn.lower()]
                                            #         else:
                                            #             if '/' in nn:
                                            #                 nn = nn.split('/')[0]
                                            #             new_name = new_name + nn + ' '
                                            #     if taxa_name == '':
                                            #         taxa_name = taxa_to_star['species']
                                            #     node_name = '{} {}'.format(taxa_name, new_name)
                                            # text = TextFace(tw.fill(text=node_name), fsize=24)
                                            # n.add_face(text, 0)

                                            n_idx = sel_otu_ids[sel_otus.index(n.name)]
                                            text = 'OTU{}'.format(n_idx)
                                            n.name = text
                                        else:
                                            n.delete()

                                    ts = TreeStyle()
                                    ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 10
                                    ts.min_leaf_separation = 10
                                    ts.show_scale = False
                                    fca.prune(sel_nodes)
                                    # fca.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)

                                    # for n in fca_copy.get_leaves():
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # for ll, n in enumerate(fca_copy.get_leaves()):
                                    #     if n.name in sel_otus:
                                    #         if self.var_annot == {}:
                                    #             nn_split = n.name.split('|')[-1]
                                    #             nn_split = nn_split.split('__')
                                    #             for k, v in taxa_to_star.items():
                                    #                 if k.lower()[0] == nn_split[0].lower():
                                    #                     taxa_name = v
                                    #             node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                    #         else:
                                    #             node_name = self.var_annot.get(n.name, '(no annotation)')
                                    #             node_name = node_name.split(" ")
                                    #             remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                    #             new_node_name = ''
                                    #             for l in node_name:
                                    #                 if not l in remove_list:
                                    #                     new_node_name = new_node_name + l + ' '
                                    #             node_name = new_node_name
                                    #             if ',including' in node_name:
                                    #                 node_name = node_name.replace(',including', '').split(" ")[:2]
                                    #                 node_name = ' '.join(map(str, node_name))
                                    #             node_name = '{}'.format(node_name)
                                    #             n_split = node_name.split(' ')
                                    #             new_name = ''
                                    #             taxa_name = ''
                                    #             for nn in n_split:
                                    #                 if nn.lower() in taxa_to_star.keys():
                                    #                     taxa_name = taxa_to_star[nn.lower()]
                                    #                 else:
                                    #                     if '/' in nn:
                                    #                         nn = nn.split('/')[0]
                                    #                     new_name = new_name + nn + ' '
                                    #             if taxa_name == '':
                                    #                 taxa_name = taxa_to_star['species']
                                    #             node_name = '{} {}'.format(taxa_name, new_name)
                                    #         n.name = node_name
                                    #     else:
                                    #         n.delete()
                                    # fca_copy.write(outfile="{}/r{}d{}_subtree_abun.newick".format(
                                    #     dirName, i, p))
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200,
                                    #     tree_style=ts)
                                    # fca_copy.render('{}/r{}d{}_subtree_ete3_abun_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)

                                    while True:
                                        if len(fca_full_copy.get_leaves()) > len(sel_otus):
                                            break
                                        else:
                                            fca_full_copy = fca_full_copy.up

                                    tree_copy = deepcopy(self.phylo_tree)
                                    for n in tree_copy.traverse():
                                        nstyle = NodeStyle()
                                        nstyle["size"] = 0
                                        if n.name in sel_otus:
                                            nstyle["vt_line_color"] = "red"
                                            nstyle["hz_line_color"] = "red"
                                            nstyle["vt_line_width"] = 2
                                            nstyle["hz_line_width"] = 2
                                            nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                            nstyle["hz_line_type"] = 0
                                        n.set_style(nstyle)

                                        if n.is_leaf():
                                            n_idx = self.var_ids.index(n.name)
                                            if self.var_annot == {}:
                                                nn_split = n.name.split('|')[-1]
                                                nn_split = nn_split.split('__')
                                                for k, v in taxa_to_star.items():
                                                    if k.lower()[0] == nn_split[0].lower():
                                                        taxa_name = v
                                                node_name = 'OTU{} {} {}'.format(n_idx, taxa_name, ' '.join(nn_split[1].split('_')))
                                            else:
                                                node_name = self.var_annot.get(n.name, '(no annotation)')
                                                node_name = node_name.split(" ")
                                                remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                                new_node_name = ''
                                                for l in node_name:
                                                    if not l in remove_list:
                                                        new_node_name = new_node_name + l + ' '
                                                node_name = new_node_name
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{}'.format(node_name)
                                                n_split = node_name.split(' ')
                                                new_name = ''
                                                taxa_name = ''
                                                for nn in n_split:
                                                    if nn.lower() in taxa_to_star.keys():
                                                        taxa_name = taxa_to_star[nn.lower()]
                                                    else:
                                                        if '/' in nn:
                                                            nn = nn.split('/')[0]
                                                        new_name = new_name + nn + ' '
                                                if taxa_name == '':
                                                    taxa_name = taxa_to_star['species']
                                                node_name = 'OTU{} {} {}'.format(n_idx, taxa_name, new_name)
                                            n.name = node_name

                                    for n in fca_full_copy.traverse():
                                        nstyle = NodeStyle()
                                        nstyle["size"] = 0
                                        if n.name in sel_otus:
                                            nstyle["vt_line_color"] = "red"
                                            nstyle["hz_line_color"] = "red"
                                            nstyle["vt_line_width"] = 2
                                            nstyle["hz_line_width"] = 2
                                            nstyle["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
                                            nstyle["hz_line_type"] = 0
                                        n.set_style(nstyle)

                                        if n.is_leaf():
                                            if self.var_annot == {}:
                                                nn_split = n.name.split('|')[-1]
                                                nn_split = nn_split.split('__')
                                                for k, v in taxa_to_star.items():
                                                    if k.lower()[0] == nn_split[0].lower():
                                                        taxa_name = v
                                                node_name = '{} {}'.format(taxa_name, ' '.join(nn_split[1].split('_')))
                                            else:
                                                node_name = self.var_annot.get(n.name, '(no annotation)')
                                                node_name = node_name.split(" ")
                                                remove_list = ['OTU', 'mapped', 'to', 'a', 'clade', 'within']
                                                new_node_name = ''
                                                for l in node_name:
                                                    if not l in remove_list:
                                                        new_node_name = new_node_name + l + ' '
                                                node_name = new_node_name
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{}'.format(node_name)
                                                n_split = node_name.split(' ')
                                                new_name = ''
                                                taxa_name = ''
                                                for nn in n_split:
                                                    if nn.lower() in taxa_to_star.keys():
                                                        taxa_name = taxa_to_star[nn.lower()]
                                                    else:
                                                        if '/' in nn:
                                                            nn = nn.split('/')[0]
                                                        new_name = new_name + nn + ' '
                                                if taxa_name == '':
                                                    taxa_name = taxa_to_star['species']
                                                node_name = '{} {}'.format(taxa_name, new_name)
                                            n.name = node_name

                                    ts = TreeStyle()
                                    # ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 1
                                    ts.min_leaf_separation = 4
                                    ts.show_scale = False
                                    # fca_full_copy.render('{}/r{}d{}_tree_ete3_slope_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name),
                                    #     tree_style=ts)


                                    # fig = plt.figure()
                                    # fig.subplots_adjust(top=0.8)
                                    # ax = fig.add_subplot(121)
                                    slope_0 = list()
                                    slope_1 = list()
                                    for k in range(self.num_subjects):
                                        t_abun = x_time_slope[k, i, p]
                                        if labels[k]:
                                            # lines_1, = ax.plot(k, t_abun,
                                            #     marker='+', color='g')
                                            slope_1.append(t_abun)
                                        else:
                                            # lines_0, = ax.plot(k, t_abun,
                                            #     marker='.', color='#FF8C00')
                                            slope_0.append(t_abun)
                                    # line_thresh = ax.axhline(y=thresh,
                                    #     c='k', linestyle='--')
                                    # ax.set_xlabel('Subjects')
                                    # ax.set_ylabel('Slope')
                                    # ax.legend([lines_0, lines_1, line_thresh],
                                    #     [self.label_0, self.label_1, 'Threshold'],
                                    #     loc='upper left',
                                    #     fancybox=True, framealpha=0.5,
                                    #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                    log_odds = fc_wts[i] * r + fc_bias
                                    rule_eng = self.get_rule(fc_wts[i], log_odds, i, p, t_min, t_max, thresh)
                                    # plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    # tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_slope_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name))
                                    # ax1 = fig.add_subplot(122)
                                    # imgplot = plt.imshow(tree_img)
                                    # ax1.set_axis_off()
                                    # plt.savefig('{}/r{}d{}_slope_{}.png'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200)
                                    # plt.close()

                                    if self.args.save_as_csv:
                                        column_names = ['Subject', 'Label', *['day_{}'.format(day) for day in np.arange(self.num_time)]]
                                        csv_det_df = pd.DataFrame(columns=column_names)
                                    # fig = plt.figure(figsize=(20, 18))
                                    # fig.subplots_adjust(top=0.8)
                                    # gs = fig.add_gridspec(2, 2)
                                    # f_ax1 = fig.add_subplot(gs[0, 0])
                                    # f_ax2 = fig.add_subplot(gs[1, 0], sharex=f_ax1)
                                    mean_0 = list()
                                    mean_1 = list()
                                    abun_0 = list()
                                    abun_1 = list()
                                    mask_0 = list()
                                    mask_1 = list()
                                    for k in range(self.num_subjects):
                                        abun = x_spat[k, i, p]
                                        if labels[k]:
                                            # lines_1, = f_ax1.plot(self.times[k].astype('int'),
                                            #     abun[self.times[k].astype('int')],
                                            #     marker='.', color='g')
                                            mean_1.append(abun[self.times[k].astype('int')].mean())
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            abun_1.append(abun)
                                            mask_1.append(mask[k])
                                        else:
                                            # lines_0, = f_ax2.plot(self.times[k].astype('int'),
                                            #     abun[self.times[k].astype('int')],
                                            #     marker='.', color='#FF8C00')
                                            mean_0.append(abun[self.times[k].astype('int')].mean())
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            abun_0.append(abun)
                                            mask_0.append(mask[k])
                                        if self.args.save_as_csv:
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            csv_det_df.loc[len(csv_det_df.index)] = [k, labels[k], *abun]
                                    # f_ax1.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    # sel_days = np.arange(t_min, t_max + 1)
                                    # line_thresh, = f_ax1.plot(sel_days,
                                    #     thresh * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_0),
                                    #     c='k',
                                    #     linestyle='--', linewidth=3)
                                    # line_slope_1, = f_ax1.plot(sel_days,
                                    #     np.median(slope_1) * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_1),
                                    #     c='r',
                                    #     linestyle='--', linewidth=3)
                                    # f_ax1.set_ylabel('Abundance')
                                    # f_ax1.set_title(self.label_1)
                                    # plt.setp(f_ax1.get_xticklabels(), visible=False)
                                    # f_ax2.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    # line_thresh, = f_ax2.plot(sel_days,
                                    #     thresh * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_0),
                                    #     c='k',
                                    #     linestyle='--', linewidth=3)
                                    # line_slope_2, = f_ax2.plot(sel_days,
                                    #     np.median(slope_0) * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_0),
                                    #     c='r',
                                    #     linestyle='--', linewidth=3)
                                    # f_ax2.set_xlabel('Days')
                                    # f_ax2.set_ylabel('Abundance')
                                    # f_ax2.set_title(self.label_0)
                                    # log_odds = fc_wts[i] * r + fc_bias
                                    # rule_eng = self.get_rule(fc_wts[i], log_odds, i, p, t_min, t_max, thresh)
                                    # plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    # # tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_slope_{}.png'.format(
                                    # #     dirName, i, p, self.args.data_name))
                                    # # f_ax3 = fig.add_subplot(gs[:, 1])
                                    # # imgplot = plt.imshow(tree_img)
                                    # # f_ax3.set_axis_off()
                                    # pdf.savefig(fig, dpi=1200)  # saves the current figure into a pdf page
                                    # plt.savefig('{}/r{}d{}_slope_{}.pdf'.format(
                                    #     dirName, i, p, self.args.data_name), dpi=1200)
                                    # plt.close()

                                    labels_0 = np.array([int(lab) for lab in labels if lab == 0])
                                    labels_1 = np.array([int(lab) for lab in labels if lab == 1])
                                    new_t_min = min([t for t in self.time_unique if t >= t_min])
                                    new_t_max = max([t for t in self.time_unique if t <= t_max])
                                    rule_path = '{}/r{}d{}_heatmap_slope_{}.pdf'.format(
                                        dirName, i, p, self.args.data_name)
                                    out_f = {
                                        'x_0': x_spat[labels_0, i, p],
                                        'x_1': x_spat[labels_1, i, p],
                                        'x_0_mean': np.array(slope_0),
                                        'x_1_mean': np.array(slope_1),
                                        'tree_img': None,
                                        'thresh': thresh,
                                        'det_type': 'Slope',
                                        'win_start': t_min,
                                        'win_end': t_max,
                                        'num_sub_0': len(labels_0),
                                        'rule_eng': rule_eng,
                                        'rule_path': rule_path,
                                        'tree': fca_copy,
                                    }

                                    x_0 = np.array(abun_0)[:, self.time_unique]
                                    x_1 = np.array(abun_1)[:, self.time_unique]
                                    x_0[x_0 == -1] = np.nan
                                    x_1[x_1 == -1] = np.nan
                                    # self.heatmap_rule_viz(x_0, x_1,
                                    #     np.array(slope_0), np.array(slope_1), np.array(mask_0)[:, self.time_unique],
                                    #     np.array(mask_1)[:, self.time_unique], self.time_unique,
                                    #     fca_copy, thresh, 'Slope', new_t_min, new_t_max, len(labels_0),
                                    #     rule_eng, rule_path, t_min, t_max, pdf=pdf)


                                    ## new viz idea
                                    slope_win_len = int(0.75 * (t_max - t_min + 1))
                                    sliding_slope_0 = list()
                                    sliding_slope_1 = list()
                                    sliding_slope_mask_0 = list()
                                    sliding_slope_mask_1 = list()
                                    for k in range(self.num_subjects):
                                        abun = x_spat[k, i, p]
                                        sliding_slope = np.zeros_like(abun)
                                        slope_mask = np.zeros_like(abun)
                                        for t in range(self.num_time):
                                            if t in self.times[k]:
                                                if t < slope_win_len:
                                                    t_start = 0
                                                    t_end = t
                                                else:
                                                    t_start = t - slope_win_len
                                                    t_end = t
                                                win_mask = self.X_mask[k, t_start:t_end+1].sum()
                                                if win_mask >= 2:
                                                    tau = np.arange(t_start, t_end + 1) - ((t_start + t_end) // 2)
                                                    x_slope = np.polyfit(tau, abun[t_start:t_end+1], 1, w=self.X_mask[k, t_start:t_end+1])[0]
                                                    sliding_slope[t] = x_slope
                                                    slope_mask[t] = 1
                                                else:
                                                    sliding_slope[t] = np.nan
                                            else:
                                                sliding_slope[t] = np.nan
                                        if labels[k]:
                                            sliding_slope_1.append(sliding_slope)
                                            sliding_slope_mask_1.append(slope_mask)
                                        else:
                                            sliding_slope_0.append(sliding_slope)
                                            sliding_slope_mask_0.append(slope_mask)


                                    rule_path = '{}/r{}d{}_heatmap_slope_view_{}.pdf'.format(
                                        dirName, i, p, self.args.data_name)
                                    # self.heatmap_rule_viz(np.array(sliding_slope_0)[:, self.time_unique],
                                    #     np.array(sliding_slope_1)[:, self.time_unique],
                                    #     np.array(slope_0), np.array(slope_1), np.array(sliding_slope_mask_0)[:, self.time_unique],
                                    #     np.array(sliding_slope_mask_1)[:, self.time_unique], self.time_unique,
                                    #     fca_copy, thresh, 'Slope', new_t_min, new_t_max,
                                    #     rule_eng, rule_path, t_min, t_max, pdf=pdf, view_type='Slope')

                                    model_info = {
                                        'kappa': kappa,
                                        'eta': eta,
                                        'dist_from_eta': sel_dist_emb,
                                        'thresh': thresh,
                                        'dist_between_otus': sel_dist,
                                    }

                                    x_abun_slope_0.append(np.array(sliding_slope_0)[:, self.time_unique])
                                    x_abun_slope_1.append(np.array(sliding_slope_1)[:, self.time_unique])
                                    x_avg_abun_slope_0.append(np.array(slope_0))
                                    x_avg_abun_slope_1.append(np.array(slope_1))
                                    x_mask_0.append(np.array(sliding_slope_mask_0)[:, self.time_unique])
                                    x_mask_1.append(np.array(sliding_slope_mask_1)[:, self.time_unique])
                                    x_threshold.append(thresh)
                                    x_det_type.append('slope')
                                    x_new_t_min.append(new_t_min)
                                    x_new_t_max.append(new_t_max)
                                    x_t_min.append(new_t_min)
                                    x_t_max.append(new_t_max)
                                    taxa_tree.append(fca)
                                    full_tree.append(tree_copy)


                                    if self.args.save_as_csv:
                                        rule_path = '{}/rule_{}_detector_{}_slope.csv'.format(dirName, i, p)
                                        csv_df.loc[len(csv_df.index)] = [i, p, 'slope', rule_path, otu_annot_str, kappa, t_min, t_max,\
                                        'N/A', thresh, np.median(slope_0), np.median(slope_1), fc_wts[i], fc_bias]
                                        csv_det_df.to_csv(rule_path, index=False)

                                    if self.args.debug:
                                        kapps = list()
                                        mus = list()
                                        sigs = list()
                                        thres = list()
                                        zs = list()
                                        zrs = list()
                                        ws = list()
                                        bs = list()

                                        # grads
                                        kapps_grad = list()
                                        slope_a_grad = list()
                                        slope_b_grad = list()
                                        thres_grad = list()
                                        zs_grad = list()
                                        zrs_grad = list()
                                        ws_grad = list()
                                        bs_grad = list()
                                        for epoch in range(self.args.epochs):
                                            best_model_init = torch.load('{}/model_epoch_{}.pth'.format(dirName, epoch))
                                            kapp = (best_model_init.spat_attn.kappas).detach().cpu().numpy()[i, p]
                                            mu = (best_model_init.time_attn.m_slope).detach().cpu().numpy()[i, p]
                                            sig = (best_model_init.time_attn.s_slope).detach().cpu().numpy()[i, p]
                                            thre = best_model_init.slope_func.slope.detach().cpu().numpy()[i, p]
                                            zz = torch.sigmoid(best_model_init.rules_slope.alpha * self.args.max_k_bc).detach().cpu().numpy()[i, p]
                                            zr = torch.sigmoid(best_model_init.fc.beta * self.args.max_k_bc).detach().cpu().numpy()[i]
                                            w = best_model_init.fc.weight.view(-1).detach().cpu().numpy()[i]
                                            b = best_model_init.fc.bias.detach().cpu().numpy()
                                            kapps.append(kapp)
                                            mus.append(mu)
                                            sigs.append(sig)
                                            thres.append(thre)
                                            zs.append(zz)
                                            zrs.append(zr)
                                            ws.append(w)
                                            bs.append(b)

                                            kapp = best_model_init.grad_dict['spat_attn.kappa'][i, p]
                                            slope_a = best_model_init.grad_dict['time_attn.slope_a'][i, p]
                                            slope_b = best_model_init.grad_dict['time_attn.slope_b'][i, p]
                                            thre = best_model_init.grad_dict['slope_func.slope'][i, p]
                                            zz = best_model_init.grad_dict['rules_slope.alpha'][i, p]
                                            zr = best_model_init.grad_dict['fc.beta'][i]
                                            w = best_model_init.grad_dict['fc.weight'].view(-1)[i]
                                            b = best_model_init.grad_dict['fc.bias']
                                            kapps_grad.append(kapp)
                                            slope_a_grad.append(slope_a)
                                            slope_b_grad.append(slope_b)
                                            thres_grad.append(thre)
                                            zs_grad.append(zz)
                                            zrs_grad.append(zr)
                                            ws_grad.append(w)
                                            bs_grad.append(b)
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(kapps)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Kappa')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(kapps_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(mus)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Mu')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(slope_a_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('epsilon (slope) Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(sigs)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Sigma')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(slope_b_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('delta (slope) graidient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(thres)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Threshold')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(thres_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(zs)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Detector selector')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(zs_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(zrs)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Rule selector')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(zrs_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(ws)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Weight')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(ws_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()
                                        fig = plt.figure()
                                        ax_0 = fig.add_subplot(121)
                                        ax_0.plot(bs)
                                        ax_0.set_xlabel('Training Epochs')
                                        ax_0.set_ylabel('Bias')
                                        ax_1 = fig.add_subplot(122)
                                        ax_1.plot(bs_grad)
                                        ax_1.set_xlabel('Training Epochs')
                                        ax_1.set_ylabel('Gradient')
                                        pdf.savefig(bbox_inches='tight')
                                        plt.close()

                                    if best_model_init is not None:
                                        kappa = kappas_init[i, p]
                                        thresh = slopes_init[i, p]
                                        eta = etas_init[i, p]
                                        d_e = dist_emb_init[i, p]
                                        mu = time_mu_slope_init[i, p]
                                        t = time_wts_slope_init.mean(axis=0)[i, p]
                                        t_min = -1
                                        t_max = self.num_time
                                        for b in range(len(t)):
                                            if t[t_min + 1] <= (1 / self.num_time):
                                                t_min += 1
                                            if t[t_max - 1] <= (1 / self.num_time):
                                                t_max -= 1
                                        o = otu_wts_init[i, p]
                                        sel_otu_ids = [l for l, ot in enumerate(o) if ot >= 0.9]
                                        sel_dist = np.zeros((len(sel_otu_ids), len(sel_otu_ids)))
                                        for ii in range(len(sel_otu_ids)):
                                            for jj in range(len(sel_otu_ids)):
                                                sel_dist[ii, jj] = np.linalg.norm(self.dist_emb[ii] - self.dist_emb[jj], axis=-1)
                                        sel_dist_emb = np.zeros((len(sel_otu_ids)))
                                        for ii in range(len(sel_otu_ids)):
                                            sel_dist_emb[ii] = d_e[sel_otu_ids[ii]]
                                        # for ii in range(len(sel_otu_ids)):
                                        #     sel_dist_emb[ii] = np.linalg.norm(eta - self.dist_emb[ii], axis=-1)
                                        sel_otus = [self.var_names[self.otu_idx[l]] for l, ot in enumerate(o) if ot >= 0.5]
                                        otu_annot_str = ''
                                        if self.var_annot == {}:
                                            for n in sel_otus:
                                                otu_annot_str = otu_annot_str + '\n' + n
                                        else:
                                            sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                            for n in sel_otu_annot:
                                                otu_annot_str = otu_annot_str + '\n' + n

                                        tree = deepcopy(self.phylo_tree)
                                        sel_nodes = list()
                                        for n in tree.traverse():
                                            if n.name in sel_otus:
                                                sel_nodes.append(n)

                                        nstyle = NodeStyle()
                                        for n in tree.traverse():
                                            if n.name in sel_otus:
                                                fca = n.get_common_ancestor(sel_nodes)
                                                break
                                        if sel_nodes == []:
                                            pass

                                        for n in fca.traverse():
                                            if n.name in sel_otus:
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
                                                node_name = '{} {}'.format(n.name, node_name)
                                                text = TextFace(tw.fill(text=node_name), fsize=20)
                                                n.add_face(text, 0)
                                            else:
                                                nstyle = NodeStyle()
                                                nstyle["size"] = 0
                                                n.set_style(nstyle)

                                        ts = TreeStyle()
                                        ts.show_leaf_name = False
                                        ts.show_branch_length = True
                                        ts.branch_vertical_margin = 10
                                        ts.min_leaf_separation = 10
                                        ts.show_scale = False
                                        # fca.render('{}/r{}d{}_subtree_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     tree_style=ts)

                                        # fig = plt.figure()
                                        # ax = fig.add_subplot()
                                        # for k in range(self.num_subjects):
                                        #     abun = x_spat_init[k, i, p]
                                        #     if labels_init[k]:
                                        #         lines_1, = ax.plot(self.times[k].astype('int'),
                                        #             abun[self.times[k].astype('int')],
                                        #             marker='+', color='g',
                                        #             linewidth=1.5, markersize=8)
                                        #     else:
                                        #         lines_0, = ax.plot(self.times[k].astype('int'),
                                        #             abun[self.times[k].astype('int')],
                                        #             marker='.', color='#FF8C00',
                                        #             linewidth=1.5, markersize=8)
                                        # ax.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                        # ax.axvline(mu)
                                        # sel_days = np.arange(t_min, t_max + 1)
                                        # line_thresh, = ax.plot(sel_days,
                                        #     thresh * (sel_days - ((t_min + t_max + 1) / 2.)) + 0.1,
                                        #     c='k',
                                        #     linestyle='solid',
                                        #     linewidth=5)
                                        # ax.set_xlabel('Days', fontsize=20)
                                        # ax.set_ylabel('Slope', fontsize=20)
                                        # ax.set_title('wt: %.2f bias: %.2f r: %.2f d: %.2f kappa : %.2f' % (
                                        #     fc_wts_init[i],
                                        #     fc_bias_init, r, d_slope, kappa), fontsize=10)
                                        # ax.legend([lines_0, lines_1, line_thresh],
                                        #     [self.label_0, self.label_1, 'Slope Threshold'],
                                        #     fontsize=10, loc='upper left',
                                        #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                        # plt.savefig('{}/r{}d{}_slope_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     bbox_inches='tight')
                                        # plt.close()

                                        # fig = plt.figure()
                                        # ax = fig.add_subplot()
                                        # for k in range(self.num_subjects):
                                        #     t_abun = x_time_slope_init[k, i, p]
                                        #     if labels_init[k]:
                                        #         lines_1, = ax.plot(k, t_abun,
                                        #             marker='+', color='r')
                                        #     else:
                                        #         lines_0, = ax.plot(k, t_abun,
                                        #             marker='.', color='g')
                                        # line_thresh = ax.axhline(y=thresh,
                                        #     c='k', linestyle='--', linewidth=5, alpha=0.5)
                                        # ax.set_title('thresh: %.4f wt: %.2f bias: %.2f r: %.2f d: %.2f' % (
                                        #     thresh,
                                        #     fc_wts_init[i],
                                        #     fc_bias_init, r, d_slope), fontsize=10)
                                        # ax.set_xlabel('Subjects', fontsize=10)
                                        # ax.legend([lines_0, lines_1, line_thresh],
                                        #     [self.label_0, self.label_1, 'Threshold'],
                                        #     fontsize=10, loc='upper left',
                                        #     handler_map={tuple: HandlerTuple(ndivide=None)})
                                        # plt.savefig('{}/r{}d{}_slopeeeeeee_init_{}.pdf'.format(
                                        #     dirName, i, p, self.args.data_name),
                                        #     bbox_inches='tight')
                                        # plt.close()

                                        model_init_info = {
                                            'kappa': kappa,
                                            'eta': eta,
                                            'dist_from_eta': sel_dist_emb,
                                            'thresh': thresh,
                                        }

                                    # fig = plt.figure(figsize=(20, 18))
                                    # ax = fig.add_subplot()
                                    full_str = ''
                                    for k, v in model_info.items():
                                        if best_model_init is not None:
                                            model_init_v = model_init_info[k]
                                            model_init_info_str = 'INIT {}: {}\n'.format(k, model_init_v)
                                        else:
                                            model_init_info_str = ''
                                        model_final_info_str = 'FINAL {}: {}\n'.format(k, v)
                                        full_str = full_str + model_init_info_str + model_final_info_str + '\n'
                                    # ax.text(0, 1, full_str,
                                    #     horizontalalignment="left",
                                    #     verticalalignment="top", fontsize=12,)
                                    # ax.set_axis_off()
                                    # pdf.savefig(fig, dpi=1200)
                                    # plt.close()



                    rules_dict['taxa_tree'].append(taxa_tree)
                    rules_dict['x_0'].append(x_abun_slope_0)
                    rules_dict['x_1'].append(x_abun_slope_1)
                    rules_dict['x_mask_0'].append(x_mask_0)
                    rules_dict['x_mask_1'].append(x_mask_1)
                    rules_dict['x_avg_0'].append(x_avg_abun_slope_0)
                    rules_dict['x_avg_1'].append(x_avg_abun_slope_1)
                    rules_dict['det_type'].append(x_det_type)
                    rules_dict['thresh'].append(x_threshold)
                    rules_dict['t_min'].append(x_t_min)
                    rules_dict['t_max'].append(x_t_max)
                    rules_dict['new_t_min'].append(x_new_t_min)
                    rules_dict['new_t_max'].append(x_new_t_max)
                    rules_dict['full_tree'].append(full_tree)


            if self.args.debug:
                for epoch in range(self.args.epochs):
                    os.remove('{}/model_epoch_{}.pth'.format(dirName, epoch))

            if self.args.save_as_csv:
                csv_df.to_csv('{}/rules_dump.csv'.format(dirName), index=False)

            with open('{}/rules_dump.pickle'.format(dirName), 'wb') as f:
                pickle.dump(rules_dict, f)


if __name__ == '__main__':
    # Parse command line args
    args = parse()

    # Init trainer object
    trainer = Trainer(args)

    # Load data
    trainer.load_data()

    # run cv loop
    trainer.train_loop()
