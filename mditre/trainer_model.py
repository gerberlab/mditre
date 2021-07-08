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

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.cluster import KMeans
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


import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.log_normal import LogNormal
from torch.distributions.beta import Beta
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal

from mditre.data import *
from mditre.models import MDITRE, binary_concrete, transf_log, inv_transf_log
from mditre.utils import AverageMeter, get_logger

import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    parser.add_argument('--data', metavar='DIR',
                        default='./datasets/david_agg_filtered.pickle',
                        help='path to dataset')
    parser.add_argument('--data_name', default='David', type=str,
                        help='Name of the dataset, will be used for log dirname')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr_kappa', default=0.001, type=float,
                        help='Initial learning rate for kappa.')
    parser.add_argument('--lr_eta', default=0.001, type=float,
                        help='Initial learning rate for eta.')
    parser.add_argument('--lr_time', default=0.01, type=float,
                        help='Initial learning rate for sigma.')
    parser.add_argument('--lr_mu', default=0.01, type=float,
                        help='Initial learning rate for mu.')
    parser.add_argument('--lr_thresh', default=0.001, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--lr_slope', default=1e-4, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--lr_alpha', default=0.001, type=float,
                        help='Initial learning rate for binary concrete logits on detectors.')
    parser.add_argument('--lr_beta', default=0.001, type=float,
                        help='Initial learning rate for binary concrete logits on rules.')
    parser.add_argument('--lr_fc', default=0.001, type=float,
                        help='Initial learning rate for linear classifier weights and bias.')
    parser.add_argument('--lr_bias', default=0.001, type=float,
                        help='Initial learning rate for linear classifier weights and bias.')
    parser.add_argument('--deterministic', action='store_true', default=True,)
    parser.add_argument('--seed', type=int, default=42,
                        help='Set random seed for reproducibility')
    parser.add_argument('--min_k_otu', default=10, type=float,
                        help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--max_k_otu', default=100, type=float,
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
    parser.add_argument('--max_k_bc', default=100, type=float,
                        help='Max Temperature for binary concretes')
    parser.add_argument('--n_d', type=int, default=10,
                        help='Number of detectors (otus)')
    parser.add_argument('--cv_type', type=str, default='None',
                        choices=['loo', 'kfold', 'None'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--z_mean', type=float, default=0,
                        help='NBD Mean active detectors per rule')
    parser.add_argument('--z_var', type=float, default=1,
                        help='NBD variance of active detectors per rule')
    parser.add_argument('--z_r_mean', type=float, default=0,
                        help='NBD Mean active rules')
    parser.add_argument('--z_r_var', type=float, default=1,
                        help='NBD variance of active rules')
    parser.add_argument('--w_var', type=float, default=1e5,
                        help='Normal prior variance on weights.')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed multiprocess training')
    parser.add_argument('--save_as_csv', action='store_true', default=True,
                        help='Debugging')
    parser.add_argument('--inner_cv', action='store_true',
                        help='Do inner cross val')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print training logs')

    args = parser.parse_args([])
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

        if self.args.distributed and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://',
                                                 timeout=datetime.timedelta(days=7))
            self.args.world_size = torch.distributed.get_world_size()
            self.args.rank = torch.distributed.get_rank()
        else:
            self.args.rank = 0
            self.args.world_size = 1

        # Check for determinism in training, set random seed
        if self.args.deterministic:
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

        # Total number of subjects in the dataset
        self.num_subjects = self.X.shape[0]

        if self.args.batch_size > self.num_subjects:
            self.args.batch_size = self.num_subjects

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
        elif 't1d' in self.args.data:
            self.label_0 = 'No T1D'
            self.label_1 = 'T1D'
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
            samples, times, otu_idx, self.y)

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


    def plot_hist(self, x, title, filename, show_stats=True, pdf=None):
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.histplot(x, ax=ax)
        if show_stats:
            med = np.median(x)
            mea = np.mean(x)
            perc_25 = np.percentile(x, 25)
            perc_75 = np.percentile(x, 75)
            ax.axvline(med,color='r', linestyle='-', label='Median: {:.2f}'.format(med))
            ax.axvline(mea,color='g', linestyle='-', label='Mean: {:.2f}'.format(mea))
            ax.axvline(perc_25, color='y', linestyle='-', label='25 percentile: {:.2f}'.format(perc_25))
            ax.axvline(perc_75, color='b', linestyle='-', label='75 percentile: {:.2f}'.format(perc_75))
        ax.set_title(title)
        plt.legend()
        if pdf is not None:
            pdf.savefig(fig)
        else:
            plt.savefig(filename)
        plt.close()
        return

    def plot_bar(self, x, title, filename):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(np.arange(len(x)), x)
        ax.set_title(title, loc='center', wrap=True)
        plt.legend()
        plt.savefig(filename)
        plt.close()
        return


    def set_model_hparams(self):
        # Number of rules
        self.num_rules = 10
        # Number of otu centers per rule
        self.num_detectors = self.args.n_d
        self.emb_dim = 10
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

        # Set mean and variance for neg bin prior for rules
        self.z_r_mean = self.args.z_r_mean
        self.z_r_var = self.args.z_r_var
        if self.use_gpu:
            mean = torch.tensor([self.z_r_mean], dtype=torch.float32).cuda()
            var = torch.tensor([self.z_r_var], dtype=torch.float32).cuda()
        else:
            mean = self.z_r_mean
            var = self.z_r_var

        ### Prior hyperparams ###
        ref_median_dist = {
            'median_genus': 0.24192747,
            'median_family': 0.31568444,
            'median_phylum': 0.9304266,
        }
        self.dist_emb = self.compute_dist_emb_mds().astype(np.float32)
        self.dist_matrix = self.compute_dist()

        # Use median of the above distribution as the prior mean
        self.kappa_prior_mean = ref_median_dist['median_family']
        self.kappa_prior_var = 1e5
        if self.use_gpu:
            mean = torch.tensor([np.log(self.kappa_prior_mean)], dtype=torch.float32).cuda()
            std = torch.tensor(np.sqrt([self.kappa_prior_var]), dtype=torch.float32).cuda()
        else:
            mean = np.log(self.kappa_prior_mean)
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
                dist[i, j] = np.linalg.norm(self.dist_emb[i] - self.dist_emb[j])

        return dist


    def train_loop(self):
        # Init model hyperparams
        self.set_model_hparams()

        # # Init otu centers
        detector_otuids = list()
        eta_init = np.zeros((self.num_rules, self.num_detectors, self.emb_dim), dtype=np.float32)
        kappa_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
        kmeans = KMeans(n_clusters=self.num_detectors, random_state=self.args.seed).fit(self.dist_emb)
        for i in range(self.num_rules):
            assigned_otus_det = list()
            for j in range(self.num_detectors):
                assigned_otus = list()
                eta_init[i, j] = kmeans.cluster_centers_[j]
                med_dist = list()
                for k in range(self.num_otus):
                    if kmeans.labels_[k] == j:
                        med_dist.append(np.linalg.norm(kmeans.cluster_centers_[j] - self.dist_emb[k]))
                        cur_assig_otu = k
                if len(med_dist) > 1:
                    kappa_init[i, j] = np.mean(med_dist)
                else:
                    d = self.dist_matrix[cur_assig_otu]
                    kappa_init[i, j] = np.percentile(d, 10)
                for k in range(self.num_otus):
                    if kmeans.labels_[k] == j:
                        dist = np.linalg.norm(kmeans.cluster_centers_[j] - self.dist_emb[k])
                        if dist <= kappa_init[i, j]:
                            assigned_otus.append(k)
                assigned_otus_det.append(assigned_otus)
            detector_otuids.append(assigned_otus_det)


        # Compute minimum time-window lengths for each possible
        # time center
        time_unique = np.unique(np.concatenate(np.array(self.times))).astype('int')
        window_len = None
        acc_center_abun = list()
        acc_len_abun = list()
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

        window_len_slope = None
        acc_center_slope = list()
        acc_len_slope = list()
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

        N_cur = min(3, len(acc_center_abun))
        N_cur_slope = min(3, len(acc_center_slope))

        # Init logsitic regression weights
        alpha_init = np.zeros((self.num_rules, self.num_detectors))
        beta_init = np.zeros((self.num_rules))
        w_init = np.random.normal(0, 1, (1, self.num_rules))
        bias_init = np.zeros((1))


        # Init cross-val data splits
        self.get_cv_splits()

        full_loader = get_data_loaders(self.X, self.y,
            self.X_mask,
            len(self.X),
            self.args.workers,
            shuffle=False, pin_memory=self.use_gpu)

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

                if not self.args.rank:
                    self.logger.info('Initializing model!')

                # Init time windows
                t_split = np.array_split(acc_center_abun, N_cur)
                t_split_slope = np.array_split(acc_center_slope, N_cur_slope)
                mu_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                mu_slope_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                sigma_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                sigma_slope_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                thresh_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                slope_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
                for l in range(self.num_rules):
                    for m in range(self.num_detectors):
                        X = np.zeros((len(train_ids), N_cur), dtype=np.float32)
                        X_slope = np.zeros((len(train_ids), N_cur_slope), dtype=np.float32)
                        selected_center_abun = list()
                        selected_center_slope = list()
                        for n in range(N_cur):
                            selected_center_abun.append(np.random.choice(t_split[n]))
                        for n in range(N_cur_slope):
                            selected_center_slope.append(np.random.choice(t_split_slope[n]))
                        for n, ts in enumerate(selected_center_abun):
                            win_len = acc_len_abun[acc_center_abun.index(ts)]
                            window_start = np.floor(ts - (win_len / 2.)).astype('int')
                            window_end = np.ceil(ts + (win_len / 2.)).astype('int')
                            x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start:window_end]
                            x_mask = self.X_mask[train_ids, :][:, window_start:window_end]
                            X[:, n] = x.sum(1).sum(-1) / x_mask.sum(-1)
                        for n, ts in enumerate(selected_center_slope):
                            win_len = acc_len_slope[acc_center_slope.index(ts)]
                            window_start = np.floor(ts - (win_len / 2.)).astype('int')
                            window_end = np.ceil(ts + (win_len / 2.)).astype('int')
                            x = self.X[train_ids][:, detector_otuids[l][m], :][:, :, window_start:window_end]
                            x_mask = self.X_mask[train_ids, :][:, window_start:window_end]
                            tau = np.arange(window_start, window_end) - ts
                            X_slope[:, n] = np.array([np.polyfit(tau, x[s].sum(0), 1, w=x_mask[s])[0] for s in range(len(train_ids))])
                        scaler = StandardScaler()
                        X_transf = scaler.fit_transform(X)
                        clf = LogisticRegressionCV(cv=LeaveOneOut(), penalty='l1',
                            random_state=self.args.seed, max_iter=10000, n_jobs=-1,
                            solver='liblinear').fit(X, self.y[train_ids])
                        best_t_id_abun = np.argsort(np.absolute(clf.coef_[0]))[::-1][0]
                        best_time_win_abun = selected_center_abun[best_t_id_abun]
                        scaler = StandardScaler()
                        X_slope_transf = scaler.fit_transform(X_slope)
                        clf = LogisticRegressionCV(cv=LeaveOneOut(), penalty='l1',
                            random_state=self.args.seed, max_iter=10000, n_jobs=-1,
                            solver='liblinear').fit(X_slope, self.y[train_ids])
                        best_t_id_slope = np.argsort(np.absolute(clf.coef_[0]))[::-1][0]
                        best_time_win_slope = selected_center_slope[best_t_id_slope]
                        sigma_init[l, m] = acc_len_abun[acc_center_abun.index(best_time_win_abun)]
                        sigma_slope_init[l, m] = acc_len_slope[acc_center_slope.index(best_time_win_slope)]
                        mu_init[l, m] = best_time_win_abun
                        mu_slope_init[l, m] = best_time_win_slope
                        thresh_init[l, m] = np.mean(np.mean(X[:, best_t_id_abun], axis=-1))
                        slope_init[l, m] = np.mean(np.mean(X_slope[:, best_t_id_slope], axis=-1))


                abun_a_init = sigma_init / self.num_time
                abun_a_init = np.clip(abun_a_init, 1e-2, 1 - 1e-2)
                abun_b_init = (mu_init - (self.num_time * abun_a_init / 2.)) / ((1 - abun_a_init) * self.num_time)
                abun_b_init = np.clip(abun_b_init, 1e-2, 1 - 1e-2)
                slope_a_init = sigma_slope_init / self.num_time
                slope_a_init = np.clip(slope_a_init, 1e-2, 1 - 1e-2)
                slope_b_init = (mu_slope_init - (self.num_time * slope_a_init / 2.)) / ((1 - slope_a_init) * self.num_time)
                slope_b_init = np.clip(slope_b_init, 1e-2, 1 - 1e-2)

                # Initiazation dict
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
                model = MDITRE(self.num_rules, self.num_otus, self.num_detectors,
                    self.num_time, 1, self.dist_emb, self.emb_dim)
                model.init_params(init_args)
                model.to(self.device)
                model_init = deepcopy(model)

                if not self.args.rank:
                    if self.args.verbose:
                        for k, v in model.named_parameters():
                            self.logger.info(k)
                            self.logger.info(v)
                    self.logger.info('Model training started!')

                # Inner loop training, store best model
                best_model = self.train_model(model, train_loader,
                    val_loader, test_loader, i)

                # Eval on test set to get f1
                probs, true = self.eval_with_preds(best_model, test_loader)
                k_fold_test_prob[test_ids] = probs
                k_fold_test_preds[test_ids] = (np.array(probs) > 0.5).astype(np.float32)
                k_fold_test_true[test_ids] = true

                if not self.args.rank:
                    if self.args.verbose:
                        for k, v in best_model.named_parameters():
                            self.logger.info(k)
                            self.logger.info(v)

                self.show_rules(best_model, full_loader, i, save_viz=True, best_model_init=model_init)


        if self.args.world_size > 1:
            k_fold_test_true_tens = torch.from_numpy(k_fold_test_true).to(self.device)
            k_fold_test_preds_tens = torch.from_numpy(k_fold_test_preds).to(self.device)

            torch.distributed.all_reduce(k_fold_test_true_tens, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(k_fold_test_preds_tens, op=torch.distributed.ReduceOp.SUM)

            k_fold_test_true = k_fold_test_true_tens.detach().cpu().numpy()
            k_fold_test_preds = k_fold_test_preds_tens.detach().cpu().numpy()


        if not self.args.rank:
            # Compute final cv f1 score
            cv_f1 = f1_score(k_fold_test_true, k_fold_test_preds)
            cv_auc = roc_auc_score(k_fold_test_true, k_fold_test_preds)
            clf_report = classification_report(k_fold_test_true, k_fold_test_preds)
            tn, fp, fn, tp = confusion_matrix(k_fold_test_true, k_fold_test_preds).ravel()

            end = time.time()

            self.logger.info('F1 score: %.2f' % (cv_f1))
            self.logger.info('AUC score: %.2f' % (cv_auc))
            self.logger.info('Preds: {}'.format(k_fold_test_preds))
            self.logger.info('labels: {}'.format(k_fold_test_true))
            self.logger.info(clf_report)
            self.logger.info('FP: {} FN: {}'.format(fp, fn))
            self.logger.info('Total train time: %.2f hrs' % ((end - start) / 3600.))

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
                detectors_slope = model.rules_slope.z
                negbin_zr_loss = self.negbin_loss(rules.sum(), self.z_r_mean, self.z_r_var)
                negbin_z_loss = self.negbin_loss(detectors.sum(dim=-1) + detectors_slope.sum(dim=-1),
                    self.z_mean, self.z_var).sum()
                l2_wts_loss = -self.normal_wts.log_prob(model.fc.weight).sum()
                time_wts_abun = model.time_attn.wts.sum(dim=-1)
                time_wts_slope = model.time_attn.wts_slope.sum(dim=-1)
                time_loss = -(torch.sigmoid((time_wts_abun - 1.) * 10).prod(dim=0)).sum()
                time_slope_loss = -(torch.sigmoid((time_wts_slope - 2.) * 10).prod(dim=0)).sum()
                emb_normal_loss = -self.normal_emb.log_prob(model.spat_attn.eta).sum()
                time_abun_a_normal_loss = -self.normal_time_abun_a.log_prob(model.time_attn.abun_a).sum()
                time_slope_a_normal_loss = -self.normal_time_slope_a.log_prob(model.time_attn.slope_a).sum()
                time_abun_b_normal_loss = -self.normal_time_abun_b.log_prob(model.time_attn.abun_b).sum()
                time_slope_b_normal_loss = -self.normal_time_slope_b.log_prob(model.time_attn.slope_b).sum()
                rule_normal_loss = -self.normal_rule.log_prob(model.fc.beta).sum()
                det_normal_loss = -(self.normal_det.log_prob(model.rules.alpha).sum() + \
                    self.normal_det.log_prob(model.rules_slope.alpha).sum())
                kappa_normal_loss = -self.normal_kappa.log_prob(model.spat_attn.kappa).sum()

                reg_loss = negbin_zr_loss + negbin_z_loss + l2_wts_loss + \
                    emb_normal_loss + time_loss + time_slope_loss + \
                    time_abun_a_normal_loss + time_slope_a_normal_loss + \
                    time_abun_b_normal_loss + time_slope_b_normal_loss + \
                    kappa_normal_loss + rule_normal_loss + det_normal_loss

                # Backprop for computing grads
                loss = train_loss + reg_loss
                loss.backward()

                # Update parameters
                optimizer_0.step()

                # Uniform prior implemented avia clipping
                model.thresh_func.thresh.data.clamp_(0, 1)
                model.slope_func.slope.data.clamp_(-1, 1)

                # track avg loss, f1
                train_loss_avg.update(train_loss, 1)
                train_f1 = f1_score(labels.detach().cpu().numpy(),
                    outputs.sigmoid().detach().cpu().numpy() > 0.5)
                train_f1_avg.update(train_f1, 1)

            # Evaluate on val data
            val_loss, val_f1 = self.eval(model, val_loader)
            # Save stats and model for best val performance
            if val_f1 >= best_val_f1:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                best_model = deepcopy(model)

            # Track test data stats
            test_loss, test_f1 = self.eval(model, test_loader)

            ls.append(loss.item())
            val_ls.append(val_loss)
            test_ls.append(test_loss)
            cels.append(train_loss.item())
            zls.append(negbin_z_loss.item())
            zrls.append(negbin_zr_loss.item())
            wls.append(l2_wts_loss.item())
            tls.append(time_loss.item())
            tsls.append(time_slope_loss.item())
            tlsa.append(time_abun_a_normal_loss.item())
            tlsb.append(time_abun_b_normal_loss.item())
            tslsa.append(time_slope_a_normal_loss.item())
            tslsb.append(time_slope_b_normal_loss.item())
            embls.append(emb_normal_loss.item())
            kls.append(kappa_normal_loss.item())
            rnls.append(rule_normal_loss.item())
            dnls.append(det_normal_loss.item())

            if self.args.save_as_csv:
                losses_csv.loc[len(losses_csv.index)] = [epoch, loss.item(), train_loss.item(),\
                val_loss, test_loss, negbin_zr_loss.item(),\
                negbin_z_loss.item(), l2_wts_loss.item(), time_loss.item(), time_slope_loss.item()]

            # Print epoch stats
            if self.args.verbose:
                if not self.args.rank:
                    kfold_epochs = 'Outer kfold: %d Epoch: %d '
                    train_stats = 'TrainLoss: %.2f TrainF1: %.2f '
                    val_stats = 'ValLoss: %.2f ValF1: %.2f '
                    test_stats = 'TestLoss: %.2f TestF1: %.2f '
                    log_str = kfold_epochs + train_stats + val_stats + test_stats
                    self.logger.info(
                        log_str % (outer_fold, epoch,
                            train_loss_avg.avg, train_f1_avg.avg,
                            val_loss, val_f1, test_loss, test_f1,
                        )
                    )

            scheduler_0.step()

        # save loss plots
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        axd = fig.subplot_mosaic(
            '''
            ABC
            DEF
            GHI
            JKL
            MNO
            PQ.
            ''')
        axd['A'] = self.simple_plot(ls, 'Training Epochs', 'Total loss', '{}/loss_traj.pdf'.format(dirName), ax=axd['A'])
        axd['B'] = self.simple_plot(cels, 'Training Epochs', 'Train CE loss', '{}/train_celoss_traj.pdf'.format(dirName), ax=axd['B'])
        axd['C'] = self.simple_plot(val_ls, 'Training Epochs', 'Val CE loss', '{}/val_celoss_traj.pdf'.format(dirName), ax=axd['C'])
        axd['D'] = self.simple_plot(test_ls, 'Training Epochs', 'Test CE loss', '{}/test_celoss_traj.pdf'.format(dirName), ax=axd['D'])
        axd['E'] = self.simple_plot(zls, 'Training Epochs', 'det loss', '{}/zloss_traj.pdf'.format(dirName), ax=axd['E'])
        axd['F'] = self.simple_plot(zrls, 'Training Epochs', 'rules loss', '{}/zrloss_traj.pdf'.format(dirName), ax=axd['F'])
        axd['G'] = self.simple_plot(wls, 'Training Epochs', 'wts loss', '{}/wloss_traj.pdf'.format(dirName), ax=axd['G'])
        axd['H'] = self.simple_plot(wls, 'Training Epochs', 'cov. loss (abun)', '{}/tloss_traj.pdf'.format(dirName), ax=axd['H'])
        axd['I'] = self.simple_plot(wls, 'Training Epochs', 'cov. loss (slope)', '{}/tslopeloss_traj.pdf'.format(dirName), ax=axd['I'])
        axd['J'] = self.simple_plot(kls, 'Training Epochs', 'kappa loss', '{}/kappaloss_traj.pdf'.format(dirName), ax=axd['J'])
        axd['K'] = self.simple_plot(tlsa, 'Training Epochs', 'eps loss (abun)', '{}/timelossa_abun_traj.pdf'.format(dirName), ax=axd['K'])
        axd['L'] = self.simple_plot(tlsb, 'Training Epochs', 'del loss (abun)', '{}/timelossb_abun_traj.pdf'.format(dirName), ax=axd['L'])
        axd['M'] = self.simple_plot(tslsa, 'Training Epochs', 'eps loss (slope)', '{}/timelossa_slope_traj.pdf'.format(dirName), ax=axd['M'])
        axd['N'] = self.simple_plot(tslsb, 'Training Epochs', 'del loss (slope)', '{}/timelossb_slope_traj.pdf'.format(dirName), ax=axd['N'])
        axd['O'] = self.simple_plot(embls, 'Training Epochs', 'Emb loss', '{}/embloss_traj.pdf'.format(dirName), ax=axd['O'])
        axd['P'] = self.simple_plot(rnls, 'Training Epochs', 'Rule Norm. loss', '{}/ruleloss_traj.pdf'.format(dirName), ax=axd['P'])
        axd['Q'] = self.simple_plot(embls, 'Training Epochs', 'Det. Norm. loss', '{}/detloss_traj.pdf'.format(dirName), ax=axd['Q'])
        plt.subplot_tool()
        plt.savefig('{}/losses.pdf'.format(dirName))
        plt.close()


        if self.args.save_as_csv:
            losses_csv.to_csv('{}/losses_dump.csv'.format(dirName), index=False)

        return best_model


    def simple_plot(self, x, xlabel, ylabel, filepath, pdf=None, ax=None):
        if ax is None:
            fig = plt.figure()
            axx = fig.add_subplot()
            axx.plot(x)
            # axx.set_xlabel(xlabel)
            axx.set_title(ylabel, fontsize='small')
            start, end = axx.get_xlim()
            axx.xaxis.set_ticks(np.arange(start, end, 0.1 * (start - end)))
        else:
            ax.plot(x)
            # ax.set_xlabel(xlabel)
            ax.set_title(ylabel)
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(start, end, 0.1 * (start - end)))
        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        elif ax is not None:
            return ax
        else:
            plt.savefig(filepath, bbox_inches='tight')
        plt.close()


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
        loss = F.binary_cross_entropy_with_logits(logits, labels,
            reduction='sum')

        return loss


    def negbin_loss(self, x, mean, var):
        r = mean ** 2 / (var - mean)
        p = (var - mean) / var
        loss_1 = -torch.lgamma(r + x + 1e-5)
        loss_2 = torch.lgamma(x + 1)
        loss_3 = -(x * np.log(p))
        loss = loss_1 + loss_2 + loss_3
        return loss


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


    def get_rule(self, w, b, r, d, t_min, t_max, thresh, metric='slope'):
        if w > 0.:
            out = self.label_1
        else:
            out = self.label_0
        rule = "Rule {} Detector {}: TRUE for {} if the average {} of selected taxa between days {} to {} is greater than {:.4f}. (Weight: {:.2f} Bias: {:.2f})".format(
            r, d, out, metric, t_min, t_max, thresh, w, b)
        return rule


    def show_rules(self, best_model, test_loader, fold, save_viz=True, best_model_init=None):
        dirName = '{}/fold_{}'.format(self.log_dir, fold)
        self.create_dir(dirName)
        torch.save(best_model.state_dict(), '{}/best_model.pth'.format(dirName))

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
            labels = labels.detach().cpu().numpy()
            fc_wts = best_model.fc.weight.view(-1).detach().cpu().numpy()
            fc_bias = best_model.fc.bias.detach().cpu().item()
            kappas = (best_model.spat_attn.kappas).detach().cpu().numpy()
            x_thresh = x_thresh.detach().cpu().numpy()
            x_slope = x_slope.detach().cpu().numpy()

            if best_model_init is not None:
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


            # Create the PdfPages object to which we will save the pages:
            # The with statement makes sure that the PdfPages object is closed properly at
            # the end of the block, even if an Exception occurs.
            with PdfPages('{}/rules.pdf'.format(dirName)) as pdf:
                for i, r in enumerate(rules):
                    if r >= 0.9:
                        for p in range(self.num_detectors):
                            # for z in range(self.args.n_w):
                            d = detectors[i, p]
                            d_slope = detectors_slope[i, p]
                            if d >= 0.9:
                                kappa = kappas[i, p]
                                thresh = threshs[i, p]
                                mu = time_mu[i, p]
                                t_min = int(time_mu[i, p] - (time_sigma[i, p] // 2))
                                t_max = int(time_mu[i, p] + (time_sigma[i, p] // 2))
                                o = otu_wts[i, p]
                                sel_otus = [self.var_names[self.otu_idx[l]] for l, ot in enumerate(o) if ot >= 0.9]
                                otu_annot_str = ''
                                if self.var_annot == {}:
                                    for n in sel_otus:
                                        otu_annot_str = otu_annot_str + '\n' + n
                                else:
                                    sel_otu_annot = [self.var_annot[l] for l in sel_otus]
                                    for n in sel_otu_annot:
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
                                    
                                    sel_nodes = list()
                                    fca_copy = deepcopy(fca)
                                    for n in fca.get_leaves():
                                        if n.name in sel_otus:
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
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{} {}'.format(n.name, node_name)
                                            text = TextFace(tw.fill(text=node_name), fsize=30)
                                            n.add_face(text, 0)
                                        else:
                                            n.delete()

                                    for n in fca_copy.get_leaves():
                                        if n.name in sel_otus:
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
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{} {}'.format(n.name, node_name)
                                            n.name = node_name
                                        else:
                                            n.delete()

                                    ts = TreeStyle()
                                    ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 10
                                    ts.min_leaf_separation = 10
                                    ts.show_scale = False
                                    fca.render('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                        dirName, i, p, self.args.data_name), dpi=1200,
                                        tree_style=ts)
                                    fca_copy.write(outfile="{}/r{}d{}_subtree_abun.newick".format(
                                        dirName, i, p))

                                    if self.args.save_as_csv:
                                        column_names = ['Subject', 'Label', *['day_{}'.format(day) for day in np.arange(self.num_time)]]
                                        csv_det_df = pd.DataFrame(columns=column_names)
                                    fig = plt.figure()
                                    fig.subplots_adjust(top=0.8)
                                    gs = fig.add_gridspec(2, 2)
                                    f_ax1 = fig.add_subplot(gs[0, 0])
                                    f_ax2 = fig.add_subplot(gs[1, 0], sharex=f_ax1)
                                    abun_0 = list()
                                    abun_1 = list()
                                    for k in range(self.num_subjects):
                                        t_abun = x_time[k, i, p]
                                        if labels[k]:
                                            abun_1.append(t_abun)
                                        else:
                                            abun_0.append(t_abun)
                                    for k in range(self.num_subjects):
                                        abun = x_spat[k, i, p]
                                        if labels[k]:
                                            lines_1, = f_ax1.plot(self.times[k].astype('int'),
                                                abun[self.times[k].astype('int')],
                                                marker='.', color='g')
                                        else:
                                            lines_0, = f_ax2.plot(self.times[k].astype('int'),
                                                abun[self.times[k].astype('int')],
                                                marker='.', color='#FF8C00')
                                        if self.args.save_as_csv:
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            csv_det_df.loc[len(csv_det_df.index)] = [k, labels[k], *abun]
                                    f_ax1.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    sel_days = np.arange(t_min, t_max + 1)
                                    line_thresh = f_ax1.axhline(y=thresh,
                                        xmin=((t_min) / self.num_time),
                                        xmax=((t_max + 1) / self.num_time),
                                        c='k', linestyle='--', linewidth=3)
                                    line_thresh_1 = f_ax1.axhline(y=np.median(abun_1),
                                        xmin=((t_min) / self.num_time),
                                        xmax=((t_max + 1) / self.num_time),
                                        c='r', linestyle='--', linewidth=3)
                                    f_ax1.set_ylabel('Relative Abundance')
                                    f_ax1.set_title(self.label_1)
                                    plt.setp(f_ax1.get_xticklabels(), visible=False)
                                    f_ax2.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    line_thresh = f_ax2.axhline(y=thresh,
                                        xmin=((t_min) / self.num_time),
                                        xmax=((t_max + 1) / self.num_time),
                                        c='k', linestyle='--', linewidth=3)
                                    line_thresh_0 = f_ax2.axhline(y=np.median(abun_0),
                                        xmin=((t_min) / self.num_time),
                                        xmax=((t_max + 1) / self.num_time),
                                        c='r', linestyle='--', linewidth=3)
                                    f_ax2.set_xlabel('Days')
                                    f_ax2.set_ylabel('Relative Abundance')
                                    f_ax2.set_title(self.label_0)
                                    rule_eng = self.get_rule(fc_wts[i], fc_bias, i,
                                        self.num_detectors, t_min, t_max, thresh, metric='abundance')
                                    plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                        dirName, i, p, self.args.data_name))
                                    f_ax3 = fig.add_subplot(gs[:, 1])
                                    imgplot = plt.imshow(tree_img)
                                    f_ax3.set_axis_off()
                                    pdf.savefig(fig, bbox_inches='tight', dpi=1200)
                                    os.remove('{}/r{}d{}_subtree_ete3_abun_{}.png'.format(
                                        dirName, i, p, self.args.data_name))
                                    plt.close()

                                    if self.args.save_as_csv:
                                        rule_path = '{}/rule_{}_detector_{}_abun.csv'.format(dirName, i, p)
                                        csv_df.loc[len(csv_df.index)] = [i, p, 'abundance', rule_path, otu_annot_str, kappa, t_min, t_max,\
                                        thresh, 'N/A', np.median(abun_0), np.median(abun_1), fc_wts[i], fc_bias]
                                        csv_det_df.to_csv(rule_path, index=False)

                            if d_slope >= 0.9:
                                kappa = kappas[i, p]
                                thresh = slopes[i, p]
                                mu = time_mu_slope[i, p]
                                t_min = int(time_mu_slope[i, p] - (time_sigma_slope[i, p] // 2))
                                t_max = int(time_mu_slope[i, p] + (time_sigma_slope[i, p] // 2))
                                o = otu_wts[i, p]
                                sel_otus = [self.var_names[self.otu_idx[l]] for l, ot in enumerate(o) if ot >= 0.9]
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

                                    sel_nodes = list()
                                    fca_copy = deepcopy(fca)
                                    for n in fca.get_leaves():
                                        if n.name in sel_otus:
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
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{} {}'.format(n.name, node_name)
                                            text = TextFace(tw.fill(text=node_name), fsize=30)
                                            n.add_face(text, 0)
                                        else:
                                            n.delete()

                                    for n in fca_copy.get_leaves():
                                        if n.name in sel_otus:
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
                                                if ',including' in node_name:
                                                    node_name = node_name.replace(',including', '').split(" ")[:2]
                                                    node_name = ' '.join(map(str, node_name))
                                                node_name = '{} {}'.format(n.name, node_name)
                                            n.name = node_name
                                        else:
                                            n.delete()


                                    ts = TreeStyle()
                                    ts.show_leaf_name = False
                                    ts.show_branch_length = False
                                    ts.branch_vertical_margin = 10
                                    ts.min_leaf_separation = 10
                                    ts.show_scale = False
                                    fca.render('{}/r{}d{}_subtree_ete3_{}.png'.format(
                                        dirName, i, p, self.args.data_name), dpi=1200,
                                        tree_style=ts)
                                    fca_copy.write(outfile='{}/r{}d{}_subtree_slope.newick'.format(
                                        dirName, i, p))

                                    if self.args.save_as_csv:
                                        column_names = ['Subject', 'Label', *['day_{}'.format(day) for day in np.arange(self.num_time)]]
                                        csv_det_df = pd.DataFrame(columns=column_names)
                                    fig = plt.figure()
                                    fig.subplots_adjust(top=0.8)
                                    gs = fig.add_gridspec(2, 2)
                                    f_ax1 = fig.add_subplot(gs[0, 0])
                                    f_ax2 = fig.add_subplot(gs[1, 0], sharex=f_ax1)
                                    mean_0 = list()
                                    mean_1 = list()
                                    slope_0 = list()
                                    slope_1 = list()
                                    for k in range(self.num_subjects):
                                        t_abun = x_time_slope[k, i, p]
                                        if labels[k]:
                                            slope_1.append(t_abun)
                                        else:
                                            slope_0.append(t_abun)
                                    for k in range(self.num_subjects):
                                        abun = x_spat[k, i, p]
                                        if labels[k]:
                                            lines_1, = f_ax1.plot(self.times[k].astype('int'),
                                                abun[self.times[k].astype('int')],
                                                marker='.', color='g')
                                            mean_1.append(abun[self.times[k].astype('int')].mean())
                                        else:
                                            lines_0, = f_ax2.plot(self.times[k].astype('int'),
                                                abun[self.times[k].astype('int')],
                                                marker='.', color='#FF8C00')
                                            mean_0.append(abun[self.times[k].astype('int')].mean())
                                        if self.args.save_as_csv:
                                            for day in range(self.num_time):
                                                if not day in self.times[k].astype('int'):
                                                    abun[day] = -1.
                                            csv_det_df.loc[len(csv_det_df.index)] = [k, labels[k], *abun]
                                    f_ax1.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    sel_days = np.arange(t_min, t_max + 1)
                                    line_thresh, = f_ax1.plot(sel_days,
                                        thresh * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_0),
                                        c='k',
                                        linestyle='--', linewidth=3)
                                    line_slope_1, = f_ax1.plot(sel_days,
                                        np.median(slope_1) * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_1),
                                        c='r',
                                        linestyle='--', linewidth=3)
                                    f_ax1.set_ylabel('Abundance')
                                    f_ax1.set_title(self.label_1)
                                    plt.setp(f_ax1.get_xticklabels(), visible=False)
                                    f_ax2.axvspan(t_min, t_max, facecolor='0.5', alpha=0.4, label='Time window')
                                    line_thresh, = f_ax2.plot(sel_days,
                                        thresh * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_0),
                                        c='k',
                                        linestyle='--', linewidth=3)
                                    line_slope_2, = f_ax2.plot(sel_days,
                                        np.median(slope_0) * (sel_days - ((t_min + t_max + 1) / 2.)) + np.mean(mean_0),
                                        c='r',
                                        linestyle='--', linewidth=3)
                                    f_ax2.set_xlabel('Days')
                                    f_ax2.set_ylabel('Abundance')
                                    f_ax2.set_title(self.label_0)
                                    rule_eng = self.get_rule(fc_wts[i], fc_bias, i, p, t_min, t_max, thresh)
                                    plt.suptitle(rule_eng, y=0.98, wrap=True)
                                    tree_img = mpimg.imread('{}/r{}d{}_subtree_ete3_{}.png'.format(
                                        dirName, i, p, self.args.data_name))
                                    f_ax3 = fig.add_subplot(gs[:, 1])
                                    imgplot = plt.imshow(tree_img)
                                    f_ax3.set_axis_off()
                                    pdf.savefig(fig, bbox_inches='tight', dpi=1200)  # saves the current figure into a pdf page
                                    os.remove('{}/r{}d{}_subtree_ete3_{}.png'.format(
                                        dirName, i, p, self.args.data_name))
                                    plt.close()

                                    if self.args.save_as_csv:
                                        rule_path = '{}/rule_{}_detector_{}_slope.csv'.format(dirName, i, p)
                                        csv_df.loc[len(csv_df.index)] = [i, p, 'slope', rule_path, otu_annot_str, kappa, t_min, t_max,\
                                        'N/A', thresh, np.median(slope_0), np.median(slope_1), fc_wts[i], fc_bias]
                                        csv_det_df.to_csv(rule_path, index=False)


            if self.args.save_as_csv:
                csv_df.to_csv('{}/rules_dump.csv'.format(dirName), index=False)


if __name__ == '__main__':
    # Parse command line args
    args = parse()

    # Init trainer object
    trainer = Trainer(args)

    # Load data
    trainer.load_data()

    # run cv loop
    trainer.train_loop()
