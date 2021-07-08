import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
from ete3 import Tree, TreeStyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

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
            dist_matrix[i, j] = src.get_distance(dst, topology_only=False)
    return dist_matrix


def fasta_to_dict(filename):
    """ Read a mapping of IDs to sequences from a fasta file.
    For relabeling DADA2 RSVs. Returns a dict {sequence1: name1, ...}.
    """
    with open(filename) as f:
        whole = f.read()
    pairs = whole.split('\n>')
    table = dict()
    # remove leading '>'
    pairs[0] = pairs[0][1:]
    for pair in pairs:
        name, sequence = pair.strip().split('\n')
        table[name] = sequence
    return table


# Compute the data matrix, consisting of abundances of OTUs only
# Also compute a mask that contains ones for time points with samples
# else zero
def get_data_matrix(num_otus, num_time, samples, times, otu_idx, labels):
    num_subjects = len(samples)
    X = np.zeros((num_subjects, num_otus, num_time), dtype=np.float32)
    X_mask = np.ones((num_subjects, num_time), dtype=np.float32)
    sub_rem_ids = list()
    for i in range(num_subjects):
        x = samples[i]
        t = times[i].astype('int').tolist()
        if len(t) < 2:
            sub_rem_ids.append(i)
        for j in range(num_otus):
            for k in range(num_time):
                if k in t:
                    X[i, j, k] = x[otu_idx[j], t.index(k)]
                else:
                    X_mask[i, k] = 0.

    X = np.delete(X, sub_rem_ids, 0)
    X_mask = np.delete(X_mask, sub_rem_ids, 0)
    labels = np.delete(labels, sub_rem_ids, 0)
    times = np.delete(times, sub_rem_ids, 0)

    return X, X_mask, labels, times


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


class SynData:
    """synthetic data for testing the model code"""
    def __init__(self, num_subjects, num_time, sample_rate, n_variables, tree_dict=None):
        super(SynData, self).__init__()
        self.n_subjects = num_subjects
        self.num_time = num_time
        self.sample_rate = sample_rate
        self.n_variables = n_variables
        if tree_dict is not None:
            self.variable_tree = tree_dict['variable_tree']
            self.variable_names = [n.name for n in self.variable_tree.get_leaves()]
            self.variable_annotations = tree_dict['variable_annotations']
        else:
            self.variable_names = ['otu_{}'.format(i) for i in range(n_variables)]
            self.variable_tree = self.random_tree(self.variable_names)
            self.variable_annotations = dict()
        self.y = np.zeros(num_subjects, dtype=np.int64)
        self.y[(num_subjects + 1) // 2:] = 1
        self.T = np.tile(np.arange(0, num_time, 
            num_time // (int(num_time * sample_rate)), dtype=np.int64), (num_subjects, 1))
        self.experiment_start = 0
        self.experiment_end = num_time - 1

    def save_as_pickle(self, filename):
        syndata = dict()
        syndata['n_variables'] = self.n_variables
        syndata['variable_names'] = self.variable_names
        syndata['variable_tree'] = self.variable_tree
        syndata['X'] = self.X
        syndata['y'] = self.y
        syndata['T'] = self.T
        syndata['experiment_start'] = self.experiment_start
        syndata['experiment_end'] = self.experiment_end
        syndata['variable_annotations'] = self.variable_annotations

        with open(filename, 'wb') as f:
            pickle.dump(syndata, f)

    # Creates a random tree, given node names, taken from mitre code
    def random_tree(self, nodes, mean_log_distance=0, std_log_distance=1):
        working_nodes = [Tree(name=n) for n in nodes]
        internal_node_count = 0
        while len(working_nodes) > 1:
            left = working_nodes.pop(0)
            right = working_nodes.pop(0)
            new = Tree(name='node_{}'.format(internal_node_count))
            d1, d2 = np.exp(mean_log_distance + std_log_distance * np.random.randn(2))
            new.add_child(left, dist=d1)
            new.add_child(right, dist=d2)
            internal_node_count += 1
            working_nodes.append(new)
            self.variable_names.append(new.name)
            self.n_variables += 1
        return working_nodes[0]

    # Creates synthetic abundances for various test scenarios
    def create_syn_traj(self, case_id):
        # Different scenarios of rules to generate trajectories
        if case_id == 0:
            X = self.create_syn_traj_case_0()
        elif case_id == 1:
            X = self.create_syn_traj_case_1()
        elif case_id == 2:
            X = self.create_syn_traj_case_2()
        elif case_id == 3:
            X = self.create_syn_traj_case_3()
        elif case_id == 4:
            X = self.create_syn_traj_case_4()
        else:
            raise ValueError('Only 5 cases supported for synthetic data!')

        self.X = X

    # Subjects get a postive perturbation in a given time window
    # to OTUs belonging to a particular subtree. Abundances (of selected OTUs)
    # of subjects with outcome 1 increase at a higher rate than those of
    # subjects with outcome 0 in the selected time window. 
    # This is supposed to capture 1 rule 1 detector (slope type)
    def create_syn_traj_case_0(self):
        X = np.full((self.n_subjects, self.n_variables, self.T.shape[-1]), 5.)
        pert_window_start = int(self.T.shape[-1] / 3.)
        pert_window_end = int(2. * self.T.shape[-1] / 3.)
        for n in self.variable_tree.traverse():
            if not n.is_leaf() and len(n.get_leaves()) > 10 and len(n.get_leaves()) < 20:
                sub_tree = n
                break
        sub_tree.render('syndata_0_selected_subtree.pdf')
        selected_leaves = [self.variable_names.index(n.name) for n in sub_tree.get_leaves()]
        slope_0 = 1e-1
        slope_1 = 5e-1
        for i in range(self.n_subjects):
            if self.y[i]:
                for j in range(self.n_variables):
                    if j in selected_leaves:
                        for t in range(pert_window_start, pert_window_end):
                            X[i, j, t] = 10. + slope_1 * (t - ((pert_window_start + pert_window_end) // 2))
            else:
                for j in range(self.n_variables):
                    if j in selected_leaves:
                        for t in range(pert_window_start, pert_window_end):
                            X[i, j, t] = 10. + slope_0 * (t - ((pert_window_start + pert_window_end) // 2))

        self.selected_sub_tree = sub_tree

        X = X / X.sum(axis=1, keepdims=True)

        self.plot_abun_slope(X, self.y, selected_leaves, pert_window_start, pert_window_end, 0)

        return X

    # Subjects get a postive perturbation in a given time window
    # to OTUs belonging to a particular subtree. Abundances (of selected OTUs)
    # of subjects with outcome 1 are higher than those of
    # subjects with outcome 0 in the selected time window, but increase at the same rate. 
    # This is supposed to capture 1 rule 1 detector (agg. abundance type)
    def create_syn_traj_case_1(self):
        X = np.full((self.n_subjects, self.n_variables, self.T.shape[-1]), 5.)
        pert_window_start = 0
        pert_window_end = self.T.shape[-1]
        for n in self.variable_tree.traverse():
            if not n.is_leaf() and len(n.get_leaves()) > 10 and len(n.get_leaves()) < 20:
                sub_tree = n
                break
        sub_tree.render('syndata_1_selected_subtree.pdf')
        selected_leaves = [self.variable_names.index(n.name) for n in sub_tree.get_leaves()]
        pert_1 = 10.
        for i in range(self.n_subjects):
            if self.y[i]:
                for j in range(self.n_variables):
                    if j in selected_leaves:
                        for t in range(pert_window_start, pert_window_end):
                            X[i, j, t] += pert_1

        self.selected_sub_tree = sub_tree

        X = X / X.sum(axis=1, keepdims=True)

        self.plot_abun(X, self.y, selected_leaves, pert_window_start, pert_window_end, 1)

        return X

    def create_syn_traj_case_2(self):
        X = np.full((self.n_subjects, self.n_variables, self.T.shape[-1]), 5.)
        pert_window_start_slope = int(2 * self.T.shape[-1] // 3)
        pert_window_end_slope = self.T.shape[-1]
        pert_window_start = 0
        pert_window_end = int(self.T.shape[-1] // 3)
        slope_0 = 1e-1
        slope_1 = 5e-1
        sub_ids_0_abun, sub_ids_0_slope = np.array_split(np.arange(self.n_subjects)[self.y == 0], 2)
        for n in self.variable_tree.traverse():
            if not n.is_leaf() and len(n.get_leaves()) > 10 and len(n.get_leaves()) < 20:
                sub_tree = n
                break
        sub_tree.render('syndata_2_selected_subtree_abun.pdf')
        selected_leaves_abun = [self.variable_names.index(n.name) for n in sub_tree.get_leaves()]
        for n in self.variable_tree.traverse():
            if not n.is_leaf() and len(n.get_leaves()) > 10 and len(n.get_leaves()) < 20:
                if sub_tree.name != n.name:
                    sub_tree_slope = n
                    break
        sub_tree_slope.render('syndata_2_selected_subtree_slope.pdf')
        selected_leaves_slope = [self.variable_names.index(n.name) for n in sub_tree_slope.get_leaves()]

        for i in range(self.n_subjects):
            for j in range(self.n_variables):
                if self.y[i]:
                    if j in selected_leaves_slope:
                        for t in range(pert_window_start_slope, pert_window_end_slope):
                            X[i, j, t] = 10. + slope_1 * (t - ((pert_window_start_slope + pert_window_end_slope) // 2))
                else:
                    if j in selected_leaves_slope:
                        if i in sub_ids_0_slope:
                            for t in range(pert_window_start_slope, pert_window_end_slope):
                                X[i, j, t] = 10. + slope_1 * (t - ((pert_window_start_slope + pert_window_end_slope) // 2))


        pert = 10.
        for i in range(self.n_subjects):
            if self.y[i]:
                for j in range(self.n_variables):
                    if j in selected_leaves_abun:
                        for t in range(pert_window_start, pert_window_end):
                            X[i, j, t] += pert
            else:
                for j in range(self.n_variables):
                    if j in selected_leaves_abun:
                        if i in sub_ids_0_abun:
                            for t in range(pert_window_start, pert_window_end):
                                X[i, j, t] += pert


        self.selected_sub_tree_abun = sub_tree
        self.selected_sub_tree_slope = sub_tree_slope

        X = X / X.sum(axis=1, keepdims=True)

        self.plot_abun_slope_both(X, self.y, selected_leaves_abun, pert_window_start, pert_window_end,
            selected_leaves_slope, pert_window_start_slope, pert_window_end_slope, 2)

        return X

    def create_syn_traj_case_3(self):
        X = np.full((self.n_subjects, self.n_variables, self.T.shape[-1]), 5.)
        pert_window_start_slope = int(2 * self.T.shape[-1] // 3)
        pert_window_end_slope = self.T.shape[-1]
        pert_window_start = 0
        pert_window_end = int(self.T.shape[-1] / 3.)
        slope_0 = 1e-1
        slope_1 = 5e-1
        sub_ids_0_abun, sub_ids_0_slope = np.array_split(np.arange(self.n_subjects)[self.y == 1], 2)
        for n in self.variable_tree.traverse():
            if not n.is_leaf() and len(n.get_leaves()) > 10 and len(n.get_leaves()) < 20:
                sub_tree = n
                break
        sub_tree.render('syndata_3_selected_subtree_abun.pdf')
        selected_leaves_abun = [self.variable_names.index(n.name) for n in sub_tree.get_leaves()]
        for n in self.variable_tree.traverse():
            if not n.is_leaf() and len(n.get_leaves()) > 10 and len(n.get_leaves()) < 20:
                if sub_tree.name != n.name:
                    sub_tree_slope = n
                    break
        sub_tree_slope.render('syndata_3_selected_subtree_slope.pdf')
        selected_leaves_slope = [self.variable_names.index(n.name) for n in sub_tree_slope.get_leaves()]

        for i in range(self.n_subjects):
            for j in range(self.n_variables):
                if i in sub_ids_0_slope:
                    if j in selected_leaves_slope:
                        for t in range(pert_window_start_slope, pert_window_end_slope):
                            X[i, j, t] = 10. + slope_1 * (t - ((pert_window_start_slope + pert_window_end_slope) // 2))


        pert = 10.
        for i in range(self.n_subjects):
            if i in sub_ids_0_abun:
                for j in range(self.n_variables):
                    if j in selected_leaves_abun:
                        for t in range(pert_window_start, pert_window_end):
                            X[i, j, t] += pert


        self.selected_sub_tree_abun = sub_tree
        self.selected_sub_tree_slope = sub_tree_slope

        X = X / X.sum(axis=1, keepdims=True)

        self.plot_abun_slope_both(X, self.y, selected_leaves_abun, pert_window_start, pert_window_end,
            selected_leaves_slope, pert_window_start_slope, pert_window_end_slope, 3)

        return X

    def plot_abun_slope_both(self, x, y, s_l, t_s, t_e, s_l_s, t_s_s, t_e_s, r_id):
        with PdfPages('syndata_case_{}.pdf'.format(r_id)) as pdf:
            ts = TreeStyle()
            ts.show_branch_length = False
            ts.branch_vertical_margin = 10
            ts.min_leaf_separation = 10
            ts.show_scale = False
            self.selected_sub_tree_abun.render(
                'syndata_case_{}_abun_subtree.png'.format(r_id),
                dpi=1200,
                tree_style=ts)

            fig = plt.figure()
            fig.subplots_adjust(top=0.8)
            gs = fig.add_gridspec(2, 2)
            f_ax1 = fig.add_subplot(gs[0, 0])
            f_ax2 = fig.add_subplot(gs[1, 0])
            mean_0 = list()
            mean_1 = list()
            for k in range(self.n_subjects):
                abun = 0.
                for l in s_l:
                    abun += x[k, l]
                if y[k]:
                    lines_1, = f_ax1.plot(self.T[k],
                        abun[self.T[k]],
                        marker='.', color='g')
                    mean_1.append(np.mean(abun, axis=-1))
                else:
                    lines_0, = f_ax2.plot(self.T[k],
                        abun[self.T[k]],
                        marker='.', color='#FF8C00')
                    mean_0.append(np.mean(abun, axis=-1))

            f_ax1.axvspan(t_s, t_e, facecolor='0.5', alpha=0.4, label='Time window')
            sel_days = np.arange(t_s, t_e + 1)
            line_thresh_1 = f_ax1.axhline(y=np.median(mean_1),
                xmin=((t_s) / self.num_time),
                xmax=((t_e + 1) / self.num_time),
                c='r', linestyle='--', linewidth=3)
            f_ax1.set_ylabel('Abundance')
            f_ax1.set_title('Outcome 1')
            plt.setp(f_ax1.get_xticklabels(), visible=False)
            f_ax2.axvspan(t_s, t_e, facecolor='0.5', alpha=0.4, label='Time window')
            line_thresh_0 = f_ax2.axhline(y=np.median(mean_0),
                xmin=((t_s) / self.num_time),
                xmax=((t_e + 1) / self.num_time),
                c='r', linestyle='--', linewidth=3)
            f_ax2.set_xlabel('Days')
            f_ax2.set_ylabel('Abundance')
            f_ax2.set_title('Outcome 0')
            tree_img = mpimg.imread('syndata_case_{}_abun_subtree.png'.format(r_id))
            f_ax3 = fig.add_subplot(gs[:, 1])
            imgplot = plt.imshow(tree_img)
            f_ax3.set_axis_off()
            plt.suptitle('Median (agg.) abund. Outcome 0: {:.5f} Outcome 1: {:.5f}'.format(np.median(mean_0), np.median(mean_1)))
            pdf.savefig(fig, dpi=1200)
            plt.close()


            ts = TreeStyle()
            ts.show_branch_length = False
            ts.branch_vertical_margin = 10
            ts.min_leaf_separation = 10
            ts.show_scale = False
            self.selected_sub_tree_slope.render(
                './syndata_case_{}_slope_subtree.png'.format(r_id),
                dpi=1200,
                tree_style=ts)

            fig = plt.figure()
            fig.subplots_adjust(top=0.8)
            gs = fig.add_gridspec(2, 2)
            f_ax1 = fig.add_subplot(gs[0, 0])
            f_ax2 = fig.add_subplot(gs[1, 0])
            mean_0 = list()
            mean_1 = list()
            slope_0 = list()
            slope_1 = list()
            tau = np.arange(t_s_s, t_e_s) - ((t_s_s + t_e_s) // 2)
            for k in range(self.n_subjects):
                abun = 0.
                for l in s_l_s:
                    abun += x[k, l]
                if y[k]:
                    lines_1, = f_ax1.plot(self.T[k],
                        abun[self.T[k]],
                        marker='.', color='g')
                    mean_1.append(np.mean(abun, axis=-1))
                    slope_1.append(np.polyfit(tau, abun[t_s_s:t_e_s], 1)[0])
                else:
                    lines_0, = f_ax2.plot(self.T[k],
                        abun[self.T[k]],
                        marker='.', color='#FF8C00')
                    mean_0.append(np.mean(abun, axis=-1))
                    slope_0.append(np.polyfit(tau, abun[t_s_s:t_e_s], 1)[0])

            f_ax1.axvspan(t_s_s, t_e_s, facecolor='0.5', alpha=0.4, label='Time window')
            sel_days = np.arange(t_s_s, t_e_s + 1)
            line_slope_1, = f_ax1.plot(sel_days,
                np.median(slope_1) * (sel_days - ((t_s_s + t_e_s) / 2.)) + np.mean(mean_1),
                c='r',
                linestyle='--', linewidth=3)
            f_ax1.set_ylabel('Abundance')
            f_ax1.set_title('Outcome 1')
            plt.setp(f_ax1.get_xticklabels(), visible=False)
            f_ax2.axvspan(t_s_s, t_e_s, facecolor='0.5', alpha=0.4, label='Time window')
            line_slope_2, = f_ax2.plot(sel_days,
                np.median(slope_0) * (sel_days - ((t_s_s + t_e_s) / 2.)) + np.mean(mean_0),
                c='r',
                linestyle='--', linewidth=3)
            f_ax2.set_xlabel('Days')
            f_ax2.set_ylabel('Abundance')
            f_ax2.set_title('Outcome 0')
            tree_img = mpimg.imread('./syndata_case_{}_slope_subtree.png'.format(r_id))
            f_ax3 = fig.add_subplot(gs[:, 1])
            imgplot = plt.imshow(tree_img)
            f_ax3.set_axis_off()
            plt.suptitle('Median slope Outcome 0: {:.5f} Outcome 1: {:.5f}'.format(np.median(slope_0), np.median(slope_1)))
            pdf.savefig(fig, dpi=1200)
            plt.close()

    def plot_abun(self, x, y, s_l, t_s, t_e, r_id):
        ts = TreeStyle()
        ts.show_branch_length = False
        ts.branch_vertical_margin = 10
        ts.min_leaf_separation = 10
        ts.show_scale = False
        self.selected_sub_tree.render(
            './syndata_case_{}_subtree.png'.format(r_id),
            dpi=1200,
            tree_style=ts)

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        gs = fig.add_gridspec(2, 2)
        f_ax1 = fig.add_subplot(gs[0, 0])
        f_ax2 = fig.add_subplot(gs[1, 0])
        mean_0 = list()
        mean_1 = list()
        for k in range(self.n_subjects):
            abun = 0.
            for l in s_l:
                abun += x[k, l]
            if y[k]:
                lines_1, = f_ax1.plot(self.T[k],
                    abun[self.T[k]],
                    marker='.', color='g')
                mean_1.append(np.mean(abun, axis=-1))
            else:
                lines_0, = f_ax2.plot(self.T[k],
                    abun[self.T[k]],
                    marker='.', color='#FF8C00')
                mean_0.append(np.mean(abun, axis=-1))

        f_ax1.axvspan(t_s, t_e, facecolor='0.5', alpha=0.4, label='Time window')
        sel_days = np.arange(t_s, t_e + 1)
        line_thresh_1 = f_ax1.axhline(y=np.median(mean_1),
            xmin=((t_s) / self.num_time),
            xmax=((t_e + 1) / self.num_time),
            c='r', linestyle='--', linewidth=3)
        f_ax1.set_ylabel('Abundance')
        f_ax1.set_title('Outcome 1')
        plt.setp(f_ax1.get_xticklabels(), visible=False)
        f_ax2.axvspan(t_s, t_e, facecolor='0.5', alpha=0.4, label='Time window')
        line_thresh_0 = f_ax2.axhline(y=np.median(mean_0),
            xmin=((t_s) / self.num_time),
            xmax=((t_e + 1) / self.num_time),
            c='r', linestyle='--', linewidth=3)
        f_ax2.set_xlabel('Days')
        f_ax2.set_ylabel('Abundance')
        f_ax2.set_title('Outcome 0')
        tree_img = mpimg.imread('./syndata_case_{}_subtree.png'.format(r_id))
        f_ax3 = fig.add_subplot(gs[:, 1])
        imgplot = plt.imshow(tree_img)
        f_ax3.set_axis_off()
        plt.suptitle('Median (agg.) abund. Outcome 0: {:.5f} Outcome 1: {:.5f}'.format(np.median(mean_0), np.median(mean_1)))
        plt.savefig('./syndata_case_{}_abun.pdf'.format(r_id), dpi=1200)
        plt.close()

    def plot_abun_slope(self, x, y, s_l, t_s, t_e, r_id):
        ts = TreeStyle()
        ts.show_branch_length = False
        ts.branch_vertical_margin = 10
        ts.min_leaf_separation = 10
        ts.show_scale = False
        self.selected_sub_tree.render(
            './syndata_case_{}_subtree.png'.format(r_id),
            dpi=1200,
            tree_style=ts)

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        gs = fig.add_gridspec(2, 2)
        f_ax1 = fig.add_subplot(gs[0, 0])
        f_ax2 = fig.add_subplot(gs[1, 0])
        mean_0 = list()
        mean_1 = list()
        slope_0 = list()
        slope_1 = list()
        tau = np.arange(t_s, t_e) - ((t_s + t_e) // 2)
        for k in range(self.n_subjects):
            abun = 0.
            for l in s_l:
                abun += x[k, l]
            if y[k]:
                lines_1, = f_ax1.plot(self.T[k],
                    abun[self.T[k]],
                    marker='.', color='g')
                mean_1.append(np.mean(abun, axis=-1))
                slope_1.append(np.polyfit(tau, abun[t_s:t_e], 1)[0])
            else:
                lines_0, = f_ax2.plot(self.T[k],
                    abun[self.T[k]],
                    marker='.', color='#FF8C00')
                mean_0.append(np.mean(abun, axis=-1))
                slope_0.append(np.polyfit(tau, abun[t_s:t_e], 1)[0])

        f_ax1.axvspan(t_s, t_e, facecolor='0.5', alpha=0.4, label='Time window')
        sel_days = np.arange(t_s, t_e + 1)
        line_slope_1, = f_ax1.plot(sel_days,
            np.median(slope_1) * (sel_days - ((t_s + t_e) / 2.)) + np.mean(mean_1),
            c='r',
            linestyle='--', linewidth=3)
        f_ax1.set_ylabel('Abundance')
        f_ax1.set_title('Outcome 1')
        plt.setp(f_ax1.get_xticklabels(), visible=False)
        f_ax2.axvspan(t_s, t_e, facecolor='0.5', alpha=0.4, label='Time window')
        line_slope_2, = f_ax2.plot(sel_days,
            np.median(slope_0) * (sel_days - ((t_s + t_e) / 2.)) + np.mean(mean_0),
            c='r',
            linestyle='--', linewidth=3)
        f_ax2.set_xlabel('Days')
        f_ax2.set_ylabel('Abundance')
        f_ax2.set_title('Outcome 0')
        tree_img = mpimg.imread('./syndata_case_{}_subtree.png'.format(r_id))
        f_ax3 = fig.add_subplot(gs[:, 1])
        imgplot = plt.imshow(tree_img)
        f_ax3.set_axis_off()
        plt.suptitle('Median slope Outcome 0: {:.5f} Outcome 1: {:.5f}'.format(np.median(slope_0), np.median(slope_1)))
        plt.savefig('./syndata_case_{}_abun.pdf'.format(r_id), dpi=1200)
        plt.close()


if __name__ == '__main__':
    dataset = load_from_pickle('./datasets/david_agg_filtered.pickle')
    tree_dict = {
        'variable_tree': dataset['variable_tree'],
        'variable_annotations': dataset['variable_annotations']
    }

    case_id = 3
    syndata = SynData(20, 50, 1, len(dataset['variable_tree']), tree_dict=tree_dict)
    syndata.create_syn_traj(case_id)
    syndata.save_as_pickle('./datasets/syndata_{}_davtree_rel.pkl'.format(case_id))