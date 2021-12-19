import matplotlib.pyplot as plt
plt.rcParams.update({
	"font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import matplotlib.colors as mc
from matplotlib.cm import ScalarMappable
from matplotlib.legend_handler import HandlerTuple
import matplotlib.transforms as mtransforms
from matplotlib.widgets import Button

import seaborn as sns
from ete3 import TreeStyle, Tree

import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

from mditre.visualize import *



class RuleVisualizer:
	"""docstring for RuleVisualizer"""
	def __init__(self, rule_dict, rule_dict_path):
		super(RuleVisualizer, self).__init__()
		self.rule_dict_path = rule_dict_path
		self.log_odds = rule_dict['log_odds']
		self.subjects_0_log_odds = np.array(rule_dict['subjects_0_log_odds'])
		self.subjects_1_log_odds = np.array(rule_dict['subjects_1_log_odds'])
		self.num_subjects = len(self.subjects_0_log_odds) + len(self.subjects_1_log_odds)
		self.labels = np.concatenate([np.zeros((len(self.subjects_0_log_odds))), np.ones((len(self.subjects_1_log_odds)))])
		self.sub_0_log_odds = np.array(rule_dict['sub_0_log_odds'])
		self.sub_1_log_odds = np.array(rule_dict['sub_1_log_odds'])
		self.label_0 = rule_dict['outcome_0']
		self.label_1 = rule_dict['outcome_1']
		self.sel_taxa = rule_dict['taxa_tree']
		self.thresh = rule_dict['thresh']
		self.t_min = rule_dict['t_min']
		self.t_max = rule_dict['t_max']
		self.new_t_min = rule_dict['new_t_min']
		self.new_t_max = rule_dict['new_t_max']
		self.det_type = rule_dict['det_type']
		self.x_0 = rule_dict['x_0']
		self.x_1 = rule_dict['x_1']
		self.mask_0 = rule_dict['x_mask_0']
		self.mask_1 = rule_dict['x_mask_1']
		self.x_avg_0 = rule_dict['x_avg_0']
		self.x_avg_1 = rule_dict['x_avg_1']
		self.time_unique = rule_dict['time_unique']
		self.full_tree = rule_dict['full_tree']
		self.rule_str = dict()

		self.filtered_rule_ids = list()
		for i in range(len(self.log_odds)):
			num_taxa = self.sel_taxa[i]
			if len(num_taxa) > 0:
				self.filtered_rule_ids.append(i)

		self.num_rules = len(self.filtered_rule_ids)
		self.sub_ids = np.linspace(0, self.num_subjects, int(0.25 * self.num_subjects))

	def get_detector(self, d, t_min, t_max, thresh, metric='slope'):
		if metric == 'slope':
			det = "Detector {}: the average slope of selected taxa\nbetween days {} to {} is greater than {:.4f}% per day".format(
				d, t_min, t_max, thresh * 1e2)
		else:
			det = "Detector {}: the average abundance of selected taxa\nbetween days {} to {} is greater than {:.4f}%".format(
				d, t_min, t_max, thresh * 1e2)
		return det

	def plot_log_odds(self):
		fig = plt.figure()
		gs = fig.add_gridspec(3 + self.num_rules, 3,
			width_ratios=[1e-1 * self.num_subjects, self.num_subjects, 1e-2 * self.num_subjects])

		cmap_0 = mc.LinearSegmentedColormap.from_list('cmap_0', ['#4833FF', '#33ADFF', '#33FDFF'])
		cmap_1 = mc.LinearSegmentedColormap.from_list('cmap_1', ['#F3FF33', '#FF9D33', '#FF3A33'])
		newcolors = np.vstack((cmap_0(np.linspace(0, 1, 256)),
							   cmap_1(np.linspace(0, 1, 256))))
		cmap = mc.ListedColormap(newcolors)
		divnorm = mc.TwoSlopeNorm(vmin=min(self.log_odds), vcenter=0, vmax=max(self.log_odds))

		ax_0 = fig.add_subplot(gs[-2, 1], label='combined_rule_logodds')
		cbar_ax = fig.add_subplot(gs[:-1, 2], label='rule_cbar')
		v = np.concatenate([self.subjects_0_log_odds, self.subjects_1_log_odds])
		v_sorted_ids = np.argsort(v)
		labels = self.labels
		h1 = sns.heatmap(v[np.newaxis, :], ax=ax_0,
			cmap=cmap, norm=divnorm, square=True,
			cbar_ax=cbar_ax,
			cbar_kws={
				'orientation': "vertical",
			})
		h1.collections[0].colorbar.set_label('Log-odds')
		ax_0.set_title('Log-odds of all rules')
		ax_0.set_xlabel('Subjects')
		ax_0.set_yticks([])

		j = 1
		for i in range(self.num_rules):
			rule_id = self.filtered_rule_ids[i]
			ax_0 = fig.add_subplot(gs[j, 1], label='rule_{}'.format(j))
			v = np.concatenate([self.sub_0_log_odds[:, rule_id], self.sub_1_log_odds[:, rule_id]])
			h1 = sns.heatmap(v[np.newaxis, :], ax=ax_0,
				cmap=cmap, norm=divnorm, cbar=False, square=True)
			ax_0.set_title('Log-odds of rule: {}'.format(i + 1))
			ax_0.set_xlabel('Subjects')
			ax_0.set_yticks([])

			ax_0 = fig.add_subplot(gs[j, 0], label='rule_{}_button'.format(rule_id))
			ax_0.set_box_aspect(1)
			b = Button(ax_0, 'Click')

			j += 1


		v = np.array(self.log_odds[np.array(self.filtered_rule_ids)])
		ax_0 = fig.add_subplot(gs[0, 1], label='rule_all_logodds')
		h1 = sns.heatmap(v[np.newaxis, :], ax=ax_0,
				cmap=cmap, norm=divnorm, cbar=False, square=True,
				linewidths=2)
		ax_0.set_title('Log-odds of rules')
		ax_0.set_yticks([])
		ax_0.set_xticklabels(np.arange(1, self.num_rules + 1))
		ax_0.set_xlabel('Rules')

		ax_0 = fig.add_subplot(gs[-1, 1], label='lab')
		cbar_ax = fig.add_subplot(gs[-1, 2])
		cmap = mc.ListedColormap(['grey', 'green'])
		h1 = sns.heatmap(labels[np.newaxis, :], ax=ax_0,
			cmap=cmap, square=True,
			cbar_ax=cbar_ax,
			cbar_kws={
				'orientation': "vertical",
			})
		h1.collections[0].colorbar.set_ticks([0.25, 0.75])
		h1.collections[0].colorbar.set_ticklabels([self.label_0, self.label_1])
		ax_0.set_title('Ground truth labels of subjects')
		ax_0.set_xlabel('Subjects')
		ax_0.set_yticks([])

		fig.canvas.mpl_connect('button_press_event', self.on_click_rule)
		gs.tight_layout(fig)
		plt.show()

	def on_click_rule(self, event):
		if event.inaxes is not None:
			rule_id = int(event.inaxes._label.split('_')[1])
			self.show_rule_det(rule_id)


	def show_rule_det(self, rule_id):
		lodds = self.log_odds[rule_id]
		num_det = len(self.sel_taxa[rule_id])

		if lodds > 0:
			outcome = self.label_1
		else:
			outcome = self.label_0
		if num_det > 0:
			rule_str = ("Rule {}: TRUE for {} with log-odds {:.2f}, IF:").format(
				1 + self.filtered_rule_ids.index(rule_id), outcome, lodds)
			num_plots = 2 * num_det
		else:
			rule_str = ("Rule {}: TRUE for {} with log-odds {:.2f}, IF no detectors are active.").format(
				1 + self.filtered_rule_ids.index(rule_id), outcome, lodds)
			num_plots = 1

		fig = plt.figure()
		if num_det > 1:
			gs = fig.add_gridspec(3 + num_det, 3,
				width_ratios=[1e-1 * self.num_subjects, self.num_subjects, 1e-2 * self.num_subjects])
		else:
			gs = fig.add_gridspec(3, 3,
				width_ratios=[1e-1 * self.num_subjects, self.num_subjects, 1e-2 * self.num_subjects])

		if num_det > 1:
			x_mean_0 = np.array(self.x_avg_0[rule_id]).T * 1e2
			x_mean_1 = np.array(self.x_avg_1[rule_id]).T * 1e2
			thresh = np.array(self.thresh[rule_id]) * 1e2
			det_0 = np.prod((x_mean_0 - thresh) > 0, axis=-1)
			det_1 = np.prod((x_mean_1 - thresh) > 0, axis=-1)
		else:
			x_mean_0 = np.array(self.x_avg_0[rule_id][0]).T * 1e2
			x_mean_1 = np.array(self.x_avg_1[rule_id][0]).T * 1e2
			thresh = np.array(self.thresh[rule_id][0]) * 1e2
			det_0 = (x_mean_0 - thresh) > 0
			det_1 = (x_mean_1 - thresh) > 0

		if lodds < 0:
			det_0 = np.logical_not(det_0)
			det_1 = np.logical_not(det_1)
		
		v = np.concatenate([det_0, det_1])
		v_sorted_ids = np.argsort(v)
		labels = self.labels

		cmap_0 = mc.LinearSegmentedColormap.from_list('cmap_0', ['#4833FF', '#33ADFF', '#33FDFF'])
		cmap_1 = mc.LinearSegmentedColormap.from_list('cmap_1', ['#F3FF33', '#FF9D33', '#FF3A33'])
		newcolors = np.vstack((cmap_0(np.linspace(0, 1, 256)),
							   cmap_1(np.linspace(0, 1, 256))))
		cmap = mc.ListedColormap(newcolors)
		divnorm = mc.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

		if num_det > 1:
			ax_0 = fig.add_subplot(gs[-2, 1], label='00')
			cbar_ax = fig.add_subplot(gs[1:-1, 2])
			h1 = sns.heatmap(v[np.newaxis, :], ax=ax_0,
				cmap=cmap, square=True, norm=divnorm,
				cbar_ax=cbar_ax,
				cbar_kws={
					'orientation': "vertical",
				})
			h1.collections[0].colorbar.set_label('Detector activation')
			ax_0.set_title('Conjunction of all detectors')
			ax_0.set_xlabel('Subjects')
			ax_0.set_yticks([])


		for i in range(num_det):
			x_mean_0 = self.x_avg_0[rule_id][i] * 1e2
			x_mean_1 = self.x_avg_1[rule_id][i] * 1e2
			thresh = self.thresh[rule_id][i] * 1e2

			det_0 = (x_mean_0 - thresh) > 0
			det_1 = (x_mean_1 - thresh) > 0

			ax_0 = fig.add_subplot(gs[i + 1, 1], label=str(i))
			v = np.concatenate([det_0, det_1])
			if num_det == 1:
				cbar_ax = fig.add_subplot(gs[1:-1, 2])
				h1 = sns.heatmap(v[np.newaxis, :], ax=ax_0,
					cmap=cmap, square=True, norm=divnorm,
					cbar_ax=cbar_ax,
					cbar_kws={
						'orientation': "vertical",
					})
				h1.collections[0].colorbar.set_label('Detector activation')
			else:
				h1 = sns.heatmap(v[np.newaxis, :], ax=ax_0,
					cmap=cmap, square=True, norm=divnorm,
					cbar=False)
			ax_0.set_title('Activation of detector id: {}'.format(i+1))
			ax_0.set_xlabel('Subjects')
			ax_0.set_yticks([])

			ax_0 = fig.add_subplot(gs[i+1, 0], label='det_{}_button'.format(i))
			ax_0.set_box_aspect(1)
			b = Button(ax_0, 'Click')


		cmap = mc.ListedColormap(['grey', 'green'])
		ax_0 = fig.add_subplot(gs[-1, 1], label='l')
		cbar_ax = fig.add_subplot(gs[-1, 2])
		h1 = sns.heatmap(labels[np.newaxis, :], ax=ax_0,
			cmap=cmap, square=True,
			cbar_ax=cbar_ax,
			cbar_kws={
				'orientation': "vertical",
			})
		h1.collections[0].colorbar.set_ticks([0.25, 0.75])
		h1.collections[0].colorbar.set_ticklabels([self.label_0, self.label_1])
		ax_0.set_title('Ground truth labels of subjects')
		ax_0.set_xlabel('Subjects')
		ax_0.set_yticks([])


		gs00 = gridspec.GridSpecFromSubplotSpec(num_plots, 1, subplot_spec=gs[0, :])
		axes = fig.add_subplot(gs00[0, :])
		axes.text(0.5, 0.5, rule_str, ha='center', va='center', wrap=True, transform=axes.transAxes,
		)
		axes.set_axis_off()

		if num_det > 0:
			j = 0
			self.rule_str['rule_{}'.format(rule_id)] = list()
			for i in range(1, num_plots):
				axes = fig.add_subplot(gs00[i, :], label=str(i))
				if i % 2 == 0:
					axes.text(0.5, 0.5, 'AND', ha='center', va='center', wrap=True, transform=axes.transAxes,
					)
				else:
					t_min = self.t_min[rule_id][j]
					t_max = self.t_max[rule_id][j]
					thresh = self.thresh[rule_id][j]
					det_type = self.det_type[rule_id][j]
					det_str = self.get_detector(j + 1, t_min, t_max, thresh, metric=det_type)
					axes.text(0.5, 0.5, det_str, ha='center', va='center', wrap=True,
						transform=axes.transAxes,
						bbox=dict(boxstyle="square", fill=False),
					)
					j += 1
				axes.set_axis_off()

				self.rule_str['rule_{}'.format(rule_id)].append(det_str)


		fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_text(event, rule_id))

		gs.tight_layout(fig)
		plt.show(block=False)

	def show_rule(self, rule_id):
		num_det = len(self.sel_taxa[rule_id])
		lodds = self.log_odds[rule_id]
		if lodds > 0:
			outcome = self.label_1
		else:
			outcome = self.label_0
		if num_det > 0:
			rule_str = ("Rule {}: TRUE for {} with log-odds {:.2f}, IF:").format(rule_id, outcome, lodds)
			num_plots = 2 * num_det
		else:
			rule_str = ("Rule {}: TRUE for {} with log-odds {:.2f}, IF no detectors are active.").format(rule_id, outcome, lodds)
			num_plots = 1

		fig, axes = plt.subplots(1, 1, constrained_layout=True)
		axes.text(0.5, 0.8, rule_str, ha='center', va='center', wrap=True, transform=axes.transAxes)
		axes.set_axis_off()

		if num_det > 0:
			j = 0
			for i in range(1, num_plots):
				if i % 2 == 0:
					axes.text(0.5, 0.8 - i * 0.1, 'AND', ha='center', va='center', wrap=True, transform=axes.transAxes)
				else:
					t_min = self.t_min[rule_id][j]
					t_max = self.t_max[rule_id][j]
					thresh = self.thresh[rule_id][j]
					det_type = self.det_type[rule_id][j]
					det_str = self.get_detector(j, t_min, t_max, thresh, metric=det_type)
					axes.text(0.5, 0.8 - i * 0.1, det_str, ha='center', va='center', wrap=True,
						transform=axes.transAxes, picker=True,
						bbox=dict(boxstyle="square", fill=False))
					j += 1
				axes.set_axis_off()

		fig.canvas.mpl_connect('pick_event', lambda event: self.on_click_text(event, rule_id))
		plt.show(block=False)

	def on_click_text(self, event, rule_id):
		if event.inaxes is not None:
			det_id = int(event.inaxes._label.split('_')[1])
			det_str = self.rule_str['rule_{}'.format(rule_id)][det_id]
			self.show_detector(det_id, rule_id, det_str)

	def show_detector(self, det_id, rule_id, det_str):
		x_0 = self.x_0[rule_id][det_id] * 1e2
		x_1 = self.x_1[rule_id][det_id] * 1e2
		mask_0 = self.mask_0[rule_id][det_id]
		mask_1 = self.mask_1[rule_id][det_id]
		x_mean_0 = self.x_avg_0[rule_id][det_id] * 1e2
		x_mean_1 = self.x_avg_1[rule_id][det_id] * 1e2

		x_mean_0_sorted_ids = np.argsort(x_mean_0)
		x_mean_1_sorted_ids = np.argsort(x_mean_1)
		x_0 = x_0[x_mean_0_sorted_ids, :]
		x_1 = x_1[x_mean_1_sorted_ids, :]
		mask_0 = mask_0[x_mean_0_sorted_ids, :]
		mask_1 = mask_1[x_mean_1_sorted_ids, :]
		x_mean_0 = x_mean_0[x_mean_0_sorted_ids]
		x_mean_1 = x_mean_1[x_mean_1_sorted_ids]

		tree = self.sel_taxa[rule_id][det_id]
		thresh = self.thresh[rule_id][det_id] * 1e2
		new_t_min = self.new_t_min[rule_id][det_id]
		new_t_max = self.new_t_max[rule_id][det_id]
		t_min = self.t_min[rule_id][det_id]
		t_max = self.t_max[rule_id][det_id]
		det_type = self.det_type[rule_id][det_id]
		phy_tree = self.full_tree[rule_id][det_id]
		phy_tree_names = phy_tree.get_leaf_names()

		for n in tree.get_leaves():
			n_name = n.name
			for i, nn in enumerate(phy_tree_names):
				if n_name in nn:
					n.name = nn

		self.heatmap_rule_viz(x_0, x_1, x_mean_0, x_mean_1,
			mask_0, mask_1, self.time_unique,
			tree, phy_tree, thresh, det_type, new_t_min, new_t_max,
			det_str, t_min, t_max, view_type=det_type)

	def heatmap_rule_viz(self, x_0, x_1, x_mean_0, x_mean_1,
		mask_0, mask_1, x_ticks,
		tree, phylo_tree, thresh, det_type, win_start, win_end,
		rule_eng, t_min, t_max, view_type=None):
		plt.rcParams["font.family"] = 'sans-serif'

		fig = plt.figure(figsize=(50, 50))
		fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.08)
		gs = fig.add_gridspec(2, 4,
			width_ratios=[x_0.shape[1], 0.02 * x_0.shape[1], 0.02 * x_0.shape[1], 0.25 * x_0.shape[1]],
			height_ratios=[x_0.shape[0], x_1.shape[0]])
		ax0 = fig.add_subplot(gs[0, 1-1])
		ax1 = fig.add_subplot(gs[0, 2-1])
		ax2 = fig.add_subplot(gs[1, 1-1])
		ax3 = fig.add_subplot(gs[1, 2-1])
		cbar_ax = fig.add_subplot(gs[:, 3-1])

		x_0_ma = np.ma.masked_array(x_0, mask=np.logical_not(mask_0))
		x_1_ma = np.ma.masked_array(x_1, mask=np.logical_not(mask_1))
		x_mean_0_ma = np.ma.masked_array(x_mean_0, mask=np.zeros_like(x_mean_0))
		x_mean_1_ma = np.ma.masked_array(x_mean_1, mask=np.zeros_like(x_mean_1))

		x_ma_con = np.ma.concatenate((x_0_ma, x_1_ma), axis=0)
		x_mean_ma_con = np.ma.concatenate((x_mean_0_ma, x_mean_1_ma), axis=0)
		x_con = np.ma.concatenate((x_ma_con, x_mean_ma_con[:, np.newaxis]), axis=1)
		x_min = np.ma.min(x_con)
		x_max = np.ma.max(x_con)

		
		# cmap_0 = plt.get_cmap('cool_r', 256)
		# cmap_1 = plt.get_cmap('autumn_r', 256)
		# newcolors = np.vstack((cmap_0(np.linspace(0, 1, 256)),
		#                        cmap_1(np.linspace(0, 1, 256))))
		# cmap = mc.ListedColormap(newcolors)
		cmap_0 = mc.LinearSegmentedColormap.from_list('cmap_0', ['#4833FF', '#33ADFF', '#33FDFF'])
		cmap_1 = mc.LinearSegmentedColormap.from_list('cmap_1', ['#F3FF33', '#FF9D33', '#FF3A33'])
		newcolors = np.vstack((cmap_0(np.linspace(0, 1, 256)),
							   cmap_1(np.linspace(0, 1, 256))))
		cmap = mc.ListedColormap(newcolors)
		divnorm = mc.TwoSlopeNorm(vmin=x_min, vcenter=thresh, vmax=x_max)

		h1 = sns.heatmap(x_0, ax=ax0, cmap=cmap,
			xticklabels=x_ticks, yticklabels=False,
			cbar=False,
			norm=divnorm,
			mask=np.logical_not(mask_0))
		# for i in range(x_0.shape[0]):
		#     for j in range(x_0.shape[1]):
		#         if mask_0[i, j]:
		#             h1.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='k', lw=0.5))
		h2 = sns.heatmap(x_1, ax=ax2, cmap=cmap,
			xticklabels=x_ticks, yticklabels=False,
			cbar=False,
			norm=divnorm,
			mask=np.logical_not(mask_1))
		# for i in range(x_1.shape[0]):
		#     for j in range(x_1.shape[1]):
		#         if mask_1[i, j]:
		#             h2.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='k', lw=0.5))
		h3 = sns.heatmap(x_mean_0[:, np.newaxis], ax=ax1, cmap=cmap,
					xticklabels=False, yticklabels=False,
				   cbar=False,
				   norm=divnorm,
				   )
		h4 = sns.heatmap(x_mean_1[:, np.newaxis], ax=ax3, cmap=cmap,
			xticklabels=False, yticklabels=False,
				   norm=divnorm,
				   cbar_ax=cbar_ax,
				   cbar_kws={
					   'orientation': "vertical",
				   },
				   )

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
		ax0.add_patch(Rectangle((win_start_idx, 0), win_end_idx - win_start_idx + 1, x_0.shape[0],
			alpha=0.7, fill=False, edgecolor='k', lw=6))
		ax0.text(ax2.get_xticks()[win_start_idx], -.05, str(t_min), color='red', transform=ax0.get_xaxis_transform(),
			ha='center', va='top', fontsize=20)
		ax0.text(ax2.get_xticks()[win_end_idx], -.05, str(t_max), color='red', transform=ax0.get_xaxis_transform(),
			ha='center', va='top', fontsize=20)
		t_step = int(0.20 * x_0.shape[1])
		for i in range(len(ax2.get_xticks())):
			if i != win_start_idx and i != win_end_idx:
				if i % t_step == 0:
					if np.absolute(i - win_start_idx) > 0.05*x_0.shape[1] and np.absolute(i - win_end_idx) > 0.05*x_0.shape[1]:
						if np.absolute(i - len(ax2.get_xticks())) > 0.05*x_0.shape[1]:
							ax0.text(ax2.get_xticks()[i], -.05, str(x_ticks[i]), color='black',
								transform=ax0.get_xaxis_transform(),
								ha='center', va='top', fontsize=15)
				elif i == len(ax2.get_xticks()) - 1:
					ax0.text(ax2.get_xticks()[i], -.05, str(x_ticks[i]), color='black',
						transform=ax0.get_xaxis_transform(),
						ha='center', va='top', fontsize=15)

		ax2.add_patch(Rectangle((win_start_idx, 0), win_end_idx - win_start_idx + 1, x_1.shape[0],
			alpha=0.7, fill=False, edgecolor='k', lw=6))
		ax2.text(ax2.get_xticks()[win_start_idx], -.05, str(t_min), color='red', transform=ax2.get_xaxis_transform(),
			ha='center', va='top', fontsize=20)
		ax2.text(ax2.get_xticks()[win_end_idx], -.05, str(t_max), color='red', transform=ax2.get_xaxis_transform(),
			ha='center', va='top', fontsize=20)
		for i in range(len(ax2.get_xticks())):
			if i != win_start_idx and i != win_end_idx:
				if i % t_step == 0:
					if np.absolute(i - win_start_idx) > 0.05*x_0.shape[1] and np.absolute(i - win_end_idx) > 0.05*x_0.shape[1]:
						if np.absolute(i - len(ax2.get_xticks())) > 0.05*x_0.shape[1]:
							ax2.text(ax2.get_xticks()[i], -.05, str(x_ticks[i]), color='black',
								transform=ax2.get_xaxis_transform(),
								ha='center', va='top', fontsize=15)
				elif i == len(ax2.get_xticks()) - 1:
					ax2.text(ax2.get_xticks()[i], -.05, str(x_ticks[i]), color='black',
						transform=ax2.get_xaxis_transform(),
						ha='center', va='top', fontsize=15)
		ax2.text(np.median(ax2.get_xticks()), -.3, s='Time (days)',
				horizontalalignment='center', fontsize=20,
				transform=ax2.get_xaxis_transform())
		ax0.set_ylabel('{}'.format(self.label_0), fontsize=20)
		ax2.set_ylabel('{}'.format(self.label_1), fontsize=20)
		if det_type.lower() == 'slope':
			ax0.set_title('Average slope over time', pad=30, fontsize=20)
		else:
			ax0.set_title('Abundances over time', pad=30, fontsize=20)
		ax0.set_xticks([])
		ax2.set_xticks([])

		cbar_ax.axhline(thresh, linewidth=6, color='black', alpha=0.7)
		y_pos = (thresh - x_min) / (x_max - x_min)
		ax1.set_title('Average {}\nover days {} to {}'.format(det_type, t_min, t_max),
			wrap=True, pad=30, fontsize=20)

		tree_ax = fig.add_subplot(gs[:, -1])
		t = plot_tree(tree, axe=tree_ax, font_size=12, show_names=True)
		tree_ax.set_axis_off()
		tree_ax.set_title('Selected sub-tree\nof taxa', pad=30, fontsize=20, wrap=True, ha='left')

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
		tree_ax.annotate(taxa_str,
			xy=(0.945, 0.009),
			xycoords='figure fraction',
			bbox=dict(facecolor='none'),
			fontsize=8,
			ha='left')

		plt.suptitle(rule_eng, wrap=True, fontsize=24, fontweight='bold')
		ax_0 = plt.axes((tree_ax.get_position().x0, 0.01, 0.05, 0.05))
		ax_0.set_box_aspect(1)
		b = Button(ax_0, 'Click')

		fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_tree(event, phylo_tree))
		gs.tight_layout(fig)
		plt.show(block=False)

	def on_click_tree(self, event, tree):
		tree.show()

		return