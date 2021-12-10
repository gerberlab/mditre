from math import floor
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns


def plot_hist(x, title, filename, show_stats=True, pdf=None):
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


def plot_bar(x, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(np.arange(len(x)), x)
    ax.set_title(title, loc='center', wrap=True)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    return


def simple_plot(x, xlabel, ylabel, filepath, pdf=None, ax=None):
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


def round_sig(x, sig=2):
    return round(x, sig - int(floor(np.log10(abs(x)))) - 1)


def to_coord(x, y, xmin, xmax, ymin, ymax, plt_xmin, plt_ymin, plt_width, plt_height):
    x = (x - xmin) / (xmax - xmin) * plt_width  + plt_xmin
    y = (y - ymin) / (ymax - ymin) * plt_height + plt_ymin
    return x, y


def plot_tree(tree, align_names=False, name_offset=None, max_dist=None,
    font_size=9, axe=None, show_names=True, **kwargs):
    """
    Plots a ete3.Tree object using matploltib.
    
    :param tree: ete Tree object
    :param False align_names: if True names will be aligned vertically
    :param None max_dist: if defined any branch longer than the given value will be 
       reduced by this same value.
    :param None name_offset: offset relative to tips to write leaf_names. In bL scale
    :param 12 font_size: to write text
    :param None axe: a matploltib.Axe object on which the tree will be painted.
    :param kwargs: for tree edge drawing (matplotlib LineCollection) 
    :param 1 ms: marker size for tree nodes (relative to number of nodes)
    
    :returns: a dictionary of node objects with their coordinates
    """
    
    if axe is None:
        axe = plt.subplot(111)

    
    def __draw_edge_nm(c, x):
        h = node_pos[c]
        hlinec.append(((x, h), (x + c.dist, h)))
        hlines.append(cstyle)
        return (x + c.dist, h)

    def __draw_edge_md(c, x):
        h = node_pos[c]
        if c in cut_edge:
            offset = max_x / 600.
            hlinec.append(((x, h), (x + c.dist / 2 - offset, h)))
            hlines.append(cstyle)
            hlinec.append(((x + c.dist / 2 + offset, h), (x + c.dist, h)))
            hlines.append(cstyle)
            hlinec.append(((x + c.dist / 2, h - 0.05), (x + c.dist / 2 - 2 * offset, h + 0.05)))
            hlines.append(cstyle)
            hlinec.append(((x + c.dist / 2 + 2 * offset, h - 0.05), (x + c.dist / 2, h + 0.05)))
            hlines.append(cstyle)
            axe.text(x + c.dist / 2, h - 0.07, '+%g' % max_dist, va='top', 
                     ha='center', size=2. * font_size / 3, fontname='Arial')
        else:
            hlinec.append(((x, h), (x + c.dist, h)))
            hlines.append(cstyle)
        return (x + c.dist, h)

    __draw_edge = __draw_edge_nm if max_dist is None else __draw_edge_md
    
    vlinec = []
    vlines = []
    hlinec = []
    hlines = []
    nodes = []
    nodex = []
    nodey = []
    ali_lines = []
    
    # to align leaf names
    tree = tree.copy()
    max_x = max(n.get_distance(tree) for n in tree.iter_leaves())

    coords = {}
    node_pos = dict((n2, i) for i, n2 in enumerate(tree.get_leaves()[::-1]))
    node_list = tree.iter_descendants(strategy='postorder')
    node_list = chain(node_list, [tree])

    # reduce branch length
    cut_edge = set()
    if max_dist is not None:
        for n in tree.iter_descendants():
            if n.dist > max_dist:
                n.dist -= max_dist
                cut_edge.add(n)

    if name_offset is None:
        name_offset = max_x / 100.
    # draw tree
    for n in node_list:
        style = n._get_style()
        x = sum(n2.dist for n2 in n.iter_ancestors()) + n.dist
        if n.is_leaf():
            y = node_pos[n]
            if align_names:
                if show_names:
                    axe.text(max_x + name_offset, y, n.name, 
                             va='center', size=font_size, fontname='Arial')
                ali_lines.append(((x, y), (max_x + name_offset, y)))
            else:
                if show_names:
                    axe.text(x + name_offset, y, n.name,
                             va='center', size=font_size, fontname='Arial')
        else:
            y = np.mean([node_pos[n2] for n2 in n.children])
            node_pos[n] = y

            # draw vertical line
            vlinec.append(((x, node_pos[n.children[0]]), (x, node_pos[n.children[-1]])))
            vlines.append(style)

            # draw horizontal lines
            for child in n.children:
                cstyle = child._get_style()
                coords[child] = __draw_edge(child, x)
        nodes.append(style)
        nodex.append(x)
        nodey.append(y)

    # draw root
    cstyle = tree._get_style()
    __draw_edge(tree, 0)

    lstyles = ['-', '--', ':']
    hline_col = LineCollection(hlinec, colors=[l['hz_line_color'] for l in hlines], 
                              linestyle=[lstyles[l['hz_line_type']] for l in hlines],
                              linewidth=[(l['hz_line_width'] + 1.) / 2 for l in hlines])
    vline_col = LineCollection(vlinec, colors=[l['vt_line_color'] for l in vlines], 
                              linestyle=[lstyles[l['vt_line_type']] for l in vlines],
                              linewidth=[(l['vt_line_width'] + 1.) / 2 for l in vlines])
    ali_line_col = LineCollection(ali_lines, colors='k')

    axe.add_collection(hline_col)
    axe.add_collection(vline_col)
    axe.add_collection(ali_line_col)

    nshapes = dict((('circle', 'o'), ('square', 's'), ('sphere', 'o')))
    shapes = set(n['shape'] for n in nodes)
    for shape in shapes:
        indexes = [i for i, n in enumerate(nodes) if n['shape'] == shape]
        scat = axe.scatter([nodex[i] for i in indexes], 
                           [nodey[i] for i in indexes], 
                           s=0, marker=nshapes.get(shape, shape))
        scat.set_sizes([(nodes[i]['size'])**2 / 2 for i in indexes])
        scat.set_color([nodes[i]['fgcolor'] for i in indexes])
        scat.set_zorder(10)

    axe.set_axis_off()
    return coords