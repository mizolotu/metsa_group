import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from scipy.stats import spearmanr
from config import *
from calculate_prediction_error import set_seeds, load_meta
from matplotlib import pyplot as pp
from matplotlib.lines import Line2D

def plot_bars(tags, heights, colors, items_for_argsort, fname, reverse=False):
    fpath = osp.join(task_figures_dir, f'{fname}{pdf}')
    idx = np.argsort(items_for_argsort)
    if reverse:
        idx = idx[::-1]
    items = tags[idx]
    h = heights[idx]
    c = colors[idx]
    pp.bar(items, height=h, color=c)
    pp.xlabel('Tag name', fontdict={'size': 4})
    pp.ylabel('Feature importance', fontdict={'size': 4})
    pp.xticks(fontsize=2, rotation='vertical')
    pp.yticks(fontsize=4)
    pp.legend(legend_items, legend_names, prop={'size': 4})
    pp.savefig(fpath)
    pp.close()

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature importance')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    args = parser.parse_args()

    # meta

    meta = load_meta(processed_data_dir, args.task)
    tags = meta['tags']

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # delay classes and tags

    dc_list, tag_list = [], []
    for key in sorted(tags.keys()):
        dc_list.extend([key for _ in tags[key]])
        tag_list.extend(tags[key])
    tags = np.array(tag_list)
    dcs = np.array(dc_list, dtype=int)

    # plot settings

    unique_colors = np.array(['darkviolet', 'royalblue', 'seagreen', 'gold', 'firebrick'])
    legend_items = [Line2D([0], [0], color=color) for color in unique_colors]
    legend_names = [f'Delay class {dc}' for dc in np.unique(dcs)]
    color_idx = dcs - 1
    colors = unique_colors[color_idx]

    # pearson correlation

    pearson_corr = np.zeros(nfeatures)
    for i in range(nfeatures):
        pearson_corr[i] = np.corrcoef(X[:, i], y)[0, 1]
    pearson_corr_abs = np.abs(pearson_corr)

    # spearman correlation

    s_ranks, _ = spearmanr(X, y)
    spearman_corr = s_ranks[:-1, -1]
    spearman_corr_abs = np.abs(spearman_corr)

    # error by tag

    fpath = osp.join(results_dir, args.task, error_by_tag_csv)
    p = pd.read_csv(fpath)
    assert np.all(tags == p['Tags'].values), 'Wrong tag order, something is worng :('
    errors = p.values[:, 1:]
    non_nan_idx = np.all(pd.isna(errors)==False, axis=0)
    idx = np.where(non_nan_idx==True)[0][-1]
    errors_by_tag = errors[:, idx]

    # plot results

    items_list = [pearson_corr, spearman_corr, errors_by_tag]
    items_as_list = [pearson_corr_abs, spearman_corr_abs, errors_by_tag]
    fnames = ['pearson_by_tag', 'spearman_by_tag', 'error_by_tag']
    reverses = [True, True, False]
    for items, items_as, fname, reverse in zip(items_list, items_as_list, fnames, reverses):
        print(fname)
        plot_bars(tags, items, colors, items_as, fname, reverse)






