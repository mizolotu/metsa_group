import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from scipy.stats import spearmanr
from config import *
from train_model import set_seeds, load_meta
from matplotlib import pyplot as pp
from matplotlib.lines import Line2D

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    args = parser.parse_args()

    # set seed for results reproduction

    set_seeds(seed)

    # meta

    meta = load_meta(processed_data_dir, args.task)
    tags = meta['tags']

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    pc_list, sr_list, dc_list, tag_list = [], [], [], []
    for tag_key in tags.keys():

        # delay classes and tags

        dc_list.extend([tag_key for _ in tags[tag_key]])
        tag_list.extend(tags[tag_key])

        # collect data

        vals = []
        for stage in stages:
            fpath = osp.join(processed_data_dir, f'{args.task}_{tag_key}_{stage}{csv}')
            p = pd.read_csv(fpath, header=None)
            vals.append(p.values)
        vals = np.vstack(vals)

        # calculate correlation coefficients

        X = vals[:, :-1]
        y = vals[:, -1]

        # pearson correlation

        nfeatures = X.shape[1]
        pearson_corr = np.zeros(nfeatures)
        for i in range(nfeatures):
            pearson_corr[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        pc_list.append(pearson_corr)

        # spearman correlation

        s_ranks, _ = spearmanr(X, y)
        spearman_corr = np.abs(s_ranks[:-1, -1])
        sr_list.append(spearman_corr)

    # arrays

    tags = np.array(tag_list)
    dcs = np.array(dc_list, dtype=int)
    pcs = np.hstack(pc_list)
    srs = np.hstack(sr_list)

    # plot stats

    unique_colors = np.array(['darkviolet', 'royalblue', 'seagreen', 'gold', 'firebrick'])
    legend_items = [Line2D([0], [0], color=color) for color in unique_colors]
    legend_names = [f'Delay class {dc}' for dc in np.unique(dcs)]
    color_idx = dcs - 1
    colors = unique_colors[color_idx]

    # pc plot

    fpath = osp.join(task_figures_dir, 'pearson_by_tag.pdf')
    idx = np.argsort(pcs)[::-1]
    items = tags[idx]
    h = pcs[idx]
    c = colors[idx]
    pp.bar(items, height=h, color=c)
    pp.xlabel('Tag name', fontdict={'size': 4})
    pp.ylabel('Correlation with bleach ratio', fontdict={'size': 4})
    pp.xticks(fontsize=2, rotation='vertical')
    pp.yticks(fontsize=4)
    pp.legend(legend_items, legend_names, prop={'size': 4})
    pp.savefig(fpath)
    pp.close()

    # sr plot

    fpath = osp.join(task_figures_dir, 'spearman_by_tag.pdf')
    idx = np.argsort(srs)[::-1]
    items = tags[idx]
    h = srs[idx]
    c = colors[idx]
    pp.bar(items, height=h, color=c)
    pp.xlabel('Tag name', fontdict={'size': 4})
    pp.ylabel('Correlation with bleach ratio', fontdict={'size': 4})
    pp.xticks(fontsize=2, rotation='vertical')
    pp.yticks(fontsize=4)
    pp.legend(legend_items, legend_names, prop={'size': 4})
    pp.savefig(fpath)
    pp.close()