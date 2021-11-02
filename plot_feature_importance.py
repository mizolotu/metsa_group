import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from config import *
from bruteforce_feature_test import load_meta
from matplotlib import pyplot as pp
from matplotlib.patches import Patch
from common.plot import plot_bars

def plot_bars1(tags, heights, hatches, items_for_argsort, fname, figh, ylabel, reverse=False, plot_png=False):
    fpath = osp.join(task_figures_dir, f'{fname}{pdf}')
    idx = np.argsort(items_for_argsort)
    if reverse:
        idx = idx[::-1]
    items = tags[idx]
    he = heights[idx]
    print(fname, items[0], he[0], items[1], he[1], items[2], he[2])
    ha = hatches[idx]
    pp.figure(figsize=(21.2, figh))
    pp.bar(items, height=he, color='white', edgecolor='black', hatch=ha)
    pp.xlabel('Features', fontdict={'size': 12})
    pp.ylabel(ylabel, fontdict={'size': 12})
    pp.xticks(fontsize=8, rotation='vertical')
    pp.yticks(fontsize=12)
    pp.legend(legend_items, legend_names, prop={'size': 12})
    pp.savefig(fpath, bbox_inches='tight')
    if plot_png:
        fpath = fpath.replace('.pdf', '.png')
        pp.savefig(fpath, bbox_inches='tight')
    pp.close()

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature importance')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-a', '--anonymize', help='Anonymize?', type=bool, default=False)
    args = parser.parse_args()

    # directories and meta

    task_dir = osp.join(data_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    if args.anonymize:
        postfix = '_anonymized'
        feature_names = [str(i + 1) for i in range(len(features))]
    else:
        postfix = ''
        feature_names = features

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # plot settings

    xlabel = 'Features'
    unique_colors = np.array(['darkviolet', 'royalblue', 'seagreen', 'gold', 'firebrick'])
    unique_hatches = np.array(['-', '\\', '/', '.', 'o'])
    legend_items = [Patch(facecolor='white', edgecolor='black', hatch=hatch) for hatch in unique_hatches]
    legend_names = [f'Delay class {dc}' for dc in np.unique(classes)]
    _idx = np.array(classes) - 1
    colors = unique_colors[_idx]
    hatches = unique_hatches[_idx]

    # data

    data = []
    data_to_sort = []
    reverses = []
    names = []
    fighs = []
    ylabels = []
    for fname in [xy_correlation_csv, prediction_importance_csv, permutation_importance_csv]:
        fpath = osp.join(results_dir, args.task, fname)
        p = pd.read_csv(fpath)
        assert np.all(features == p['Features'].values), 'Wrong tag order, something is worng :('
        for col in range(len(p.keys()) - 1):
            errors = p.values[:, 1 + col]
            data.append(errors)
            if fname == xy_correlation_csv:
                data_to_sort.append(np.abs(errors))
                fighs.append(12)
                ylabels.append('Correlation')
            else:
                data_to_sort.append(errors)
                if fname == permutation_importance_csv:
                    ylabels.append('Permutation feature importance')
                    fighs.append(7)
                elif fname == prediction_importance_csv:
                    ylabels.append('Prediction error')
                    fighs.append(7)
            if fname == prediction_importance_csv and col == 0:
                reverses.append(False)
            else:
                reverses.append(True)
            key = p.keys()[col + 1]
            prefix = fname.split(csv)[0]
            names.append(f'{prefix}_{key}')

    # plot results

    S = []
    for items, items_as, name, figh, ylabel, reverse in zip(data, data_to_sort, names, fighs, ylabels, reverses):
        fpath = osp.join(task_figures_dir, f'{name}{postfix}{pdf}')
        plot_bars(feature_names, items, hatches, items_as, figh, xlabel, ylabel, legend_items, legend_names, fpath, sort=True, reverse=reverse, xticks_rotation='vertical')
        if np.all(pd.isna(items) == False):
            if reverse:
                s = items_as
            else:
                s = 1 / items_as
            S.append(s)
    S = np.vstack(S)

    # rank features

    #S = S / np.sum(S, 1)[:, None]
    S = (S - np.min(S, 1)[:, None]) / (np.max(S, 1)[:, None] - np.min(S, 1)[:, None] + 1e-10)
    S = np.sum(S, 0)
    fpath = osp.join(task_figures_dir, f'features_ranked{postfix}{pdf}')
    plot_bars(feature_names, S, hatches, S, 7, xlabel, 'Feature importance score', legend_items, legend_names, fpath, sort=True, reverse=True, xticks_rotation='vertical')







