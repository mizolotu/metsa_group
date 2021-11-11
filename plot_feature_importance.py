import os, json
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from config import *
from bruteforce_feature_test import load_meta
from matplotlib.patches import Patch
from common.plot import plot_bars

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature importance')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-w', '--width', help='Figure width', type=float, default=21.2)
    parser.add_argument('-a', '--anonymize', help='Anonymize?', type=bool, default=False)
    parser.add_argument('-c', '--correlation', help='Correlation', default='pearson')
    args = parser.parse_args()

    # feature elimination metric

    if args.correlation is None:
        elim = 'all'
    else:
        elim = args.correlation

    # directories and meta

    task_dir = osp.join(data_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    all_features = meta['features']
    all_classes = meta['classes']

    # select features

    features, classes = all_features.copy(), all_classes.copy()
    features_idx = np.arange(len(features))

    # anonymize

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

    # file list

    fname_list_all = [xy_correlation_csv]
    for c in np.unique(classes):
        fname_list_all.append(permutation_importance_csv.format(elim, c))

    fname_list = []
    for fname in fname_list_all:
    #for fname in [xy_correlation_csv, prediction_importance_csv, permutation_importance_csv.format('all')]:
        if osp.isfile(osp.join(task_results_dir, fname)):
            fname_list.append(fname)

    print(fname_list)

    # data

    data = []
    data_to_sort = []
    reverses = []
    names = []
    fighs = []
    ylabels = []
    categories = []
    for fname in fname_list:
        fpath = osp.join(results_dir, args.task, fname)
        p = pd.read_csv(fpath)
        for col in range(len(p.keys()) - 1):
            errors = p.values[features_idx, 1 + col]
            data.append(errors)
            if fname == xy_correlation_csv:
                data_to_sort.append(np.abs(errors))
                fighs.append(12)
                ylabels.append('Correlation')
                categories.append(0)
            else:
                data_to_sort.append(errors)
                if fname.startswith(permutation_importance_csv.format('', '').split('_')[0]):
                    ylabels.append('Permutation feature importance')
                    fighs.append(6)
                elif fname == prediction_importance_csv:
                    ylabels.append('Prediction error')
                    fighs.append(6)
                categories.append(int(fname.split('_')[-1].split('.csv')[0]))
            if fname == prediction_importance_csv and col == 0:
                reverses.append(False)
            else:
                reverses.append(True)
            key = p.keys()[col + 1]
            fname_prefix = fname.split(csv)[0]
            names.append(f'{fname_prefix}_{key}')

    # plot results

    S = {c: [] for c in np.unique(classes)}
    for category, items, items_as, name, figh, ylabel, reverse in zip(categories, data, data_to_sort, names, fighs, ylabels, reverses):
        fpath = osp.join(task_figures_dir, f'{name}{postfix}{pdf}')
        plot_bars(feature_names, items.copy(), hatches, items_as.copy(), figh, xlabel, ylabel, legend_items, legend_names, fpath, sort=True, reverse=reverse, xticks_rotation='vertical', figw=args.width)
        s = items_as
        if category == 0:
            for c in np.unique(classes):
                S[c].append(s)
        else:
            S[category].append(s)
        if 0: #  np.all(pd.isna(items) == False):
            if reverse:
                s = items_as
            else:
                s = 1 / items_as
            if np.all(~pd.isna(s)):
                S[category].append(s)
    for c in np.unique(classes):
        S[c] = np.vstack(S[c])

    # rank features

    #S = S / np.sum(S, 1)[:, None]
    for c in np.unique(classes):
        S[c] = (S[c] - np.nanmin(S[c], 1)[:, None]) / (np.nanmax(S[c], 1)[:, None] - np.nanmin(S[c], 1)[:, None] + 1e-10)
        S[c] = np.nansum(S[c], 0)
        fpath = osp.join(task_figures_dir, f'features_ranked{postfix}_{c}{pdf}')
        plot_bars(feature_names, S[c], hatches, S[c], 7, xlabel, 'Feature importance score', legend_items, legend_names, fpath, sort=True, reverse=True, xticks_rotation='vertical', figw=args.width)