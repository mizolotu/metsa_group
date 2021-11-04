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
    parser.add_argument('-f', '--features', help='Feature indexes list in json format', default='less_correlated_spearman.json')
    parser.add_argument('-w', '--width', help='Figure width', type=float, default=10)
    parser.add_argument('-a', '--anonymize', help='Anonymize?', type=bool, default=False)
    args = parser.parse_args()

    # directories and meta

    task_dir = osp.join(data_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    all_features = meta['features']
    all_classes = meta['classes']

    # feature indexes

    try:
        with open(osp.join(task_results_dir, args.features)) as f:
            feature_indexes = json.load(f)
    except:
        feature_indexes = None

    # select features

    if feature_indexes is not None:
        features, classes = [all_features[i] for i in feature_indexes], [all_classes[i] for i in feature_indexes]
        features_idx = np.array(feature_indexes)
    else:
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

    # files

    if args.features is not None:
        prefix = f"{args.features.split('.json')[0]}_"
    else:
        prefix = ''

    permutation_importance_csv_with_prefix = f'{prefix}{permutation_importance_csv}'

    fname_list = []
    for fname in [xy_correlation_csv, prediction_importance_csv, permutation_importance_csv_with_prefix]:
        if osp.isfile(osp.join(task_results_dir, fname)):
            fname_list.append(fname)

    # data

    data = []
    data_to_sort = []
    reverses = []
    names = []
    fighs = []
    ylabels = []
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
            else:
                data_to_sort.append(errors)
                if fname == permutation_importance_csv_with_prefix:
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
            fname_prefix = fname.split(csv)[0]
            names.append(f'{fname_prefix}_{key}')

    # plot results

    S = []
    for items, items_as, name, figh, ylabel, reverse in zip(data, data_to_sort, names, fighs, ylabels, reverses):
        if not name.startswith(prefix):
            name = f'{prefix}{name}'
        fpath = osp.join(task_figures_dir, f'{name}{postfix}{pdf}')
        plot_bars(feature_names, items, hatches, items_as, figh, xlabel, ylabel, legend_items, legend_names, fpath, sort=True, reverse=reverse, xticks_rotation='vertical', figw=args.width)
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
    fpath = osp.join(task_figures_dir, f'{prefix}features_ranked{postfix}{pdf}')
    plot_bars(feature_names, S, hatches, S, 7, xlabel, 'Feature importance score', legend_items, legend_names, fpath, sort=True, reverse=True, xticks_rotation='vertical', figw=args.width)