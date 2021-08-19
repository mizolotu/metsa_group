import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from config import *
from calculate_prediction_error import load_meta
from matplotlib import pyplot as pp
from matplotlib.patches import Patch

def plot_bars(tags, heights, hatches, items_for_argsort, fname, figh, ylabel, reverse=False):
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
    unique_hatches = np.array(['-', '\\', '/', '.', 'o'])
    legend_items = [Patch(facecolor='white', edgecolor='black', hatch=hatch) for hatch in unique_hatches]
    legend_names = [f'Delay class {dc}' for dc in np.unique(dcs)]
    _idx = dcs - 1
    colors = unique_colors[_idx]
    hatches = unique_hatches[_idx]

    # data

    data = []
    data_to_sort = []
    reverses = []
    names = []
    fighs = []
    ylabels = []
    for fname in [correlation_csv, prediction_error_csv, permutation_error_csv]:
        fpath = osp.join(results_dir, args.task, fname)
        p = pd.read_csv(fpath)
        assert np.all(tags == p['Tags'].values), 'Wrong tag order, something is worng :('
        for col in range(len(p.keys()) - 1):
            errors = p.values[:, 1 + col]
            data.append(errors)
            if fname == correlation_csv:
                data_to_sort.append(np.abs(errors))
                fighs.append(12)
                ylabels.append('Correlation')
            else:
                data_to_sort.append(errors)
                if fname == permutation_error_csv:
                    ylabels.append('Permutation feature importance')
                    fighs.append(7)
                elif fname == prediction_error_csv:
                    ylabels.append('Prediction error')
                    fighs.append(7)
            if fname == prediction_error_csv and col == 0:
                reverses.append(False)
            else:
                reverses.append(True)
            key = p.keys()[col + 1]
            prefix = fname.split(csv)[0]
            names.append(f'{prefix}_{key}')

    # plot results

    S = []
    for items, items_as, name, figh, ylabel, reverse in zip(data, data_to_sort, names, fighs, ylabels, reverses):
        plot_bars(tags, items, hatches, items_as, name, figh, ylabel, reverse)
        if np.all(pd.isna(items) == False):
            if reverse:
                s = items_as
            else:
                s = 1 / items_as
            S.append(s)
    S = np.vstack(S)

    # rank features

    S = S / np.sum(S, 1)[:, None]
    S = np.sum(S, 0)
    plot_bars(tags, S, hatches, S, 'features_ranked', 7, 'Feature importance score', True)







