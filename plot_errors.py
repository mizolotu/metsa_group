import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from train_models import load_meta
from config import *
from matplotlib import pyplot as pp
from matplotlib.lines import Line2D

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
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

    # data

    fpath = osp.join(results_dir, args.task, 'error.csv')
    p = pd.read_csv(fpath)
    dc_combs = p['Delay classes']
    model_names = p.keys()[1:]
    errors = p.values[:, 1:]
    nans = pd.isna(errors)
    non_nan_idx = []
    for i in range(len(model_names)):
        if np.all(nans[:, i] == False):
            non_nan_idx.append(i)
    errors = errors[:, np.array(non_nan_idx, dtype=int)]
    model_names = model_names[np.array(non_nan_idx, dtype=int)]

    # colors

    unique_colors = np.array(['darkviolet', 'royalblue', 'seagreen', 'firebrick'])
    legend_items = [Line2D([0], [0], color=unique_colors[i]) for i in range(len(model_names))]
    legend_names = [f"{' '.join(mn.split('_'))}" for mn in model_names]
    colors = np.vstack([unique_colors[:len(model_names)] for dc_comb in dc_combs])

    # plot by combination

    errors_mean = np.mean(errors, axis=1)
    idx = np.argsort(errors_mean)
    fpath = osp.join(task_figures_dir, 'error_per_combinarion.pdf')
    items = dc_combs[idx]
    x = np.arange(len(dc_combs))
    width = 0.2
    for i in range(len(model_names)):
        h = errors[idx, i]
        c = colors[:, i]
        pp.bar(x + i * width, height=h, width=width, color=c, label=legend_names[i])
    pp.xlabel('Delay class combination', fontdict={'size': 4})
    pp.ylabel('Prediction error', fontdict={'size': 4})
    pp.xticks(x + (len(model_names) - 1) * width / 2, dc_combs[idx])
    pp.legend(loc='best', prop={'size': 4})
    pp.xticks(fontsize=4, rotation='vertical')
    pp.yticks(fontsize=4)
    pp.savefig(fpath)
    pp.close()

    # plot by delay class

    errors_ = []
    for tag_key in tags.keys():
        idx = []
        for i, dc_comb in enumerate(dc_combs):
            if tag_key in dc_comb:
                idx.append(i)
        idx = np.array(idx)
        errors_.append(np.mean(errors[idx, :], axis=0))
    errors_ = np.vstack(errors_)
    errors_mean = np.mean(errors_, axis=1)
    idx = np.argsort(errors_mean)
    fpath = osp.join(task_figures_dir, 'error_per_delay_class.pdf')
    items = [f'Delay class {dc}' for dc in tags.keys()]
    x = np.arange(len(items))
    width = 0.2
    for i in range(len(model_names)):
        h = errors[idx, i]
        c = colors[:, i]
        pp.bar(x + i * width, height=h, width=width, color=c, label=legend_names[i])
    pp.xlabel('Delay class', fontdict={'size': 4})
    pp.ylabel('Prediction error', fontdict={'size': 4})
    pp.xticks(x + (len(model_names) - 1) * width / 2, dc_combs[idx])
    pp.legend(loc='best', prop={'size': 4})
    pp.xticks(fontsize=4)
    pp.yticks(fontsize=4)
    pp.savefig(fpath)
    pp.close()

