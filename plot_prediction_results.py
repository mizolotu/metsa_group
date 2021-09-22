import os
import os.path as osp

import pandas as pd
import numpy as np
import argparse as arp

import pylab as p

from config import *
from matplotlib import pyplot as pp
from matplotlib.patches import Patch
from common.plot import plot_bars

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot prediction errors')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-e', '--extractors', help='Feature extractors', nargs='+', default=[])
    args = parser.parse_args()

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # load data

    task_results_dir = osp.join(results_dir, args.task)
    results_mode_dir = osp.join(task_results_dir, args.mode)
    e_mean_path = osp.join(results_mode_dir, prediction_mean_errors_fname)
    e_max_path = osp.join(results_mode_dir, prediction_max_errors_fname)
    r_path = osp.join(results_mode_dir, prediction_results_fname)
    p_mean = pd.read_csv(e_mean_path)
    p_max = pd.read_csv(e_max_path)
    p_results = pd.read_csv(r_path)

    tbl_dc_combs = p_mean[dc_combs_col_name].values
    assert np.any(tbl_dc_combs == p_max[dc_combs_col_name].values)

    # plot settings

    color = 'k'
    lines = ['-', '--', ':', '-.']
    markers = ['o', 's', 'X', '^', '<', '>', 'v', 'D']
    unique_hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    models = np.sort([key for key in p_results.keys() if key not in [br_key, ts_key]])
    dcs = np.unique([int(m.split('_')[-1]) for m in models])

    for dc in dcs:
        dc_comb = ','.join([str(item) for item in np.arange(dc) + 1])
        legend_names = [m for m in models if int(m.split('_')[-1]) == dc]
        hatches = [unique_hatches[0] for _ in legend_names]
        legend_items = [Patch(facecolor='white', edgecolor='black', hatch=hatch) for hatch in hatches]

        # plot mean errors

        p_mean_keys = list(p_mean.keys())
        tags, heights = [], []
        for name in legend_names:
            name_without_dc = '_'.join(name.split('_')[:-1])
            tags.append(' '.join(name_without_dc.split('_')))
            assert name_without_dc in p_mean_keys
            idx = np.where(tbl_dc_combs == dc_comb)[0]
            assert len(idx) == 1
            idx = idx[0]
            heights.append(p_mean[name_without_dc].values[idx])

        fpath = osp.join(task_figures_dir, f'mean_prediction_errors_{dc}.pdf')
        figh = 8
        plot_bars(tags, heights, hatches, heights, figh, 'Model', 'Mean error', legend_items, legend_names, fpath, sort=True, reverse=False, plot_png=False)

        # plot max errors

        p_mean_keys = list(p_max.keys())
        tags, heights = [], []
        for name in legend_names:
            name_without_dc = '_'.join(name.split('_')[:-1])
            tags.append(' '.join(name_without_dc.split('_')))
            assert name_without_dc in p_mean_keys
            idx = np.where(tbl_dc_combs == dc_comb)[0]
            assert len(idx) == 1
            idx = idx[0]
            heights.append(p_max[name_without_dc].values[idx])

        fpath = osp.join(task_figures_dir, f'max_prediction_errors_{dc}.pdf')
        figh = 8
        plot_bars(tags, heights, hatches, heights, figh, 'Model', 'Max error', [], [], fpath, sort=True, reverse=False, plot_png=False)


