import os
import os.path as osp

import pandas as pd
import numpy as np
import argparse as arp

from config import *
from matplotlib.patches import Patch
from common.plot import plot_multiple_bars

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot prediction errors')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-l', '--legend', help='Feature extractors', nargs='+', default=['mlp', 'lstm', 'bilstm', 'cnn1lstm', 'cnn1'])
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
    p_mean_keys = list(p_mean.keys())
    p_max_keys = list(p_max.keys())

    tbl_dc_combs = p_mean[dc_combs_col_name].values
    assert np.any(tbl_dc_combs == p_max[dc_combs_col_name].values)

    # plot settings

    color = 'k'
    lines = ['-', '--', ':', '-.']
    markers = ['o', 's', 'X', '^', '<', '>', 'v', 'D']
    unique_hatches = ['-', '\\', '/', '.', 'o']
    models = np.sort([key for key in p_results.keys() if key not in [br_key, ts_key]])
    dcs = np.unique([int(m.split('_')[-1]) for m in models])

    # plot mean errors

    dc_combs = []
    for dc in dcs:
        dc_comb = ','.join([str(item) for item in np.arange(dc) + 1])
        dc_combs.append(dc_comb)
    if args.legend is None:
        legend_names = list(set([m.split('_')[0] for m in models]))
    else:
        legend_names = args.legend
    hatches = [unique_hatches[i] for i in range(len(legend_names))]
    legend_items = [Patch(facecolor='white', edgecolor='black', hatch=hatch) for hatch in hatches]

    # plot mean errors

    heights = []
    for name in legend_names:
        heights.append([])
        for dc_comb in dc_combs:
            idx = np.where(tbl_dc_combs == dc_comb)[0]
            assert len(idx) == 1
            idx = idx[0]
            heights[-1].append(p_mean[name].values[idx])

    fpath = osp.join(task_figures_dir, f'mean_prediction_errors.pdf')
    figh = 8
    plot_multiple_bars(dc_combs, heights, hatches, figh, 'Feature delay class combinations', 'Mean error', legend_items, legend_names, fpath, plot_png=False)

    # plot max errors

    heights = []
    for name in legend_names:
        heights.append([])
        for dc_comb in dc_combs:
            idx = np.where(tbl_dc_combs == dc_comb)[0]
            assert len(idx) == 1
            idx = idx[0]
            heights[-1].append(p_max[name].values[idx])

    fpath = osp.join(task_figures_dir, f'max_prediction_errors.pdf')
    figh = 8
    plot_multiple_bars(dc_combs, heights, hatches, figh, 'Feature delay class combinations', 'Max error', legend_items, legend_names, fpath, plot_png=False)