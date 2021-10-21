import os
import os.path as osp

import pandas as pd
import numpy as np
import argparse as arp

from config import *
from common.plot import plot_multiple_lines
from sklearn.metrics import roc_curve

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot ROC')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    args = parser.parse_args()

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # load data

    task_results_dir = osp.join(results_dir, args.task)
    results_mode_dir = osp.join(task_results_dir, args.mode)
    r_path = osp.join(results_mode_dir, anomaly_detection_results_fname)
    p_results = pd.read_csv(r_path)
    p_keys = list(p_results.keys())
    y_true = p_results[br_key]

    # plot settings

    color = 'k'
    lines = ['-', '--', ':', '-.']
    models = np.sort([key for key in p_results.keys() if key not in [br_key, ts_key]])
    dcs = np.unique([int(m.split('_')[-1]) for m in models])[1:]
    model_types = list(set([m.split('_')[0] for m in models]))

    # plot mean errors

    dc_combs = []
    for dc in dcs:
        dc_comb = ','.join([str(item) for item in np.arange(dc) + 1])
        dc_combs.append(dc_comb)
    legend_names = dc_combs

    # plot mean errors

    for model_type in model_types:
        x, y = [], []
        for dc in dcs:
            key = f'{model_type}_{dc}'
            fpr, tpr, thr = roc_curve(y_true, p_results[key])
            x.append(fpr)
            y.append(tpr)

        fpath = osp.join(task_figures_dir, f'anomaly_detection_roc.pdf')
        figh = 8
        plot_multiple_lines(x, y, lines, 'False positive rate', 'True positive rate', fpath)