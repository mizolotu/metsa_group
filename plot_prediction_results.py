import os
import os.path as osp
import sys

import pandas as pd
import numpy as np
import argparse as arp
import seaborn as sn

from config import *
from matplotlib import pyplot as pp

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot prediction errors')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-e', '--extractors', help='Feature extractors', nargs='+', default=['split_mlp', 'split_cnn1'])
    args = parser.parse_args()

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # load data

    task_results_dir = osp.join(results_dir, args.task)
    results_mode_dir = osp.join(task_results_dir, args.mode)
    e_path = osp.join(results_mode_dir, prediction_errors_fname)
    p = pd.read_csv(e_path)

    # extract data to plot

    pkeys = list(p.keys())
    assert dc_combs_col_name in pkeys
    x = p[dc_combs_col_name].values
    y = np.zeros((len(x), len(args.extractors)))
    for j, extractor in enumerate(args.extractors):
        i = 0
        for pkey in pkeys:
            spl = pkey.split('_')
            mname = '_'.join(spl[:2])
            if mname == extractor:
                if '-' in spl[2]:
                    raise NotImplemented
                else:
                    fc = int(spl[2])
                    er = p[pkey].values[fc - 1]
                y[i, j] = er
                i += 1

    # plot error

    fname = '_vs_'.join(args.extractors)
    fpath = osp.join(task_figures_dir, f'prediction_errors_{fname}{pdf}')
    pp.figure(figsize=(16, 12))
    pp.plot(x, y, 'ko-')
    pp.xlabel('Feature class combinations', fontdict={'size': 12})
    pp.ylabel('Mean absolute error', fontdict={'size': 12})
    pp.xticks(x, fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()