import os
import os.path as osp

import pandas as pd
import numpy as np
import argparse as arp

from config import *
from matplotlib import pyplot as pp

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot prediction errors')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-e', '--extractors', help='Feature extractors', nargs='+', default=['split_mlp', 'split_lstm', 'split_bilstm'])
    parser.add_argument('-l', '--lines', help='Number of lines for each dash: solid, dashed, dotted, dash-dotted', type=int, nargs='+', default=[1, 2])
    args = parser.parse_args()

    # sanitize input

    if args.lines is None:
        nlines = [1 for _ in args.extractors]
    else:
        nlines = args.lines
    assert len(nlines) <= 4
    assert len(args.extractors) <= np.sum(nlines)

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

    # plot markers

    color = 'k'
    lines = ['-', '--', ':', '-.']
    markers = ['o', 's', 'X', '^', '<', '>', 'v', 'D']

    # extract data to plot

    csum = np.cumsum(nlines)
    pkeys = list(p.keys())
    assert dc_combs_col_name in pkeys
    x = p[dc_combs_col_name].values
    ys = []
    ls = []
    ns = []
    for j, extractor in enumerate(args.extractors):
        ns.append(extractor.split('_')[1].upper())
        ys.append([])
        line_i = np.where(len(ys) <= csum)[0][0]
        ls.append(f'{color}{markers[len(ys) - 1]}{lines[line_i]}')
        for pkey in pkeys:
            spl = pkey.split('_')
            mname = '_'.join(spl[:2])
            if mname == extractor:
                if '-' in spl[2]:
                    raise NotImplemented
                else:
                    fc = int(spl[2])
                    er = p[pkey].values[fc - 1]
                ys[-1].append(er)
                print(ns[-1], fc, er)
        assert len(ys[-1]) == len(x), print('Length mismatch for x and y')

    # plot error

    fname = '_vs_'.join(args.extractors)
    fpath = osp.join(task_figures_dir, f'prediction_errors_{fname}{pdf}')
    pp.figure(figsize=(16, 12))
    for y, l, n in zip(ys, ls, ns):
        pp.plot(x, y, l, linewidth=2, markersize=12, label=n)
    pp.xlabel('Feature class combinations', fontdict={'size': 12})
    pp.ylabel('Mean absolute error', fontdict={'size': 12})
    pp.xticks(x, fontsize=12)
    pp.yticks(fontsize=12)
    pp.legend()
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()