import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from config import *
from calculate_prediction_error import load_meta
from matplotlib import pyplot as pp

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature importance')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-f', '--feature', help='Feature to plot', default='126A0519-QI') # '126A0118-QI' '126A0333-QI' '126A0519-QI' '126A0535-QIC'
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

    tags_ = []
    for key in sorted(tags.keys()):
        tags_.extend(tags[key])

    # load data

    vals = []
    for stage in stages:
        fpath = osp.join(processed_data_dir, f'{args.task}_{stage}{csv}')
        p = pd.read_csv(fpath)
        vals.append(p.values)
    vals = np.vstack(vals)
    X = vals[:, :-1]
    y = vals[:, -1]
    nfeatures = X.shape[1]

    # plot feature

    assert args.feature in tags_
    idx = tags_.index(args.feature)
    x = X[:, idx]
    idx_ = np.where((X[:, idx] != meta['xmax'][idx]) & (X[:, idx] != meta['xmin'][idx]))
    fpath = osp.join(task_figures_dir, f'{args.feature}_vs_{br_key}.pdf')
    pp.figure(figsize=(12, 12))
    pp.plot(x[idx_], y[idx_], 'k.')
    pp.xlabel(args.feature, fontdict={'size': 12})
    pp.ylabel(br_key, fontdict={'size': 12})
    pp.legend(loc='best', prop={'size': 12})
    pp.xticks(fontsize=12)
    pp.yticks(fontsize=12)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()