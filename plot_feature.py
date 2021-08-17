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
    parser.add_argument('-f', '--feature', help='Feature to plot', default='125A0321-WI')
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
    print(np.min(X[:, idx]), np.max(X[:, idx]))
    fpath = osp.join(task_figures_dir, f'{args.feature}_vs_{br_key}.pdf')
    pp.figure(figsize=(6, 6))
    pp.plot(X[:, idx], y, '.')
    pp.xlabel(args.feature, fontdict={'size': 4})
    pp.ylabel(br_key, fontdict={'size': 4})
    pp.legend(loc='best', prop={'size': 4})
    pp.xticks(fontsize=4)
    pp.yticks(fontsize=4)
    pp.savefig(fpath)
    pp.close()