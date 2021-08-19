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
    parser.add_argument('-f', '--feature', help='Feature to plot', default='126A0333-QI') # '126A0118-QI' '126A0333-QI' '126A0519-QI' '126A0535-QIC'
    parser.add_argument('-y', '--target', help='Target to plot feature against', default=br_key) # 126A0079-QT 126A0318-QI
    parser.add_argument('-p', '--permute', help='Plot feature permuted?', type=bool, default=False)
    parser.add_argument('-n', '--nfeatures', help='Number of features to plot', type=int, default=50000)
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
    nfeatures = X.shape[1]

    # plot feature

    if args.target == br_key:
        y = vals[:, -1]
    else:
        assert args.target in tags_
        idxy = tags_.index(args.target)
        y = X[:, idxy]
    ymin = np.min(y)
    assert args.feature in tags_
    idx = tags_.index(args.feature)
    x = X[:, idx]
    idx_ = np.where((x != meta['xmin'][idx]) & (y[idx] != ymin))[0]
    fpath = osp.join(task_figures_dir, f'{args.feature}_vs_{args.target}.pdf')
    pp.figure(figsize=(6, 6))
    pp.plot(x[idx_[:args.nfeatures]], y[idx_[:args.nfeatures]], 'ko')
    if args.permute:
        idx_perm = np.arange(len(idx_))
        np.random.shuffle(idx_perm)
        pp.plot(x[idx_[idx_perm[:args.nfeatures]]], y[idx_[:args.nfeatures]], 'kx')
    pp.xlabel(args.feature, fontdict={'size': 6})
    pp.ylabel(br_key, fontdict={'size': 6})
    pp.xticks(fontsize=6)
    pp.yticks(fontsize=6)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()