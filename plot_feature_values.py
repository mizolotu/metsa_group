import os
import os.path as osp
import numpy as np
import argparse as arp

from config import *
from common.utils import load_meta, load_data
from matplotlib import pyplot as pp

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature values')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-f', '--feature', help='Feature to plot', default='126A0546-QI1')  # 126A0546-QI1 126A0103-QI_A1 126A0228-FIC_A3 126A0503-QI_A2
    parser.add_argument('-s', '--size', help='Fgiure size', default=12, type=int)
    parser.add_argument('-y', '--target', help='Target to plot feature against', default=br_key)
    parser.add_argument('-p', '--permute', help='Plot feature permuted?', type=bool, default=False)
    parser.add_argument('-n', '--npoints', help='Number of points to plot', type=int, default=5000)
    parser.add_argument('-a', '--anonymize', help='Anonymize?', type=bool, default=False)
    args = parser.parse_args()

    # meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    # load data

    values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features)
    nfeatures = len(features)

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # x feature

    idx_x = features.index(args.feature)
    x = values[:, idx_x]
    idx_ = np.where(np.isnan(x) == False)[0]
    np.random.shuffle(idx_)
    idx_ = idx_[:args.npoints]
    x = x[idx_]

    # y feature

    if args.target == br_key:
        y = labels[idx_]
    else:
        idx_y = features.index(args.target)
        y = values[idx_, idx_y]

    # anonymize

    if args.anonymize:
        postfix = '_anonymized'
        xlabel = f'Feature {str(idx_x + 1)}'
        if args.target == br_key:
            ylabel = 'Target'
        else:
            ylabel = f'Feature {idx_y}'
    else:
        postfix = ''
        xlabel = features[idx_x]
        ylabel = br_key

    # file path

    fname = f'{args.feature}_vs_{args.target}'.replace('.', '_')
    fpath = osp.join(task_figures_dir, f'{fname}{postfix}{pdf}')

    # figure

    pp.figure(figsize=(args.size, args.size))
    pp.plot(x, y, 'ko')
    if args.permute:
        idx_perm = np.arange(len(idx_))
        np.random.shuffle(idx_perm)
        pp.plot(x[idx_perm], y, 'kx')
    pp.xlabel(xlabel, fontdict={'size': args.size})
    pp.ylabel(ylabel, fontdict={'size': args.size})
    pp.xticks(fontsize=args.size)
    pp.yticks(fontsize=args.size)
    pp.savefig(fpath, bbox_inches='tight')
    fpath = fpath.replace('.pdf', '.png')
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()