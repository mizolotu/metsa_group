import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp
import seaborn as sn

from config import *
from common.utils import load_meta
from matplotlib import pyplot as pp

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature correlation')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-a', '--anonymize', help='Anonymize?', type=bool, default=False)
    args = parser.parse_args()

    # directories and meta

    task_dir = osp.join(data_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # load data

    data = {}
    for xx_name in [xx_pearson_correlation_csv, xx_spearman_correlation_csv]:
        xx_path = osp.join(task_results_dir, xx_name)
        values = pd.read_csv(xx_path, index_col=0).values
        if args.anonymize:
            postfix = '_anonymized'
            names = [str(i) for i in range(len(features))]
        else:
            postfix = ''
            names = features
        data[xx_name] = pd.DataFrame(np.abs(values), index=features, columns=features)


    # data frames and colormap

    cmap = sn.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

    # plot correlations

    for key in data:
        fpath = osp.join(task_figures_dir, f'{key.split(csv)[0]}{postfix}{pdf}')
        pp.figure(figsize=(16, 12))
        sn.heatmap(data[key], cmap=cmap, xticklabels=1, yticklabels=1)
        pp.xlabel('Features', fontdict={'size': 12})
        pp.ylabel('Features', fontdict={'size': 12})
        pp.xticks(fontsize=6)
        pp.yticks(fontsize=6)
        pp.savefig(fpath, bbox_inches='tight')
        pp.close()