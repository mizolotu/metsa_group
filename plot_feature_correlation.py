import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp
import seaborn as sn

from config import *
from calculate_prediction_error import load_meta
from matplotlib import pyplot as pp
from scipy.stats import spearmanr

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature correlation')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
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

    # calculate correlations

    r = np.zeros((nfeatures, nfeatures))
    s = np.zeros((nfeatures, nfeatures))
    for i in range(nfeatures):
        for j in range(nfeatures):
            r[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
            s[i, j], _ = spearmanr(X[:, i], X[:, j])

    # max correlations

    for i in range(nfeatures):
        a = np.abs(r[i, :])
        b = np.abs(s[i, :])
        idxa = np.argsort(a)
        idxb = np.argsort(b)
        print(tags_[i], tags_[idxa[-2]], r[i, idxa[-2]], tags_[idxa[-3]], r[i, idxa[-3]], tags_[idxb[-2]], s[i, idxb[-2]], tags_[idxb[-3]], s[i, idxb[-3]])

    # data frames and colormap

    df_r = pd.DataFrame(np.abs(r), index=tags_, columns=tags_)
    df_s = pd.DataFrame(np.abs(s), index=tags_, columns=tags_)
    cmap = sn.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

    # plot pearson correlation

    fpath = osp.join(task_figures_dir, 'feature_pearson_corrleation.pdf')
    pp.figure(figsize=(16, 12))
    sn.heatmap(df_r, cmap=cmap, xticklabels=1, yticklabels=1)
    pp.xlabel('Features', fontdict={'size': 12})
    pp.ylabel('Features', fontdict={'size': 12})
    pp.xticks(fontsize=6)
    pp.yticks(fontsize=6)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()

    # plot spearman correlation

    fpath = osp.join(task_figures_dir, 'feature_spearman_corrleation.pdf')
    pp.figure(figsize=(16, 12))
    sn.heatmap(df_r, cmap=cmap, xticklabels=1, yticklabels=1)
    pp.xlabel('Features', fontdict={'size': 12})
    pp.ylabel('Features', fontdict={'size': 12})
    pp.xticks(fontsize=8)
    pp.yticks(fontsize=8)
    pp.savefig(fpath, bbox_inches='tight')
    pp.close()