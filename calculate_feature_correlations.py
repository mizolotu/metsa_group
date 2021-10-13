import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from scipy.stats import spearmanr
from config import *
from common.utils import load_meta, load_data, substitute_nan_values

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Calculate feature correlations')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    args = parser.parse_args()

    # meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    # load data

    values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features)
    nfeatures = len(features)

    # preprocess data

    values = substitute_nan_values(values)

    # calculate feature-vs-feature correlations

    pearson_corr_xx = np.zeros((nfeatures, nfeatures))
    s_ranks_xx = np.zeros((nfeatures, nfeatures))
    for i in range(nfeatures):
        for j in range(nfeatures):
            pearson_corr_xx[i, j] = np.corrcoef(values[:, i], values[:, j])[0, 1]
            s_ranks_xx[i, j], _ = spearmanr(values[:, i], values[:, j])

    # calculate feature-vs-target correlations

    pearson_corr_xy = np.zeros(nfeatures)
    for i in range(nfeatures):
        pearson_corr_xy[i] = np.corrcoef(values[:, i], labels)[0, 1]
    s_ranks_xy, _ = spearmanr(values, labels)
    s_ranks_xy = s_ranks_xy[:-1, -1]

    # create directories

    task_results_dir = osp.join(results_dir, args.task)
    for dir in [results_dir, task_results_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # save the results

    for xx_name, data in zip([xx_pearson_correlation_csv, xx_spearman_correlation_csv], [pearson_corr_xx, s_ranks_xx]):
        xx_path = osp.join(task_results_dir, xx_name)
        p = pd.DataFrame({
            'Features': [tag for tag in features]
        })
        for i, feature in enumerate(features):
            p[feature] = data[:, i]
        p.to_csv(xx_path, index=None)

    xy_name = xy_correlation_csv
    xy_path = osp.join(task_results_dir, xy_name)
    p = pd.DataFrame({
        'Features': [tag for tag in features],
        'Pearson': pearson_corr_xy,
        'Spearman': s_ranks_xy,
    })
    p.to_csv(xy_path, index=None)