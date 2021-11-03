import json
import os
import os.path as osp
import numpy as np
import argparse as arp

from scipy.stats import spearmanr
from config import *
from common.utils import load_meta, load_data, substitute_nan_values

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Calculate feature correlations')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--max', help='Maximum correlation', default=0.5, type=float)
    args = parser.parse_args()

    # meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    # load data

    values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features)
    nfeatures = len(features)
    less_correlated_features = {}

    # preprocess data

    values = substitute_nan_values(values)

    # create directories

    task_results_dir = osp.join(results_dir, args.task)
    for dir in [results_dir, task_results_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    # eliminate features

    for corr in ['pearson', 'spearman']:

        # calculate feature-vs-target correlations

        if corr == 'pearson':
            corr_xy = np.zeros(nfeatures)
            for i in range(nfeatures):
                corr_xy[i] = np.corrcoef(values[:, i], labels)[0, 1]
        elif corr == 'spearman':
            corr_xy, _ = spearmanr(values, labels)
            corr_xy = corr_xy[:-1, -1]

        # calculate feature-vs-feature correlations

        done = False
        features_indexes = np.arange(nfeatures).tolist()
        while not done:
            n = len(features_indexes)
            corr_xx = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i > j:
                        if corr == 'pearson':
                            corr_xx[i, j] = np.abs(np.corrcoef(values[:, features_indexes[i]], values[:, features_indexes[j]])[0, 1])
                        elif corr == 'spearman':
                            corr_xx[i, j], _ = np.abs(spearmanr(values[:, features_indexes[i]], values[:, features_indexes[j]]))

            corr_xx_max = np.max(corr_xx)
            if corr_xx_max > args.max:
                i, j = np.unravel_index(np.argmax(corr_xx), corr_xx.shape)
                f_i, f_j = features_indexes[i], features_indexes[j]
                if corr_xy[f_i] >= corr_xy[f_j]:
                    features_indexes.remove(f_j)
                else:
                    features_indexes.remove(f_i)
            else:
                done = True
                print(f'Only {len(features_indexes)} features left!')

                # save the results

                fname = less_correlated_json.format(corr)
                with open(osp.join(task_results_dir, fname), 'w') as f:
                    json.dump(features_indexes, f)