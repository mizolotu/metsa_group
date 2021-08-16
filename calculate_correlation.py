import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse as arp

from scipy.stats import spearmanr
from config import *
from calculate_prediction_error import load_meta

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Plot feature importance')
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
    y = vals[:, -1]
    nfeatures = X.shape[1]

    # pearson correlation

    pearson_corr = np.zeros(nfeatures)
    for i in range(nfeatures):
        pearson_corr[i] = np.corrcoef(X[:, i], y)[0, 1]

    # spearman correlation

    s_ranks, _ = spearmanr(X, y)
    spearman_corr = s_ranks[:-1, -1]

    # save the results

    task_results_dir = osp.join(results_dir, args.task)
    r_name = correlation_csv
    r_path = osp.join(task_results_dir, r_name)
    p = pd.DataFrame({
        'Tags': [tag for tag in tags_],
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
    })
    p.to_csv(r_path, index=None)