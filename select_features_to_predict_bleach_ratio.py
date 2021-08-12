import json, os
import os.path as osp
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from sklearn.feature_selection import f_regression
from scipy.stats import spearmanr
from config import *
from train_model_to_predict_bleach_ratio import set_seeds, load_meta

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    args = parser.parse_args()

    # set seed for results reproduction

    set_seeds(seed)

    # meta and standardization values

    meta = load_meta(processed_data_dir, args.task)
    tags = meta['tags']

    # create output directory

    task_figures_dir = osp.join(figures_dir, args.task)
    for dir in [figures_dir, task_figures_dir]:
        if not osp.isdir(dir):
            os.mkdir(dir)

    fv_list, dc_list, tag_list = [], [], []
    for tag_key in tags.keys():

        # collect data

        vals = []
        for stage in stages:
            fpath = osp.join(processed_data_dir, f'{args.task}_{tag_key}_{stage}{csv}')
            p = pd.read_csv(fpath, header=None)
            vals.append(p.values)
        vals = np.vstack(vals)

        # calculate correlation coefficients

        X = vals[:, :-1]
        y = vals[:, -1]
        f_values, p_values = f_regression(X, y)
        s_ranks = spearmanr(X, y)
        print(tag_key,X.shape,s_ranks)
        for tag, f_value in zip(tags[tag_key], f_values):
            dc_list.append(tag_key)
            tag_list.append(tag)
            fv_list.append(f_value)






