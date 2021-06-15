import os, time
import argparse as arp
import os.path as osp
import dateutil.parser as dparser
import numpy as np
import matplotlib.pyplot as pp
import pandas as pd

from sklearn.preprocessing import MinMaxScaler as mmscaler
from sklearn.manifold import TSNE as tsne
from config import *

def load_tags(fname):
    if fname in os.listdir(data_dir):
        p = pd.read_excel(osp.join(data_dir, fname))
        positions = p[position_column].values
        non_yellow_indexes = np.array([i for i, p in enumerate(positions) if p not in yellow_tags])
        positions = positions[non_yellow_indexes]
        delay_classes = p[delay_class_column].values[non_yellow_indexes]
        unique_delay_classes = np.unique(delay_classes)
        tags = {}
        for i, unique_delay_class in enumerate(unique_delay_classes):
            idx = np.where(delay_classes == unique_delay_class)[0]
            tags[unique_delay_class] = positions[idx]
    else:
        tags = None
    return tags

def load_samples(fname):
    if fname in os.listdir(data_dir):
        p = pd.read_csv(osp.join(data_dir, fname))
        keys = p.keys().tolist()[1:]
        keys = [key.replace('_', '.') for key in keys]
        values = p.values[:, 1:]
        timestamps = np.array([time.mktime(dparser.parse(item).timetuple()) for item in p.values[:, 0]])
    else:
        keys, values, timestamps = None, None, None
    return keys, values, timestamps

def check_data(keys, tags):
    tags = np.hstack(tags.values()).tolist()
    for key in keys:
        if key not in tags:
            print(f'Tag {key} not found in tags!')
    for tag in tags:
        if tag not in keys:
            print(f'Tag {tag} not found in keys!')

def select_data(keys, values, timestamps):
    return keys, values, timestamps

if __name__ == '__main__':

    # parse args

    parser = arp.ArgumentParser(description='')
    parser.add_argument('-s', '--samples', help='File with samples', default='samples_15062021.csv')
    args = parser.parse_args()

    # laod data

    tags = load_tags(tags_fname)
    keys, values, timestamps = load_samples(args.samples)
    print(len(np.hstack(tags.values())), len(keys))
    check_data(keys, tags)
    keys, values, timestamps = select_samples(keys, values, timestamps)

    # filter data



    if keys is not None:

        # standardize the data

        scaler = mmscaler()
        values_std = scaler.fit_transform(values)

        # plot tsne

        non_nan_br_idx = np.where(pd.isna(values_std[:, br_index]) == False)[0]
        data_subset = values_std[non_nan_br_idx, :]
        non_nan_col_ids = np.where(np.sum(pd.isna(data_subset), axis=0) == False)[0]
        data_subset = data_subset[:, non_nan_col_ids]
        print(f'Subset shape: {data_subset.shape}')
        manifold = tsne(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = manifold.fit_transform(data_subset)
        pp.plot(tsne_results[:, 0], tsne_results[:, 1], 'o')
        fig_name = f'tsne_{args.dataset.split(csv)[0]}{pdf}'
        pp.savefig(osp.join(fig_dir, fig_name))
        pp.close()