import os, time, json
import argparse as arp
import os.path as osp
import dateutil.parser as dparser
import numpy as np
import pandas as pd

from itertools import chain, combinations
from config import *

def load_tags(fname, remove_yellow=False):
    if fname in os.listdir(data_dir):
        p = pd.read_excel(osp.join(data_dir, fname), engine = 'openpyxl')
        positions = p[position_column].values
        delay_classes = p[delay_class_column].values
        if remove_yellow:
            non_yellow_indexes = np.array([i for i, p in enumerate(positions) if p not in yellow_tags])
            positions = positions[non_yellow_indexes]
            delay_classes = delay_classes[non_yellow_indexes]
        unique_delay_classes = np.unique(delay_classes)
        tags = {}
        for i, unique_delay_class in enumerate(unique_delay_classes):
            idx = np.where(delay_classes == unique_delay_class)[0]
            tags[int(unique_delay_class)] = positions[idx].tolist()
    else:
        tags = None
    return tags

def load_samples(dpath):
    keys = None
    timestamps, values = [], []
    for fname in os.listdir(dpath):
        if osp.isfile(osp.join(dpath, fname)) and fname.endswith(csv):
            p = pd.read_csv(osp.join(dpath, fname))
            timestamps = np.hstack([timestamps, p[ts_key].values])
            if keys is not None:
                assert keys == [key for key in p.keys() if key != ts_key]
            else:
                keys = [key for key in p.keys() if key != ts_key]
            values.append(p[keys].values)
    if keys is not None:
        keys = [key.replace('_', '.') for key in keys]
    if len(values) > 0:
        values = np.vstack(values)
    return keys, values, timestamps

def select_keys(keys, values, tags):
    tags = np.hstack(list(tags.values()))
    assert br_key in tags
    assert br_key in keys
    br_index = keys.index(br_key)
    cols = [col for col, key in enumerate(keys) if key in tags and key != br_key]
    return [keys[i] for i in cols], values[:, cols], values[:, br_index]

def powerset(items):
    return chain.from_iterable(combinations(items, r) for r in range(1, len(items)+1))

def sort_by_delay_class(keys, values, tags):
    key_indexes_sorted = []
    nans = pd.isna(values)
    for tag in tags:
        if tag in keys:
            idx = keys.index(tag)
            if np.any(nans[:, idx] == False):
                key_indexes_sorted.append(keys.index(tag))
            else:
                print(f'No values in column {tag}?')
    return [keys[i] for i in key_indexes_sorted], values[:, key_indexes_sorted]

def select_values(values, labels, timestamps, label_thr=80.0):
    u, idx = np.unique(timestamps, return_index=True)
    values = values[idx, :]
    labels = labels[idx]
    timestamps = timestamps[idx]
    idx = np.where((pd.isna(labels) == False) & (labels > label_thr))[0]
    return values[idx, :], labels[idx], timestamps[idx]

if __name__ == '__main__':

    # parse args

    parser = arp.ArgumentParser(description='')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    args = parser.parse_args()

    # laod data

    tags = load_tags(tags_fname)
    keys, values, timestamps = load_samples(raw_data_dir)
    assert keys is not None, 'No data?'
    assert len(keys) == values.shape[1], 'Number of keys is not equal to the number of value columns'
    keys, values, labels = select_keys(keys, values, tags)
    assert len(keys) == values.shape[1], 'Number of keys is not equal to the number of value columns'
    values, labels, timestamps = select_values(values, labels, timestamps)
    print(f'Data sample timestamps are between {np.min(timestamps)} and {np.max(timestamps)}')

    # set seed for results reproduction

    np.random.seed(seed)

    # train, test, validation split

    inds = np.arange(len(labels))
    inds_splitted = [[] for _ in stages]
    np.random.shuffle(inds)
    val, remaining = np.split(inds, [int(validation_share * len(inds))])
    tr, te = np.split(remaining, [int(train_test_ratio * len(remaining))])
    inds_splitted[0] = tr
    inds_splitted[1] = te
    inds_splitted[2] = val

    # bleach ratio limits

    y_min = np.min(labels)
    y_max = np.max(labels)
    print(f'Data sample labels are between {y_min} and {y_max}')

    # meta

    meta = {'tags': {}, 'ntrain': len(tr), 'nvalidate': len(val), 'ninference': len(te), 'xmin': [], 'xmax': []}

    # output directory

    if not osp.isdir(processed_data_dir):
        os.mkdir(processed_data_dir)

    # process data for each delay class

    tag_names = []
    vals = []

    for key in tags.keys():

        # select tags and values for each delay class

        keys_, values_ = sort_by_delay_class(keys, values, tags[key])
        meta['tags'][key] = keys_
        tag_names.extend(keys_)

        # substitute nan values

        xmin = np.nanmin(values_, axis=0)
        xmax = np.nanmax(values_, axis=0)
        values_std = (values_ - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + eps)
        values_std[np.where(pd.isna(values_std))] = nan_value
        values_no_nan = values_std * (xmax[None, :] - xmin[None, :]) + xmin[None, :]
        vals.append(values_no_nan)
    tag_names.append(br_key)
    vals = np.hstack(vals)

    # save datasets

    for fi, stage in enumerate(stages):
        fname = f'{args.task}_{stage}{csv}'
        fpath = osp.join(processed_data_dir, fname)
        data = np.hstack([timestamps[inds_splitted[fi], None], vals[inds_splitted[fi], :], labels[inds_splitted[fi], None]])
        pd.DataFrame(data, columns=[ts_key] + tag_names).to_csv(fpath, mode='w', index=False)
        if stage == 'train':
            meta['xmin'] = np.min(vals[inds_splitted[fi], :], axis=0).tolist()
            meta['xmax'] = np.max(vals[inds_splitted[fi], :], axis=0).tolist()
            meta['ymin'] = np.min(labels[inds_splitted[fi]])
            meta['ymax'] = np.max(labels[inds_splitted[fi]])

    # save meta

    meta_fpath = osp.join(processed_data_dir, f'{args.task}_metainfo.json')
    with open(meta_fpath, 'w') as jf:
        json.dump(meta, jf)