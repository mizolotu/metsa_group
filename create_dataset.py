import os, json, shutil, sys
import argparse as arp
import os.path as osp
import numpy as np
import pandas as pd
import dateutil.parser as dp

from config import *
from common.utils import interpolate

if __name__ == '__main__':

    # parse args

    parser = arp.ArgumentParser(description='')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-s', '--series', help='Extract as timeseries?', type=bool, default=False)
    args = parser.parse_args()

    # laod delay classes

    delay_classes_by_feature = {}
    try:
        p = pd.read_excel(osp.join(data_dir, tags_fname), engine='openpyxl')
        positions = p[position_column].values
        delay_classes = p[delay_class_column].values
        unique_delay_classes = np.unique(delay_classes)
        for i, unique_delay_class in enumerate(unique_delay_classes):
            idx = np.where(delay_classes == unique_delay_class)[0]
            for key in [item.replace('.', '_') for item in positions[idx].tolist()]:
                delay_classes_by_feature[key] = int(unique_delay_class)
    except Exception as e:
        print(e)
        print('No file with feature delay classes found!')
        sys.exit(1)

    # load data

    keys = None
    timestamps, values = [], []
    for fname in os.listdir(raw_data_dir):
        if osp.isfile(osp.join(raw_data_dir, fname)) and fname.endswith(csv):
            try:
                p = pd.read_csv(osp.join(raw_data_dir, fname))
                timestamps = np.hstack([timestamps, p[ts_key].values])
                if keys is not None:
                    assert keys == [key for key in p.keys() if key != ts_key]
                else:
                    keys = [key for key in p.keys() if key != ts_key]
                values.append(p[keys].values)
            except Exception as e:
                print(e)
    if len(values) > 0:
        values = np.vstack(values)
    else:
        print('No data have been found!')
        sys.exit(1)

    # select features

    assert len(keys) == values.shape[1], 'Number of keys is not equal to the number of value columns'
    assert br_key in delay_classes_by_feature.keys()
    assert br_key in keys
    br_index = keys.index(br_key)
    cols = [col for col, key in enumerate(keys) if key in delay_classes_by_feature.keys() and key != br_key and np.any(pd.isna(values[:, col]) == False)]
    features = [keys[i] for i in cols]
    labels = values[:, br_index]
    values = values[:, cols]
    delay_classes = [delay_classes_by_feature[key] for key in features]

    # select values

    assert len(features) == values.shape[1], 'Number of features is not equal to the number of value columns'
    u, idx = np.unique(timestamps, return_index=True)
    values = values[idx, :]
    labels = labels[idx]
    timestamps = timestamps[idx]
    idx = np.where((pd.isna(labels) == False) & (labels > br_thr))[0]
    values = values[idx, :]
    labels = labels[idx]
    timestamps = timestamps[idx]

    # sort features

    feature_indexes_sorted = []
    unique_delay_classes = np.sort(np.unique(delay_classes))
    for udc in unique_delay_classes:
        ids = [i for i, dc in enumerate(delay_classes) if dc == udc]
        feature_indexes_sorted.extend(ids)
    features = [features[i] for i in feature_indexes_sorted]
    delay_classes = [delay_classes[i] for i in feature_indexes_sorted]
    values = values[:, feature_indexes_sorted]
    feature_cols = features

    if args.series:
        ts = np.array([dp.parse(item).timestamp() for item in timestamps])
        deltas = labels[1:] - labels[:-1]
        idx = np.where(deltas != 0)[0] + 1
        series = np.zeros((len(idx), series_len, values.shape[1]))
        for i, id in enumerate(idx):
            t = ts[id]
            t_idx = np.where((ts > (t - series_len * 60)) & (ts <= t))[0]
            t_deltas = t - ts[t_idx]
            series[i, :, :] = interpolate(np.arange(series_len)[::-1] * 60, t_deltas, values[t_idx, :])
        labels = labels[idx]
        timestamps = timestamps[idx]
        values = series.reshape(series.shape[0], -1)
        feature_cols = [f for f in feature_cols for _ in range(series_len)]

    print(f'Data sample timestamps are between {np.min(timestamps)} and {np.max(timestamps)}')
    print(f'Number of samples with label in interval [{np.min(labels)}, {br_min}): {len(np.where(labels < br_min)[0])}')
    print(f'Number of samples with label in interval [{br_min}, {br_max}): {len(np.where((labels >= br_min) & (labels <= br_max))[0])}')
    print(f'Number of samples with label in interval [{br_max}, {np.max(labels)}): {len(np.where(labels > br_max)[0])}')

    # meta

    meta = {'features': features, 'classes': delay_classes, 'label': br_key, 'timestamp': ts_key}

    # output directory

    task_dir = osp.join(data_dir, args.task)
    if not osp.isdir(task_dir):
        os.mkdir(task_dir)

    # save datasets

    fname = features_fname
    fpath = osp.join(task_dir, fname)
    data = np.hstack([timestamps[:, None], values, labels[:, None]])
    print(f'Number of samples: {data.shape[0]}')
    pd.DataFrame(data, columns=[ts_key] + feature_cols + [br_key]).to_csv(fpath, mode='w', index=False)

    # save meta

    meta_fpath = osp.join(task_dir, meta_fname)
    with open(meta_fpath, 'w') as jf:
        json.dump(meta, jf)

    # copy meta to each function directory

    if osp.isdir(functions_dir):
        dnames = os.listdir(functions_dir)
        for dname in dnames:
            dpath = osp.join(functions_dir, dname)
            if osp.isdir(dpath):
                shutil.copy(meta_fpath, dpath)