import json
import tensorflow as tf
import numpy as np
import pandas as pd

from itertools import chain, combinations
from config import br_key, ts_key

def load_meta(fpath):
    meta = None
    try:
        with open(fpath) as f:
            meta = json.load(f)
    except Exception as e:
        print(e)
    return meta

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def load_data(fpath, tags):
    df = pd.read_csv(fpath)
    values = df[tags].values
    labels = df[br_key].values
    timestamps = df[ts_key].values
    return values, labels, timestamps

def pad_data(X, x_features, features, delay_classes, dc_comb):
    nan_cols = []
    dc_comb = [int(dc) for dc in dc_comb.split(',')]
    for i, xf in enumerate(x_features):
        dc = delay_classes[features.index(xf)]
        if dc not in dc_comb:
            nan_cols.append(i)
    X_padded = X.copy()
    X_padded[:, nan_cols] = np.nan
    return X_padded

def powerset(items):
    return chain.from_iterable(combinations(items, r) for r in range(1, len(items)+1))