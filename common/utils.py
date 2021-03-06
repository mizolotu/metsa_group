import json, sqlalchemy
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats as ss

from itertools import chain, combinations
from config import br_key, ts_key, series_step_prefix

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

def load_data(fpath, tags, dtype=None, nan_to_none=False):
    df = pd.read_csv(fpath, dtype=dtype)
    if nan_to_none:
        df = df.where(pd.notnull(df), None)
    tags_without_ts = [key for key in df.keys() if key.split(f'_{series_step_prefix}')[0] in tags]
    values = df[tags_without_ts].values
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

def create_table_pointer(table, meta, timestamp_column, other_columns):
    table_pointer = sqlalchemy.Table(
        table,
        meta,
        sqlalchemy.Column(timestamp_column, sqlalchemy.DateTime),
        *[sqlalchemy.Column(column, sqlalchemy.String) for column in other_columns]
    )
    meta.create_all(checkfirst=True)
    return table_pointer

def insert_data_row(table_pointer, conn, sample):
    ins = table_pointer.insert()
    conn.execute(ins.values(sample))

def insert_data_batch(table_pointer, conn, batch):
    ins = table_pointer.insert()
    conn.execute(ins, batch)

def select_last_data_rows(db_connection, table_pointer, timestamp_column, n=10):
    sel = table_pointer.select(limit=n, order_by=sqlalchemy.Column(timestamp_column).desc())
    result = db_connection.execute(sel)
    return [dict(row) for row in result.fetchall()]

def get_best_distribution(data, distributions = ['norm', 'exponweib', 'weibull_max', 'weibull_min', 'pareto', 'genextreme']):
    dist_results = []
    params = {}
    for distribution in distributions:
        dist = getattr(ss, distribution)
        param = dist.fit(data)

        params[distribution] = param
        D, p = ss.kstest(data, distribution, args=param)
        dist_results.append((distribution, p))

    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    return best_dist, best_p, params[best_dist]

def interpolate(x, xp, yp):
    y_interpolated = np.nan * np.ones((len(x), yp.shape[1]))
    for i in range(yp.shape[1]):
        col = yp[:, i]
        idx = np.where(np.isnan(col) == False)[0]
        if len(idx) > 0:
            y_interpolated[:, i] = np.interp(x, xp[idx], yp[idx, i])
    return y_interpolated

def substitute_nan_values(X):
    xmin = np.nanmin(X, 0)
    xmax = np.nanmax(X, 0)
    masks = xmin - (xmax - xmin)
    is_nan = np.isnan(X)
    inputs_without_nan = X * ~is_nan
    inputs_without_nan[is_nan] = 0
    inputs_without_nan += masks * is_nan
    return inputs_without_nan
