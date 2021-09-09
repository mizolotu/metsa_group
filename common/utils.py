import json, sqlalchemy
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

def create_table_pointer(table, meta, timestamp_column, other_columns):
    table_pointer = sqlalchemy.Table(
        table,
        meta,
        sqlalchemy.Column(timestamp_column, sqlalchemy.String),
        *[sqlalchemy.Column(column, sqlalchemy.Float) for column in other_columns]
    )
    meta.create_all(checkfirst=True)
    return table_pointer

def insert_data_row(table_pointer, conn, sample):
    ins = table_pointer.insert()
    conn.execute(ins.values(sample))

def select_last_data_rows(db_connection, table_pointer, timestamp_column, n=10):
    sel = table_pointer.select(limit=n, order_by=sqlalchemy.Column(timestamp_column).desc())
    result = db_connection.execute(sel)
    return [dict(row) for row in result.fetchall()]