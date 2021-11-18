import urllib, sqlalchemy
import os.path as osp
import numpy as np
import pandas as pd

from config import *
from common.utils import load_meta, load_data, create_table_pointer, insert_data_row
from datetime import datetime

if __name__ == '__main__':

    # sql connection data

    server = 'tcp:jyusqlserver.database.windows.net'
    database = 'IoTSQL'
    driver = '{ODBC Driver 17 for SQL Server}'
    table = 'metsa_brp_sample_data'
    username = 'jyusqlserver'
    password = '#jyusql1'

    db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
    params = urllib.parse.quote_plus(db_connection_str)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    db_connection = engine.connect()
    sqlmeta = sqlalchemy.MetaData(engine)

    # task dir

    task = 'predict_bleach_ratio'
    task_dir = osp.join(data_dir, task)

    # load meta

    datameta = load_meta(osp.join(task_dir, meta_fname))

    # table pointer

    cols = datameta['features'] + [datameta['label']]
    table_pointer = create_table_pointer(table, sqlmeta, datameta['timestamp'], cols)

    # load data

    fpath = osp.join(task_dir, test_samples_fname)
    df = pd.read_csv(fpath)
    features = df[datameta['features']].values
    labels = df[datameta['label']].values
    n = len(labels)

    # inserting

    idx = None

    while True:

        if idx is None:
            idx = np.random.randint(n)
        else:
            idx += 1

        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + '0'  # sql has 7 digits precision for some reason
        X, y = features[idx, :], labels[idx]

        sample = {}
        for f, x in zip(datameta['features'], X):
            if not np.isnan(x):
                sample[f] = str(x)
        sample[ts_key] = current_time
        sample[br_key] = str(y)

        insert_data_row(table_pointer, db_connection, sample)

        answer = None
        while answer is None:
            answer = input('Next? (y/n)\n')
            if answer not in ['y', 'n']:
                answer = None
        if answer == 'n':
            break