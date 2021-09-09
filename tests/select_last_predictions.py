import urllib, sqlalchemy
import os.path as osp
import numpy as np
import pandas as pd

from config import *
from common.utils import load_meta, create_table_pointer, select_last_data_rows

if __name__ == '__main__':

    # sql connection data

    server = 'tcp:jyusqlserver.database.windows.net'
    database = 'IoTSQL'
    driver = '{ODBC Driver 17 for SQL Server}'
    table = 'metsa_brp_sample_predictions'
    username = 'jyusqlserver'
    password = ''

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

    uclasses = sorted(list(set(datameta['classes'])))
    prediction_cols = [f'Prediction {uc}' for uc in uclasses]
    error_cols = [f'Prediction {uc} error' for uc in uclasses]
    cols = [datameta['label']] + prediction_cols + error_cols
    table_pointer = create_table_pointer(table, sqlmeta, datameta['timestamp'], cols)

    # get and print data

    rows = select_last_data_rows(db_connection, table_pointer, datameta['timestamp'], 10)
    for row in rows:
        print(row)
