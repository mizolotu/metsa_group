import os, urllib, sqlalchemy, json, requests

import dateutil.parser as dparser

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
data_table = 'metsa_brp_sample_data'
prediction_table = 'metsa_brp_sample_predictions'
username = 'jyusqlserver'
password = ''

db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
params = urllib.parse.quote_plus(db_connection_str)
engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
db_connection = engine.connect()
meta = sqlalchemy.MetaData(engine)

endpoint_url = 'http://20.93.236.203/api/v1/service/metsa-brp/score'

with open(os.path.join(scriptdir, 'metainfo.json')) as f:
    columns = json.load(f)

data_table_pointer = sqlalchemy.Table(
    data_table,
    meta,
    sqlalchemy.Column(columns['timestamp'], sqlalchemy.String),
    *[sqlalchemy.Column(key, sqlalchemy.Float) for key in columns['features']],
    sqlalchemy.Column(columns['label'], sqlalchemy.Float)
)

uclasses = sorted(list(set(columns['classes'])))
prediction_cols = [f'Prediction {uc}' for uc in uclasses]
error_cols = [f'Prediction {uc} error' for uc in uclasses]
prediction_table_pointer = sqlalchemy.Table(
    prediction_table,
    meta,
    sqlalchemy.Column(columns['timestamp'], sqlalchemy.String),
    sqlalchemy.Column(columns['label'], sqlalchemy.Float),
    *[sqlalchemy.Column(key, sqlalchemy.Float) for key in prediction_cols],
    *[sqlalchemy.Column(key, sqlalchemy.Float) for key in error_cols]
)
meta.create_all(checkfirst=True)

def select_last_prediction_row():
    sel = prediction_table_pointer.select(limit=1, order_by=sqlalchemy.Column(columns['timestamp']).desc())
    result = db_connection.execute(sel)
    return [dict(row) for row in result.fetchall()]

def select_last_data_rows(n=10):
    sel = data_table_pointer.select(limit=n, order_by=sqlalchemy.Column(columns['timestamp']).desc())
    result = db_connection.execute(sel)
    return [dict(row) for row in result.fetchall()]

def select_new_data_samples(data_rows, prediction_row):
    last_prediction_ts = dparser.parse(prediction_row[columns['timestamp']], fuzzy=True)
    rows_selected = [row for row in data_rows if dparser.parse(row[columns['timestamp']], fuzzy=True) > last_prediction_ts]
    return rows_selected

def prepare_samples(row):
    timestamp = row[columns['timestamp']]
    uclasses = sorted(list(set(columns['classes'])))
    samples = []
    for uc in uclasses:
        features = [f for f, c in zip(columns['features'], columns['classes']) if c <= uc]
        sample = {}
        for feature in features:
            if feature in row.keys():
                sample[feature] = row[feature]
        samples.append(sample)
    if columns['label'] in row.keys():
        label = row[columns['label']]
    else:
        label = None
    return timestamp, samples, label

def predict(sample):
    r = requests.post(url=endpoint_url, json=sample)
    jdata = r.json()
    if jdata['status'] == 'ok':
        prediction = jdata[columns['label']]
        model = jdata['model']
    else:
        prediction, model = None, None
    return prediction, model

def insert_prediction_row(timestamp, label, predictions, errors):
    prediction_row = {
        columns['timestamp']: timestamp,
        columns['label']: label
    }
    uclasses = sorted(list(set(columns['classes'])))
    for uc, prediction, error, prediction_col, error_col in zip(uclasses, predictions, errors, prediction_cols, error_cols):
        prediction_row[prediction_col] = prediction
        prediction_row[error_col] = error
    ins = prediction_table_pointer.insert()
    db_connection.execute(ins.values(prediction_row))