import os, urllib, sqlalchemy, pandas

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
table = 'WindSpeedPrediction'
username = 'jyusqlserver'
password = '#jyusql1'

db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

params = urllib.parse.quote_plus(db_connection_str)
engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
meta = sqlalchemy.MetaData(engine)
table_pointer = sqlalchemy.Table(
    table,
    meta,
    sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column('temperature', sqlalchemy.Float),
    sqlalchemy.Column('humidity', sqlalchemy.Float),
    sqlalchemy.Column('pressure', sqlalchemy.Float),
    sqlalchemy.Column('windspeed', sqlalchemy.Float)
)
meta.create_all(checkfirst=True)  # Only create tables if they don't exist.
conn = engine.connect()

import pickle, os
import tensorflow as tf
import numpy as np

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

# scaler

scaler_fpath = os.path.join(scriptdir, 'scaler.pkl')
with open(scaler_fpath, 'rb') as f:
    scaler = pickle.load(f)

# model

model_pb = os.path.join(scriptdir, 'model.pb')
graph_def = tf.compat.v1.GraphDef()
input_layer = 'input:0'
output_layer = 'output:0'

def get_data_rows(null_column):
    sql_df = pandas.read_sql(
        f'SELECT * FROM {table} WHERE {null_column} IS NULL',
        con=engine
    )
    return sql_df

def update_row(id, value):
    upd = table_pointer.update()
    conn.execute(
        upd.where(table_pointer.c.id == id).values(windspeed=value)
    )

def init():
    with tf.io.gfile.GFile(model_pb, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def predict(x):
    init()
    nsamples = x.shape[0]
    xy = np.hstack([x, np.zeros((nsamples, 1))])
    xy_std = scaler.transform(xy)
    x_std = xy_std[:, :-1]
    with tf.compat.v1.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        y_std = sess.run(prob_tensor, {input_layer: x_std})
    xy_std = np.hstack([x_std, y_std])
    xy = scaler.inverse_transform(xy_std)
    y = xy[:, -1]
    return y