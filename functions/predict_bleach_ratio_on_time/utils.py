import os, urllib, sqlalchemy, json, pandas

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
meta = sqlalchemy.MetaData(engine)

def load_columns(fname):
    fpath = os.path.join(scriptdir, fname)
    columns = None
    try:
        with open(fpath) as f:
            columns = json.load(f)
    except Exception as e:
        print(e)
    return columns

def create_data_table_pointer(columns):
    table_pointer = sqlalchemy.Table(
        data_table,
        meta,
        sqlalchemy.Column(columns['timestamp'], sqlalchemy.String),
        *[sqlalchemy.Column(key, sqlalchemy.Float) for key in columns['features']],
        sqlalchemy.Column(columns['label'], sqlalchemy.Float)
    )
    #meta.create_all(checkfirst=True)
    conn = engine.connect()
    return table_pointer, conn

def create_prediction_table_pointer(columns):
    uclasses = sorted(list(set(columns['classes'])))
    cols = [f'Prediction {uc}' for uc in uclasses] + [f'Prediction {uc} error' for uc in uclasses]
    table_pointer = sqlalchemy.Table(
        data_table,
        meta,
        sqlalchemy.Column(columns['timestamp'], sqlalchemy.String),
        sqlalchemy.Column(columns['label'], sqlalchemy.Float),
        *cols
    )
    meta.create_all(checkfirst=True)
    conn = engine.connect()
    return table_pointer, conn

columns = load_columns('metainfo.json')
data_table_pointer, data_table_connection = create_data_table_pointer(columns)
prediction_table_pointer, prediction_table_connection = create_prediction_table_pointer(columns)

def insert_prediction_row(sample):
    ins = prediction_table_pointer.insert()
    prediction_table_connection.execute(ins.values(sample))

def get_n_last_data_rows(table, n, desc_col_name):
    sql_df = pandas.read_sql(
        f'SELECT TOP ({n}) * FROM {table} order by {desc_col_name} desc',
        con=engine
    )
    return sql_df

def update_data_row(id, value):
    upd = table_pointer.update()
    conn.execute(
        upd.where(table_pointer.c.id == id).values(windspeed=value)
    )