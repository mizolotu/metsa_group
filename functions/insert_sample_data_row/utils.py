import os, urllib, sqlalchemy, json

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
table = 'metsa_brp_sample_data'
username = 'jyusqlserver'
password = '#jyusql1'

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

def create_table_pointer(columns):
    table_pointer = sqlalchemy.Table(
        table,
        meta,
        sqlalchemy.Column(columns['timestamp'], sqlalchemy.String),
        *[sqlalchemy.Column(key, sqlalchemy.Float) for key in columns['features']],
        sqlalchemy.Column(columns['label'], sqlalchemy.Float)
    )
    meta.create_all(checkfirst=True)
    conn = engine.connect()
    return table_pointer, conn

columns = load_columns('metainfo.json')
table_pointer, conn = create_table_pointer(columns)

def insert_data_row(sample):
    ins = table_pointer.insert()
    conn.execute(ins.values(sample))



