import os, urllib, sqlalchemy, json, pandas

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
table = 'metsa_brp_sample_data'
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

def get_data_rows(n):
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