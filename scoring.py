import json

import numpy as np
import tensorflow as tf

def init():
    global model
    print("This is init")
    m_path = 'models/predict_bleach_ratio/development/split_mlp_2'
    model = tf.keras.models.load_model(m_path)

def run(data):
    jdata = json.loads(data)
    for key in jdata.keys():
        jdata[key] = np.array(jdata[key])
    result = model.predict(jdata)
    print(result)
    return result