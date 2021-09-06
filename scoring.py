import json
import os.path as osp
import numpy as np
import tensorflow as tf

def init():
    global models, model_selector
    model_dir = 'models/predict_bleach_ratio/development'
    model_names = {
        1: 'split_mlp_1',
        2: 'split_mlp_2',
        3: 'split_mlp_3',
        4: 'split_mlp_4',
        5: 'split_mlp_5'
    }
    models = {}
    for key in model_names.keys():
        models[key] = tf.keras.models.load_model(osp.join(model_dir, model_names[key]))
    model_selector = get_model_selector(models)

def run(data):
    jdata = json.loads(data)
    print(jdata)
    for key in jdata.keys():
        jdata[key] = np.array(jdata[key])
    result = model.predict(jdata)
    print(result)
    return result

def get_model_selector(models):
    outputs =
