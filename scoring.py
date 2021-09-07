import json
import os.path as osp
import numpy as np
import pandas as pd
import tensorflow as tf

def init():
    global models, model_selector
    model_dir = 'models/predict_bleach_ratio/production'
    model_names = {
        1: 'split_cnn1_1',
        2: 'split_cnn1_2',
        3: 'split_cnn1_3',
        4: 'split_cnn1_4',
        5: 'split_cnn1_5'
    }
    models = {}
    for key in model_names.keys():
        models[key] = tf.keras.models.load_model(osp.join(model_dir, model_names[key]))
    model_selector = get_model_selector(models)

def run(data):
    jdata = json.loads(data)
    print(jdata)
    model = get_model_selector(jdata)
    for key in jdata.keys():
        jdata[key] = np.array(jdata[key])
    result = model.predict(jdata)
    print(result)
    return result

def get_model_selector(models):
    keys = sorted(models.keys())
    model_additional_inputs = []
    for key in keys:
        input = models[key].input
        new_tags = [key for key in input if key not in [item for sublist in model_additional_inputs for item in sublist]]
        model_additional_inputs.append(new_tags)
        print(key, new_tags)

    def model_selector(data):
        non_nan_keys = [key for key in data.keys() if np.isnan(data[key]) == False]

    return model_selector


