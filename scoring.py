import os, json, sys
import os.path as osp
import numpy as np
import tensorflow as tf

def init():
    global models, model_selector
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    model_names = {
        1: 'production/split_cnn1_1',
        2: 'production/split_cnn1_2',
        3: 'production/split_cnn1_3',
        4: 'production/split_cnn1_4',
        5: 'production/split_cnn1_5'
    }
    models = {}
    for key in model_names.keys():
        models[key] = tf.keras.models.load_model(osp.join(model_dir, model_names[key]))
    model_selector = get_model_selector(models)

def run(data):
    try:
        sample = json.loads(data)
        model, dc = model_selector(sample)
        if model is not None:
            sample = adjust_input(sample, model)
            result = model.predict(sample)
            key = [key for key in result.keys()][0]
            value = result[key][0, 0].tolist()
            result = {key: value, 'model': dc, 'status': 'ok'}
        else:
            result = {'status': 'no input data provided'}
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, fname, exc_tb.tb_lineno)
        result = {'status': 'error'}
    return result

def adjust_input(data, model):
    for key in model.input:
        if key not in data or data[key] is None:
            data[key] = np.array([np.nan])
        elif type(data[key]) is not list:
            data[key] = np.array([data[key]])
    return data

def get_model_selector(models):
    keys = sorted(models.keys())
    model_additional_inputs = []
    for key in keys:
        input = models[key].input
        new_tags = [key for key in input if key not in [item for sublist in model_additional_inputs for item in sublist]]
        model_additional_inputs.append(new_tags)

    def model_selector(data):
        non_nan_keys = [key for key in data.keys() if data[key] is not None and np.isnan(data[key]) == False]
        if len(non_nan_keys) > 0:
            remaining_keys = non_nan_keys.copy()
            for k, inputs in zip(keys, model_additional_inputs):
                remaining_keys = [key for key in remaining_keys if key not in inputs]
                if len(remaining_keys) == 0:
                    break
            model = models[k]
        else:
            model, k = None, None
        return model, k

    return model_selector