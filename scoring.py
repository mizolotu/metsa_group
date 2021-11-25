import os, json, sys
import os.path as osp
import numpy as np
import tensorflow as tf

def init():
    global models, model_selector
    model_names = {
        1: 'production/mlp_1',
        2: 'production/mlp_2',
        3: 'production/mlp_3',
        4: 'production/mlp_4',
        5: 'production/mlp_5'
    }
    models = {}
    for key in model_names.keys():
        models[key] = tf.keras.models.load_model(osp.join(os.getenv('AZUREML_MODEL_DIR'), model_names[key]))
    model_selector = get_model_selector(models)

def run(data):
    try:
        jdata = json.loads(data)
        if 'dc' in jdata.keys() and jdata['dc'] in models.keys():
            dc = jdata['dc']
            model = models[dc]
        else:
            model, dc = model_selector(jdata['cols'], jdata['rows'])
        if model is not None:
            prediction_key, predictions, errors = make_predictions(jdata['cols'], jdata['rows'], model)
            resp = {'status': 'ok', prediction_key: predictions, 'errors': errors}
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        resp = {'status': f'exception {e} in {fname} at line {exc_tb.tb_lineno}'}
    return resp

def make_predictions(cols, rows, model):

    # prepare batch

    x = {}
    rows = np.array(rows)
    for key in model.input:
        if key not in cols:
            x[key] = np.array([np.nan for _ in rows])
        else:
            idx = cols.index(key)
            x[key] = np.array(rows[:, idx], dtype='float')
    keys = list(model.output.keys())
    assert len(keys) == 1
    label_key = keys[0]
    if label_key in cols:
        idx = cols.index(label_key)
        y = np.array(rows[:, idx], dtype='float')
    else:
        y = []

    # predict and process result

    result = model.predict(x)
    predictions = result[label_key][:, 0]
    if len(y) > 0 and len(y) == len(predictions):
        errors = np.abs(y - predictions).tolist()
    else:
        errors = [np.nan for _ in y]
    return label_key, predictions.tolist(), errors

def get_model_selector(models):
    keys = sorted(models.keys())
    model_additional_inputs = []
    all_tags = []
    for key in keys:
        input = models[key].input
        new_tags = [key for key in input if key not in [item for sublist in model_additional_inputs for item in sublist]]
        model_additional_inputs.append(new_tags)
        all_tags.extend(new_tags)

    def model_selector(cols, rows):
        rows = np.array(rows)
        non_nan_keys = []
        for i, key in enumerate(all_tags):
            if key in cols and np.any(rows[:, cols.index(key)] != None):
                non_nan_keys.append(key)
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