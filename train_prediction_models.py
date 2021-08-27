import os
import os.path as osp

import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from calculate_prediction_error import set_seeds, load_meta

def load_data(dpath, task, tags):
    fnames = os.listdir(dpath)
    values, labels, timestamps = {}, {}, {}
    for stage in stages:
        fpath = [osp.join(dpath, fname) for fname in fnames if osp.isfile(osp.join(dpath, fname)) and fname.startswith(task) and fname.endswith(f'{stage}{csv}')]
        assert len(fpath) == 1
        fpath = fpath[0]
        df = pd.read_csv(fpath)
        values[stage] = df[tags].values
        labels[stage] = df[br_key].values
        timestamps[stage] = df[ts_key].values
    return values, labels, timestamps, tags.copy()

def pad_data(X, x_features, features, delay_classes, dc_comb):
    nan_cols = []
    dc_comb = [int(dc) for dc in dc_comb.split(',')]
    for i, xf in enumerate(x_features):
        dc = delay_classes[features.index(xf)]
        if dc not in dc_comb:
            nan_cols.append(i)
    X_padded = X.copy()
    X_padded[:, nan_cols] = np.nan
    return X_padded

def model_input(nfeatures, xmin, xmax, latent_dim=64, batchnorm=False):

    # input layer

    nfeatures_sum = np.sum(nfeatures)
    inputs = tf.keras.layers.Input(shape=(nfeatures_sum,))

    # deal with nans

    is_nan = tf.math.is_nan(inputs)
    masks = tf.dtypes.cast(-xmax, tf.float32)
    inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
    inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
    inputs_without_nan = tf.math.multiply_no_nan(inputs, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

    # standardize the input

    inputs_std = (inputs_without_nan - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std

    # split, stack and flatten

    hidden_spl = tf.split(hidden, nfeatures, axis=1)
    hidden = []
    for spl in hidden_spl:
        hidden.append(tf.keras.layers.Dense(latent_dim, activation='relu')(spl))
    hidden = tf.stack(hidden, axis=1)

    return inputs, hidden

def mlp(hidden, nhidden=2048):
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(nhidden, activation='relu')(hidden)
    return hidden

def cnn(hidden, nfilters=1024, kernel_size=3):
    hidden = tf.keras.layers.Conv1D(nfilters, kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Conv1D(nfilters, kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def attention_block(x, nh):
    q = tf.keras.layers.Dense(nh, use_bias=False)(x)
    k = tf.keras.layers.Dense(nh, use_bias=False)(x)
    v = tf.keras.layers.Dense(nh, use_bias=False)(x)
    a = tf.keras.layers.Multiply()([q, k])
    a = tf.keras.layers.Softmax(axis=-1)(a)
    h = tf.keras.layers.Multiply()([a, v])
    return h

class Attention(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",shape=(input_shape[-1], 1),initializer="normal")
        self.b = self.add_weight(name="att_bias",shape=(input_shape[1], 1),initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et=tf.squeeze(tf.tanh(tf.tensordot(x, self.W, 1) + self.b), axis=-1)
        at=tf.math.softmax(et)
        at=tf.expand_dims(at, axis=-1)
        output=x * at
        return tf.math.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

def att(hidden, nhidden=640):
    #hidden = attention_block(hidden, attention_size)
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = Attention()(hidden)
    #hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def rnn(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nhidden, activation='relu', return_sequences=False))(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def model_output(inputs, hidden, ymin, ymax, layers=[2048, 2048], dropout=0.5, lr=2.5e-4):
    for nh in layers:
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='linear')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-e', '--extractor', help='feature extractor', default='mlp', choices=['mlp', 'cnn', 'att', 'rnn'])
    parser.add_argument('-d', '--delay', help='Delay class when prediction starts', default='1')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-y', '--ylimits', help='Use bleach ratio limits from data?', default=False, type=bool)
    parser.add_argument('-r', '--retrain', help='Retrain model?', default=False, type=bool)
    args = parser.parse_args()

    # cuda

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # set seed for results reproduction

    set_seeds(seed)

    # tags and standardization values

    meta = load_meta(processed_data_dir, args.task)
    features = meta['features']
    classes = meta['classes']
    nfeatures = []
    dcs = []
    tbl_dc_combs = []
    for uc in np.sort(np.unique(classes)):
        dcs.extend(str(uc))
        nfeatures.append(len([c for c in classes if c == uc]))
        tbl_dc_combs.append(','.join([item for item in dcs]))
    xmin = np.array(meta['xmin'])
    xmax = np.array(meta['xmax'])

    if args.ylimits:
        ymin = meta['ymin']
        ymax = meta['ymax']
    else:
        ymin = br_min
        ymax = br_max

    # delay classes combination

    if args.delay is not None:
        dc_combs = [dc for dc in tbl_dc_combs if args.delay in dc]
    else:
        dc_combs = tbl_dc_combs

    print(f'The following feature classes will be used to train the model: {dc_combs}')

    # load data

    vals, labels, timestamps, val_features = load_data(processed_data_dir, args.task, features)

    # create datasets by padding certain feature classes

    Xtv, Ytv = {}, {}

    for stage in stages[:-1]:
        Xtv[stage] = []
        Ytv[stage] = []
        for dc_comb in dc_combs:
            Xtv[stage].append(pad_data(vals[stage], val_features, features, classes, dc_comb))
            Ytv[stage].append(labels[stage])
        Xtv[stage] = np.vstack(Xtv[stage])
        Ytv[stage] = np.hstack(Ytv[stage])

    stage = stages[-1]
    Xi = {}
    for dc_comb in dc_combs:
        Xi[dc_comb] = pad_data(vals[stage], val_features, features, classes, dc_comb)
    Yi = labels[stage]
    Ti = timestamps[stage]

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    for d in [models_dir, task_models_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # model name

    model_name = f'{args.extractor}_{args.delay}'

    # results tables

    e_path = osp.join(task_results_dir, prediction_errors_csv)
    r_path = osp.join(task_results_dir, prediction_results_csv)

    try:
        pe = pd.read_csv(e_path)
    except:
        pe = pd.DataFrame({
            'Delay classes': [comb for comb in tbl_dc_combs]
        })
    if model_name not in pe.keys():
        pe[model_name] = [np.nan for comb in tbl_dc_combs]

    try:
        pr = pd.read_csv(r_path)
    except:
        pr = pd.DataFrame({
            ts_key: [value for value in Ti],
            br_key: [value for value in Yi],
        })
    for dc_comb in tbl_dc_combs:
        model_comb = f'{model_name}_{dc_comb}'
        if model_comb not in pr.keys():
            pr[model_comb] = [np.nan for _ in Yi]

    # create model and results directories

    m_path = osp.join(task_models_dir, model_name)
    if not osp.isdir(m_path):
        os.mkdir(m_path)

    # load model

    if not args.retrain:
        try:
            model = tf.keras.models.load_model(m_path)
            have_to_create_model = False
            print(f'Model {model_name} has been loaded from {m_path}')
        except Exception as e:
            print(e)
            have_to_create_model = True
    else:
        have_to_create_model = True

    # create a new model if needed

    if have_to_create_model:

        print(f'Training new model {model_name}:')
        inputs, hidden = model_input(nfeatures, xmin, xmax)
        extractor_type = locals()[args.extractor]
        hidden = extractor_type(hidden)
        model = model_output(inputs, hidden, ymin, ymax)
        model_summary_lines = []
        model.summary(print_fn=lambda x: model_summary_lines.append(x))
        model_summary = "\n".join(model_summary_lines)
        if args.verbose:
            print(model_summary)

        # train model

        model.fit(
            Xtv[stages[0]], Ytv[stages[0]],
            validation_data=(Xtv[stages[1]], Ytv[stages[1]]),
            epochs=epochs,
            verbose=args.verbose,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=False,
                patience=patience,
                mode='min',
                restore_best_weights=True
            )]
        )

        # save model

        model.save(m_path)
        with open(osp.join(m_path, summary_txt), 'w') as f:
            f.write(model_summary)

    # calculate prediction error for each class combination

    for dc_comb in dc_combs:
        predictions = model.predict(Xi[dc_comb]).flatten()
        assert len(predictions) == len(Yi)
        error = np.mean(np.abs(Yi - predictions))
        print(f'Prediction error for combination {dc_comb}: {error}')

        # save results

        assert dc_comb in tbl_dc_combs
        idx = tbl_dc_combs.index(dc_comb)
        pe[model_name].values[idx] = error
        pe.to_csv(e_path, index=None)
        model_comb = f'{model_name}_{dc_comb}'
        pr[model_comb].values[:] = predictions
        pr.to_csv(r_path, index=None)