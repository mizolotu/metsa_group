import os
import os.path as osp

import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from calculate_prediction_error import set_seeds, load_meta

def load_data(dpath, task):
    fnames = os.listdir(dpath)
    X, Y = {}, {}
    for stage in stages:
        fpath = [osp.join(dpath, fname) for fname in fnames if osp.isfile(osp.join(dpath, fname)) and fname.startswith(task) and fname.endswith(f'{stage}{csv}')]
        assert len(fpath) == 1
        fpath = fpath[0]
        vals = pd.read_csv(fpath).values
        X[stage] = vals[:, :-1]
        Y[stage] = vals[:, -1]
    return X, Y

def mlp(nfeatures, nhiddens, xmin, xmax, ymin, ymax, latent_dim=64, batchnorm=True, dropout=0.5, lr=2.5e-4):
    nfeatures_sum = np.sum(nfeatures)
    inputs = tf.keras.layers.Input(shape=(nfeatures_sum,))
    inputs_std = (inputs - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std
    hidden_spl = tf.split(hidden, nfeatures, axis=1)
    hidden = []
    for spl in hidden_spl:
        hidden.append(tf.keras.layers.Dense(latent_dim, activation='relu')(spl))
    hidden = tf.stack(hidden, axis=1)
    hidden = tf.keras.layers.Flatten()(hidden)
    for nh in nhiddens:
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

def cnn(nfeatures, nhiddens, xmin, xmax, ymin, ymax, latent_dim=64, nfilters=512, kernel_size=3, batchnorm=True, dropout=0.5, lr=2.5e-4):
    nfeatures_sum = np.sum(nfeatures)
    inputs = tf.keras.layers.Input(shape=(nfeatures_sum,))
    inputs_std = (inputs - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std
    hidden_spl = tf.split(hidden, nfeatures, axis=1)
    hidden = []
    for spl in hidden_spl:
        hidden.append(tf.keras.layers.Dense(latent_dim, activation='relu')(spl))
    hidden = tf.stack(hidden, axis=1)
    hidden = tf.keras.layers.Conv1D(nfilters, kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Conv1D(nfilters, kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    for nh in nhiddens:
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

def attention_block(x, nh):
    q = tf.keras.layers.Dense(nh, use_bias=False)(x)
    k = tf.keras.layers.Dense(nh, use_bias=False)(x)
    v = tf.keras.layers.Dense(nh, use_bias=False)(x)
    a = tf.keras.layers.Multiply()([q, k])
    a = tf.keras.layers.Softmax(axis=-1)(a)
    h = tf.keras.layers.Multiply()([a, v])
    return h

def att(nfeatures, nhiddens, xmin, xmax, ymin, ymax, latent_dim=64, attention_size=512, batchnorm=True, dropout=0.5, lr=2.5e-4):
    nfeatures_sum = np.sum(nfeatures)
    inputs = tf.keras.layers.Input(shape=(nfeatures_sum,))
    inputs_std = (inputs - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std
    hidden_spl = tf.split(hidden, nfeatures, axis=1)
    hidden = []
    for spl in hidden_spl:
        hidden.append(tf.keras.layers.Dense(latent_dim, activation='relu')(spl))
    hidden = tf.stack(hidden, axis=1)
    hidden = attention_block(hidden, attention_size)
    hidden = tf.keras.layers.Flatten()(hidden)
    for nh in nhiddens:
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-e', '--extractor', help='feature extractor', default='mlp')
    parser.add_argument('-l', '--layers', help='Number of neurons in the last fully-connected layers', default=[2048, 2048], type=int, nargs='+')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-y', '--ylimits', help='Use bleach ratio limits found from the data?', default=False, type=bool)
    args = parser.parse_args()

    # cuda

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # set seed for results reproduction

    set_seeds(seed)

    # tags and standardization values

    meta = load_meta(processed_data_dir, args.task)
    tags = meta['tags']
    tag_keys = sorted(tags.keys())
    tags_ = []
    nfeatures = []
    for key in tag_keys:
        tags_.extend(tags[key])
        nfeatures.append(len(tags[key]))

    xmin = np.array(meta['xmin'])
    xmax = np.array(meta['xmax'])

    if args.ylimits:
        ymin = meta['ymin']
        ymax = meta['ymax']
    else:
        ymin = br_min
        ymax = br_max

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    for d in [models_dir, task_models_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # load data

    X, Y = load_data(processed_data_dir, args.task)

    # model

    model_type = locals()[args.extractor]
    model_name = f"{args.extractor}_{'-'.join([str(item) for item in args.layers])}"

    # results table

    r_name = prediction_results_csv
    r_path = osp.join(task_results_dir, r_name)
    try:
        p = pd.read_csv(r_path)
        if model_name not in p.keys():
            p[model_name] = [np.nan for tag in tags_]
    except:
        p = pd.DataFrame({
            'Label': [value for value in Y['inference']],
            model_name: [np.nan for _ in Y['inference']]
        })

    # create model

    model = model_type(nfeatures, args.layers, xmin, xmax, ymin, ymax)
    if args.verbose:
        model.summary()

    # create model and results directories

    m_path = osp.join(task_models_dir, model_name)
    if not osp.isdir(m_path):
        os.mkdir(m_path)

    # load model

    try:
        model = tf.keras.models.load_model(m_path)

    except Exception as e:
        print(e)

        # train model

        model.fit(
            X['train'], Y['train'],
            validation_data=(X['validate'], Y['validate']),
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

    # load data and calculate prediction error

    predictions = model.predict(X['inference']).flatten()
    assert len(predictions) == len(Y['inference'])
    error = np.mean(np.abs(Y['inference'] - predictions))
    X = np.vstack(X)
    Y = np.hstack(Y)
    print(f'Prediction error: {error}')

    # save results

    p[model_name].values[:] = predictions
    p.to_csv(r_path, index=None)