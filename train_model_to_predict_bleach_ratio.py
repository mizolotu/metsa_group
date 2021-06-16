import json, os
import os.path as osp

import tensorflow as tf
import numpy as np
import argparse as arp

from create_datasets import powerset
from config import *

def load_batches(path, batch_size, nfeatures):
    batches = tf.data.experimental.make_csv_dataset(
        path,
        batch_size=batch_size,
        header=False,
        shuffle=True,
        column_names=[str(i) for i in range(nfeatures + 1)],
        column_defaults=[tf.float32 for _ in range(nfeatures + 1)],
        select_columns=[str(i) for i in range(nfeatures + 1)],
        label_name='{0}'.format(nfeatures),
        num_epochs=1
    )
    return batches

def load_meta(fpath, prefix):
    meta = None
    try:
        with open(osp.join(fpath, f'{prefix}_metainfo.json')) as f:
            meta = json.load(f)
    except Exception as e:
        print(e)
    return meta

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def regression_mapper(features, label, ymin, ymax):
    features = tf.stack(list(features.values()), axis=-1)
    label = tf.clip_by_value(label, ymin, ymax)
    return features, label

def mlp(nfeatures, nl, nh, ymin, ymax, dropout=0.5, batchnorm=False, lr=1e-4, print_summary=False):
    inputs = tf.keras.layers.Input(shape=(nfeatures,))
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs)
    else:
        hidden = inputs
    for _ in range(nl):
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    if print_summary:
        model.summary()
    return model, 'mlp_{0}_{1}'.format(nh, nl)

def identity_block(x, nhidden):
    h = tf.keras.layers.Dense(nhidden)(x)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Add()([x, h])
    h = tf.keras.layers.Activation(activation='relu')(h)
    return h

def dense_block(x, nhidden):
    h = tf.keras.layers.Dense(nhidden)(x)
    h = tf.keras.layers.BatchNormalization()(h)
    s = tf.keras.layers.Dense(nhidden)(x)
    s = tf.keras.layers.BatchNormalization()(s)
    h = tf.keras.layers.Add()([s, h])
    h = tf.keras.layers.Activation(activation='relu')(h)
    return h

def res(nfeatures, nb, nh, ymin, ymax, dropout=0.5, lr=1e-4, print_summary=False):
    inputs = tf.keras.layers.Input(shape=(nfeatures,))
    hidden = tf.keras.layers.Dense(nh)(inputs)
    for _ in range(nb):
        hidden = identity_block(hidden, nh)
        hidden = dense_block(hidden, nh)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    if print_summary:
        model.summary()
    return model, 'res_{0}_{1}'.format(nb, nh)

def attention_block(x, nh):
    q = tf.keras.layers.Dense(nh, use_bias=False)(x)
    k = tf.keras.layers.Dense(nh, use_bias=False)(x)
    v = tf.keras.layers.Dense(nh, use_bias=False)(x)
    a = tf.keras.layers.Multiply()([q, k])
    a = tf.keras.layers.Softmax(axis=-1)(a)
    h = tf.keras.layers.Multiply()([a, v])
    return h

def att(nfeatures, nb, nh, ymin, ymax, dropout=0.5, batchnorm=False, lr=1e-4, print_summary=False):
    inputs = tf.keras.layers.Input(shape=(nfeatures,))
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs)
    else:
        hidden = inputs
    for _ in range(nb):
        hidden = attention_block(hidden, nh)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    if print_summary:
        model.summary()
    return model, 'att_{0}_{1}'.format(nb, nh)

if __name__ == '__main__':

    # task

    task = 'predict_bleach_ratio'

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-m', '--model', help='Model', default='mlp')
    parser.add_argument('-l', '--layers', help='Number of layers', default=2, type=int)
    parser.add_argument('-n', '--neurons', help='Number of neurons', default=512, type=int)
    parser.add_argument('-d', '--delays', help='Delay classes', nargs='+', default=[])
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=False, type=bool)
    args = parser.parse_args()

    # cuda

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # set seed for results reproduction

    set_seeds(seed)

    # meta and standardization values

    meta = load_meta(processed_data_dir, task)
    tags = meta['tags']
    ymin = meta['ymin']
    ymax = meta['ymax']

    # create output directories

    task_models_dir = osp.join(models_dir, task)
    task_results_dir = osp.join(results_dir, task)
    for d in [models_dir, task_models_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # mapper

    mapper = lambda x, y: regression_mapper(x, y, ymin=ymin, ymax=ymax)

    # delay classes

    dcs = sorted(tags.keys())
    dc_list = []
    if args.delays is None or len(args.delays) == 0:
        for p in powerset(dcs):
            dc_list.append(list(p))
    else:
        dc_list.append(args.delays)

    # loop through delay class combinations

    for dc in dc_list:

        print(f'Training using delay classes {dc}')

        # load correct data from metainfo

        id = ','.join([str(item) for item in dc])
        assert id in meta['xmin'].keys()
        assert id in meta['xmax'].keys()
        xmin = np.array(meta['xmin'][id])
        xmax = np.array(meta['xmax'][id])
        nfeatures = len(xmin)
        assert len(xmax) == nfeatures

        # fpath

        fpaths = {}
        for stage in stages:
            fpaths[stage] = osp.join(processed_data_dir, f'{task}_{id}_{stage}{csv}')

        # batches

        batches = {}
        for stage in stages:
            batches[stage] = load_batches(fpaths[stage], batch_size, nfeatures).map(mapper)

        # create model

        model_type = locals()[args.model]
        model, model_name = model_type(nfeatures, args.layers, args.neurons, ymin, ymax, print_summary=args.verbose)

        # create model and results directories

        m_path = osp.join(task_models_dir, f'{model_name}_{id}')
        r_path = osp.join(task_results_dir, f'{model_name}_{id}')
        for p in [m_path, r_path]:
            if not osp.isdir(p):
                os.mkdir(p)

        # train model

        model.fit(
            batches['train'],
            validation_data=batches['validate'],
            epochs=epochs,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_mse',
                verbose=args.verbose,
                patience=patience,
                mode='min',
                restore_best_weights=True
            )]
        )

        # save model

        model.save(m_path)

        # predict and calculate inference statistics

        t_test = 0
        predictions = []
        reals = []
        for x, y in batches['test']:
            preds = model.predict(x)[:, 0]
            predictions = np.hstack([predictions, preds])
            reals = np.hstack([reals, y])
        assert len(predictions) == len(reals)
        error = np.mean(np.abs(reals - predictions))

        # save the results

        print(f'Error: {error}')
        results = [str(error)]
        stats_path = osp.join(r_path, 'stats.csv')
        with open(stats_path, 'w') as f:
            f.write(','.join(results))