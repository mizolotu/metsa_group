import json, os
import os.path as osp
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from create_datasets import powerset
from config import *

def load_batches(path, tags, batch_size, nfeatures):
    batches = tf.data.experimental.make_csv_dataset(
        path,
        batch_size=batch_size,
        #header=False,
        shuffle=True,
        #column_names=[str(i) for i in range(nfeatures + 1)],
        #column_defaults=[tf.float32 for _ in range(nfeatures + 1)],
        #select_columns=[str(i) for i in range(nfeatures + 1)],
        select_columns=tags,
        #label_name='{0}'.format(nfeatures),
        label_name=tags[-1],
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

def mlp(nfeatures, nhiddens, xmin, xmax, ymin, ymax, dropout=0.5, batchnorm=False, lr=1e-4):
    inputs = tf.keras.layers.Input(shape=(nfeatures,))
    inputs_std = (inputs - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std
    for nh in nhiddens:
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

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

def res(nfeatures, nb, nh, ymin, ymax, dropout=0.5, lr=1e-4):
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
    return model

def attention_block(x, nh):
    q = tf.keras.layers.Dense(nh, use_bias=False)(x)
    k = tf.keras.layers.Dense(nh, use_bias=False)(x)
    v = tf.keras.layers.Dense(nh, use_bias=False)(x)
    a = tf.keras.layers.Multiply()([q, k])
    a = tf.keras.layers.Softmax(axis=-1)(a)
    h = tf.keras.layers.Multiply()([a, v])
    return h

def att(nfeatures, nb, nh, ymin, ymax, dropout=0.5, batchnorm=False, lr=1e-4):
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
    return model

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--model', help='Model', default='mlp')
    parser.add_argument('-l', '--layers', help='Number of neurons in layers', default=[512, 512], type=int, nargs='+')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=True, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-r', '--reverse', help='All but the tag selected', default=False, type=bool)
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
    for key in tag_keys:
        tags_.extend(tags[key])
    xmin = np.array(meta['xmin'])
    xmax = np.array(meta['xmax'])
    ymin = meta['ymin']
    ymax = meta['ymax']

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    for d in [models_dir, task_models_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # mapper

    mapper = lambda x, y: regression_mapper(x, y, ymin=ymin, ymax=ymax)

    # model

    model_type = locals()[args.model]
    model_name = f"{args.model}_{'-'.join([str(item) for item in args.layers])}"

    # results table
    if not args.reverse:
        r_name = error_by_tag_csv
    else:
        r_name = error_by_not_tag_csv
    r_path = osp.join(task_results_dir, r_name)
    try:
        p = pd.read_csv(r_path)
        if model_name not in p.keys():
            p[model_name] = [np.nan for tag in tags_]
    except:
        p = pd.DataFrame({
            'Tags': [tag for tag in tags_],
            model_name: [np.nan for _ in tags_]
        })

    # loop through delay class combinations


    for tag in tags_:

        # features

        tag_idx = tags_.index(tag)

        if not args.reverse:
            tags_selected = [tag]
            xmin = xmin[tag_idx : tag_idx + 1]
            xmax = xmax[tag_idx : tag_idx + 1]
            print(f'Training using tag {tag}')
            m_name = f'{model_name}_{tag}'
        else:
            tags_selected = tags_.copy()
            tags_selected.remove(tag)
            xmin = np.hstack([xmin[: tag_idx], xmin[tag_idx + 1 :]])
            xmax = np.hstack([xmax[: tag_idx], xmax[tag_idx + 1 :]])
            print(f'Training using all but tag {tag}')
            m_name = f'{model_name}_not-{tag}'

        nfeatures = len(xmin)
        assert len(xmax) == nfeatures

        # fpath

        fpaths = {}
        for stage in stages:
            fpaths[stage] = osp.join(processed_data_dir, f'{args.task}_{stage}{csv}')

        # batches

        tags_and_label = tags_selected + [br_key]
        batches = {}
        for stage in stages:
            batches[stage] = load_batches(fpaths[stage], tags_and_label, batch_size, nfeatures).map(mapper)

        # create model

        model = model_type(nfeatures, args.layers, xmin, xmax, ymin, ymax)
        if args.verbose:
            model.summary()

        # create model and results directories

        m_path = osp.join(task_models_dir, m_name)
        if not osp.isdir(m_path):
            os.mkdir(m_path)

        # load model

        try:
            model = tf.keras.models.load_model(m_path)

        except Exception as e:
            print(e)

            # train model

            model.fit(
                batches['train'],
                validation_data=batches['validate'],
                epochs=epochs,
                verbose=args.verbose,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    verbose=0,
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
        for x, y in batches['inference']:
            preds = model.predict(x)[:, 0]
            predictions = np.hstack([predictions, preds])
            reals = np.hstack([reals, y])
        assert len(predictions) == len(reals)
        error = np.mean(np.abs(reals - predictions))

        # save the results

        print(f'Error: {error}')
        idx = np.where(p['Tags'].values == tag)[0]
        p[model_name].values[idx] = error
        p.to_csv(r_path, index=None)