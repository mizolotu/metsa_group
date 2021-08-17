import os
import os.path as osp
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from calculate_prediction_error import set_seeds, load_meta, load_batches, regression_mapper

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
    parser.add_argument('-m', '--model', help='Model', default='att')
    parser.add_argument('-l', '--layers', help='Number of neurons in layers', default=[512, 512], type=int, nargs='+')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-p', '--npermutations', help='Number of permutations', type=int, default=10)
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
    ymin = meta['ymin']
    ymax = meta['ymax']

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    task_results_dir = osp.join(results_dir, args.task)
    for d in [models_dir, task_models_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # mapper

    mapper = lambda x, y: regression_mapper(x, y)

    # model

    model_type = locals()[args.model]
    model_name = f"{args.model}_{'-'.join([str(item) for item in args.layers])}"

    # results table

    r_name = permutation_error_csv
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

    # fpath

    fpaths = {}
    for stage in stages:
        fpaths[stage] = osp.join(processed_data_dir, f'{args.task}_{stage}{csv}')

    # batches

    tags_and_label = tags_ + [br_key]
    batches = {}
    for stage in stages:
        batches[stage] = load_batches(fpaths[stage], tags_and_label, batch_size).map(mapper)

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
            batches['train'],
            validation_data=batches['validate'],
            epochs=epochs,
            verbose=args.verbose,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=True,
                patience=patience,
                mode='min',
                restore_best_weights=True
            )]
        )

        # save model

        model.save(m_path)

    # load data and calculate prediction error

    X, Y = [], []
    t_test = 0
    predictions = []
    reals = []
    for x, y in batches['inference']:
        X.append(x)
        Y.append(y)
        preds = model.predict(x)[:, 0]
        predictions = np.hstack([predictions, preds])
        reals = np.hstack([reals, y])
    assert len(predictions) == len(reals)
    error_full = np.mean(np.abs(reals - predictions))
    X = np.vstack(X)
    Y = np.hstack(Y)
    print(f'Prediction error when using all tags: {error_full}')

    # create permutations

    n = X.shape[0]
    perm_idx = []
    idx = np.arange(n)
    for i in range(args.npermutations):
        np.random.shuffle(idx)
        perm_idx.append(idx.copy())

    # loop through tags

    for tag_i, tag in enumerate(tags_):

        # features

        tag_idx = tags_.index(tag)
        lp = len(perm_idx)
        error = np.zeros(lp)

        for i in range(lp):

            # permute

            Xp = X.copy()
            Xp[:, tag_idx] = X[perm_idx[i], tag_idx]

            # predict and calculate inference statistics

            predictions = model.predict(Xp)[:, 0]
            error[i] = np.mean(np.abs(Y - predictions))

        feature_importance = np.mean(error) - error_full

        # save the results

        print(f'{tag_i+1}/{len(tags_)} Tag {tag} importance: {feature_importance}')
        idx = np.where(p['Tags'].values == tag)[0]
        p[model_name].values[idx] = feature_importance
        p.to_csv(r_path, index=None)