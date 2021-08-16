import json, os
import os.path as osp
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from calculate_error_by_tag import set_seeds, load_meta, load_batches, regression_mapper, mlp

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--model', help='Model', default='mlp')
    parser.add_argument('-l', '--layers', help='Number of neurons in layers', default=[512, 512], type=int, nargs='+')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=False, type=bool)
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

    r_name = error_by_tag_permutation_csv
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
    nfeatures = len(tags_)

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
            epochs=10,
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

    # create permutations

    n = X.shape[0]
    perm_idx = []
    idx = np.arange(n)
    for i in range(args.npermutations):
        np.random.shuffle(idx)
        perm_idx.append(idx.copy())

    # loop through tags

    for tag in tags_:

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

        print(f'Tag {tag} importance: {feature_importance}')
        idx = np.where(p['Tags'].values == tag)[0]
        p[model_name].values[idx] = feature_importance
        p.to_csv(r_path, index=None)