import os
import os.path as osp
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from common.ml import model_input, model_output, mlp, split, cnn1
from common.utils import set_seeds, load_meta, load_data

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', help='GPU to use', default='0')
    parser.add_argument('-v', '--verbose', help='Verbose', default=False, type=bool)
    parser.add_argument('-e', '--evalmethod', help='Evaluation method', choices=['selected', 'not-selected', 'permuted'], default='not-selected')
    parser.add_argument('-d', '--delay', help='Delay class', default=4, type=int)
    args = parser.parse_args()

    # gpu

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']
    uclasses = np.sort(np.unique(classes))
    nfeatures = []
    for uc in uclasses:
        uc_features = [f for f, c in zip(features, classes) if c == uc]
        nfeatures.append(len(uc_features))

    # load data

    values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features)

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    model_mode_dir = osp.join(task_models_dir, args.mode)
    task_results_dir = osp.join(results_dir, args.task)
    for d in [models_dir, task_models_dir, model_mode_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # results table

    r_name = prediction_importance_csv.format(args.delay)
    r_path = osp.join(task_results_dir, r_name)
    try:
        p = pd.read_csv(r_path)
        if args.evalmethod not in p.keys():
            p[args.evalmethod] = [np.nan for tag in features]
    except:
        p = pd.DataFrame({
            'Features': [tag for tag in features],
            args.evalmethod: [np.nan for _ in features]
        })

    # data split

    inds = np.arange(len(labels))
    inds_splitted = [[] for _ in stages]
    np.random.shuffle(inds)
    val, remaining = np.split(inds, [int(validation_share * len(inds))])
    tr, te = np.split(remaining, [int(train_test_ratio * len(remaining))])
    labels_ = labels.copy()
    inds_splitted[0] = tr
    inds_splitted[1] = val
    inds_splitted[2] = te
    timestamps_k, values_k, labels_k = {}, {}, {}
    for fi, stage in enumerate(stages):
        timestamps_k[stage], values_k[stage], labels_k[stage] = timestamps[inds_splitted[fi]], values[inds_splitted[fi], :], labels_[inds_splitted[fi]]
    ntrain = len(tr)
    nval = len(val)

    # standardization coefficients

    xmin = np.nanmin(values_k[stages[0]], axis=0)[:np.sum(nfeatures)]
    xmax = np.nanmax(values_k[stages[0]], axis=0)[:np.sum(nfeatures)]

    # ymin and ymax

    ymin = br_min
    ymax = br_max

    # data

    Xtv, Ytv = {}, {}
    for stage in stages[:-1]:
        Xtv[stage], Ytv[stage] = {}, {}
        for fi, f in enumerate(features):
            if classes[fi] <= args.delay:
                Xtv[stage][f] = values_k[stage][:, fi]
        Ytv[stage][br_key] = labels_k[stage]
    stage = stages[2]
    Xi = {}
    tags_selected, xmin_selected, xmax_selected = [], [], []
    for fi, f in enumerate(features):
        if classes[fi] <= args.delay:
            Xi[f] = values_k[stage][:, fi]
            tags_selected.append(f)
            xmin_selected.append(xmin[fi])
            xmax_selected.append(xmax[fi])
    Yi = labels_k[stage]
    nfeatures = []
    for uc in uclasses:
        if uc <= args.delay:
            nfeatures.append(len(np.where(classes == uc)[0]))
    xmin_selected = np.array(xmin_selected)
    xmax_selected = np.array(xmax_selected)

    # create baseline model

    inputs, inputs_processed = model_input(tags_selected, xmin_selected, xmax_selected)
    if args.evalmethod == 'selected':
        hidden = inputs_processed
        model_type = 'mlp'
    else:
        hidden = split(inputs_processed, nfeatures)
        model_type = 'cnn1'
    extractor_type = locals()[model_type]
    hidden = extractor_type(hidden)
    model = model_output(inputs, hidden, br_key, ymin, ymax)

    # train model

    model.fit(
        Xtv[stages[0]], Ytv[stages[0]],
        validation_data=(Xtv[stages[1]], Ytv[stages[1]]),
        epochs=epochs,
        verbose=args.verbose,
        batch_size=batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=0,
            patience=patience,
            mode='min',
            restore_best_weights=True
        )]
    )

    # predict and calculate inference statistics

    t_test = 0
    predictions = model.predict(Xi)
    predictions = predictions[br_key].flatten()
    error_original = np.mean(np.abs(Yi - predictions))
    print(error_original)

    # loop through features

    for tagi, tag in enumerate(features):

        if classes[tagi] <= args.delay:

            # set seed

            set_seeds(args.seed)

            # features

            if args.evalmethod == 'selected':
                Xtv, Ytv = {}, {}
                for stage in stages[:-1]:
                    Xtv[stage], Ytv[stage] = {}, {}
                    Xtv[stage][tag] = values_k[stage][:, tagi]
                    Ytv[stage][br_key] = labels_k[stage]
                stage = stages[2]
                Xi = {}
                Xi[tag] = values_k[stage][:, tagi]
                Yi = labels_k[stage]
                xmin_selected = xmin[tagi: tagi + 1]
                xmax_selected = xmax[tagi: tagi + 1]
                print(f'{tagi + 1}/{len(features)} Training using tag {tag}')
                nfeatures = 1
                tags_selected = [tag]
            elif args.evalmethod == 'not-selected':
                Xtv, Ytv = {}, {}
                for stage in stages[:-1]:
                    Xtv[stage], Ytv[stage] = {}, {}
                    for fi, f in enumerate(features):
                        if classes[fi] <= args.delay and f != tag:
                            Xtv[stage][f] = values_k[stage][:, fi]
                    Ytv[stage][br_key] = labels_k[stage]
                stage = stages[2]
                Xi = {}
                tags_selected, xmin_selected, xmax_selected = [], [], []
                for fi, f in enumerate(features):
                    if classes[fi] <= args.delay and f != tag:
                        Xi[f] = values_k[stage][:, fi]
                        tags_selected.append(f)
                        xmin_selected.append(xmin[fi])
                        xmax_selected.append(xmax[fi])
                Yi = labels_k[stage]
                nfeatures = []
                for uc in uclasses:
                    if classes[tagi] == uc:
                        nfeatures.append(len(np.where(classes == uc)[0]) - 1)
                    elif uc <= args.delay:
                        nfeatures.append(len(np.where(classes == uc)[0]))
                xmin_selected = np.array(xmin_selected)
                xmax_selected = np.array(xmax_selected)
                print(f'{tagi + 1}/{len(features)} Training using all but tag {tag}')
            elif args.evalmethod == 'permuted':
                Xtv, Ytv = {}, {}
                for stage in stages[:-1]:
                    Xtv[stage], Ytv[stage] = {}, {}
                    for fi, f in enumerate(features):
                        Xtv[stage][f] = values_k[stage][:, fi]
                    Ytv[stage][br_key] = labels_k[stage]
                stage = stages[2]
                Xi = {}
                for fi, f in enumerate(features):
                    Xi[f] = values_k[stage][:, fi]
                Yi = labels_k[stage]
                nfeatures = []
                for uc in uclasses:
                    nfeatures.append(len(np.where(classes == uc)[0]))
                tags_selected = features.copy()
                xmin_selected = xmin
                xmax_selected = xmax
                print(f'{tagi + 1}/{len(features)} Training using permuted tag {tag}')

                shuffle_idx_tr = np.arange(ntrain)
                shuffle_idx_val = np.arange(nval)
                np.random.shuffle(shuffle_idx_tr)
                np.random.shuffle(shuffle_idx_val)
                Xtv[stages[0]][tag] = Xtv[stages[0]][tag][shuffle_idx_tr]
                Xtv[stages[1]][tag] = Xtv[stages[1]][tag][shuffle_idx_val]

            # create model

            inputs, inputs_processed = model_input(tags_selected, xmin_selected, xmax_selected)
            if args.evalmethod == 'selected':
                hidden = inputs_processed
                model_type = 'mlp'
            else:
                hidden = split(inputs_processed, nfeatures)
                model_type = 'cnn1'
            extractor_type = locals()[model_type]
            hidden = extractor_type(hidden)
            model = model_output(inputs, hidden, br_key, ymin, ymax)

            # create model and results directories

            m_name = f'{model_type}_{args.evalmethod}_{tag}'
            m_path = osp.join(model_mode_dir, m_name)
            if not osp.isdir(m_path):
                os.mkdir(m_path)
            if args.verbose:
                model.summary()

            # train model

            model.fit(
                Xtv[stages[0]], Ytv[stages[0]],
                validation_data=(Xtv[stages[1]], Ytv[stages[1]]),
                epochs=epochs,
                verbose=args.verbose,
                batch_size=batch_size,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    verbose=0,
                    patience=patience,
                    mode='min',
                    restore_best_weights=True
                )]
            )

            # predict and calculate inference statistics

            t_test = 0
            predictions = model.predict(Xi)
            predictions = predictions[br_key].flatten()
            error = np.mean(np.abs(Yi - predictions)) / error_original

            # save the results

            print(f'Prediction importance: {error}')
            idx = np.where(p['Features'].values == tag)[0]
            p[args.evalmethod].values[idx] = error
            p.to_csv(r_path, index=None)