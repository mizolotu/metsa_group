import json
import os
import os.path as osp
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from common.ml import model_input, model_output, mlp, split, cnn1, lstm, bilstm, cnn1lstm
from common.utils import set_seeds, load_meta, load_data, substitute_nan_values
from scipy.stats import spearmanr

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-d', '--delay', help='Delay class', default=4, type=int)
    parser.add_argument('-e', '--extractor', help='Feature extractor', default='mlp', choices=['mlp', 'cnn1', 'lstm', 'bilstm', 'cnn1lstm'])
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', help='GPU to use', default='0')
    parser.add_argument('-v', '--verbose', help='Verbose', default=False, type=bool)
    parser.add_argument('-x', '--max', help='Maximum allowed featuer-to-feature correlation', default=0.25, type=float)
    parser.add_argument('-c', '--correlation', help='Correlation type', default='pearson')
    parser.add_argument('-p', '--permutations', help='Number of permutations', default=100, type=int)
    args = parser.parse_args()

    # corr thr

    corr_thr = np.minimum(1.0, args.max)

    # gpu

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # model type

    model_type = f'{args.extractor}'

    # load meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    all_features = meta['features']
    all_classes = meta['classes']
    all_uclasses = np.sort(np.unique(all_classes))
    all_nfeatures = []
    for uc in all_uclasses:
        uc_features = [f for f, c in zip(all_features, all_classes) if c == uc]
        all_nfeatures.append(len(uc_features))

    # load data

    values, labels, timestamps = load_data(osp.join(task_dir, features_fname), all_features)

    # preprocess data

    values_without_nans = substitute_nan_values(values)

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    model_mode_dir = osp.join(task_models_dir, args.mode)
    task_results_dir = osp.join(results_dir, args.task)
    for d in [models_dir, task_models_dir, model_mode_dir, results_dir, task_results_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # results table

    if corr_thr < 1.0:
        corr_tag = args.correlation
    else:
        corr_tag = 'all'
    r_name = permutation_importance_csv.format(corr_tag, args.delay)
    r_path = osp.join(task_results_dir, r_name)
    try:
        p = pd.read_csv(r_path)
    except:
        p = pd.DataFrame({
            'Features': [tag for tag in all_features]
        })

    if model_type not in p.keys():
        p[model_type] = [np.nan for feature in all_features]

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

    all_xmin = np.nanmin(values_k[stages[0]], axis=0)[:np.sum(all_nfeatures)]
    all_xmax = np.nanmax(values_k[stages[0]], axis=0)[:np.sum(all_nfeatures)]

    # set seed

    set_seeds(args.seed)

    # init permutations

    n = len(te)
    perm_idx = []
    idx = np.arange(n)
    for i in range(args.permutations):
        np.random.shuffle(idx)
        perm_idx.append(idx.copy())

    important_features = []

    # loop through features

    for tagi, tag in enumerate(all_features):

        if all_classes[tagi] <= args.delay:

            # ymin and ymax

            ymin = br_min
            ymax = br_max

            # eliminate correlated features

            corr_xx = np.zeros(np.sum(all_nfeatures))
            for i in range(np.sum(all_nfeatures)):
                if all_classes[i] > args.delay:
                    corr_xx[i] = 1.0
                elif i != tagi:
                    if args.correlation == 'pearson':
                        corr_xx[i] = np.abs(np.corrcoef(values_without_nans[:, tagi], values_without_nans[:, i])[0, 1])
                    elif args.correlation == 'spearman':
                        corr_xx[i], _ = np.abs(spearmanr(values_without_nans[:, tagi], values_without_nans[:, i]))
            feature_indexes = np.where(corr_xx < corr_thr)[0].tolist()

            Xtv, Ytv = {}, {}
            for stage in stages[:-1]:
                Xtv[stage], Ytv[stage] = {}, {}
                for fi, f in enumerate(all_features):
                    if fi in feature_indexes:
                        Xtv[stage][f] = values_k[stage][:, fi]
                Ytv[stage][br_key] = labels_k[stage]
            stage = stages[2]
            Xi = {}
            for fi, f in enumerate(all_features):
                if fi in feature_indexes:
                    Xi[f] = values_k[stage][:, fi]
                Yi = labels_k[stage]

            features_selected = [all_features[i] for i in feature_indexes]
            classes_selected = [all_classes[i] for i in feature_indexes]
            uclasses_selected = np.sort(np.unique(classes_selected))
            nfeatures_selected = []
            for uc in uclasses_selected:
                uc_features = [f for f, c in zip(features_selected, classes_selected) if c == uc]
                nfeatures_selected.append(len(uc_features))

            xmin_selected = np.array([all_xmin[i] for i in feature_indexes])
            xmax_selected = np.array([all_xmax[i] for i in feature_indexes])
            print(f'Feature {tagi + 1}/{len(all_features)}: Training using {len(features_selected)} features')

            # create model

            inputs, inputs_processed = model_input(features_selected, xmin_selected, xmax_selected)
            hidden = split(inputs_processed, nfeatures_selected)
            extractor_type = locals()[args.extractor]
            hidden = extractor_type(hidden)
            model = model_output(inputs, hidden, br_key, ymin, ymax)
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
            error = np.mean(np.abs(Yi - predictions))

            # permute

            lp = len(perm_idx)
            feature_importance = np.zeros(lp)

            for i in range(lp):
                # permute

                Xp = Xi.copy()
                Xp[tag] = Xi[tag][perm_idx[i]]

                # predict and calculate inference statistics

                predictions = model.predict(Xp)
                predictions = predictions[br_key].flatten()
                feature_importance[i] = np.mean(np.abs(Yi - predictions)) / error

            if args.verbose:
                print(f'Importance of feature {tagi} ({tag}): {np.mean(feature_importance)}')

            # save permutation results

            perm = np.mean(feature_importance)
            if perm >= 1.0:
                important_features.append(tag)
            idx = np.where(p['Features'].values == tag)[0]
            p[model_type].values[idx] = perm
            p.to_csv(r_path, index=None)

    # save important features

    fname = permutation_important_json.format(corr_tag, model_type, args.delay)
    with open(osp.join(task_results_dir, fname), 'w') as f:
        json.dump(important_features, f)