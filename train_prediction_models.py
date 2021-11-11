import json
import os
import os.path as osp

import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from common.utils import set_seeds, load_meta, load_data, pad_data
from common.ml import model_input, split, model_output, mlp, cnn1, lstm, bilstm, cnn1lstm, aen, som, EarlyStoppingAtMaxAUC, roc_auc

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-e', '--extractor', help='Feature extractor', default='cnn1', choices=['mlp', 'cnn1', 'lstm', 'bilstm', 'cnn1lstm', 'aen', 'som'])
    parser.add_argument('-c', '--classes', help='Delay class when prediction starts', type=int, nargs='+', default=[4, 5])
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=seed)
    parser.add_argument('-g', '--gpu', help='GPU to use', default='0')
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-y', '--ylimits', help='Use bleach ratio limits from data?', default=False, type=bool)
    parser.add_argument('-r', '--retrain', help='Retrain model?', default=False, type=bool)
    parser.add_argument('-n', '--ntests', help='Number of tests', type=int, default=3)
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-u', '--update', help='Update results?', default=False, type=bool)
    parser.add_argument('-f', '--features', help='List of the features selected in json format') #, default=['permutation_important_all_cnn1_4.json', 'permutation_important_all_cnn1_5.json'])
    args = parser.parse_args()

    # create output directories

    task_models_dir = osp.join(models_dir, args.task)
    model_mode_dir = osp.join(task_models_dir, args.mode)
    task_results_dir = osp.join(results_dir, args.task)
    results_mode_dir = osp.join(task_results_dir, args.mode)
    task_figures_dir = osp.join(figures_dir, args.task)
    for d in [models_dir, task_models_dir, model_mode_dir, results_dir, task_results_dir, results_mode_dir, task_figures_dir]:
        if not osp.isdir(d):
            os.mkdir(d)

    # model input layer

    feature_extractor = args.extractor
    if feature_extractor in ae_models:
        ae = True
    else:
        ae = False

    # feature indexes

    model_prefixes = []
    if args.features is not None:
        assert len(args.features) == len(args.classes), 'There should be file with feature indexes for every delay class tested'
        feature_list = []
        for fname in args.features:
            try:
                with open(osp.join(task_results_dir, fname)) as f:
                    feature_list.append(json.load(f))
                    model_prefixes.append(f"{fname.split('.json')[0]}_")
            except:
                feature_list.append(None)
                model_prefixes.append('')
    else:
        feature_list = [None for _ in args.classes]
        model_prefixes.append('')

    # number of tests

    if args.mode == 'production':
        ntests = 1
    else:
        ntests = args.ntests

    # gpu

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # walk through classes

    for delay_class, features, model_prefix in zip(args.classes, feature_list, model_prefixes):

        # set seed for results reproduction

        set_seeds(args.seed)

        # load meta

        task_dir = osp.join(data_dir, args.task)
        meta = load_meta(osp.join(task_dir, meta_fname))
        all_features = meta['features']
        all_classes = meta['classes']
        all_uclasses = np.sort(np.unique(all_classes))
        if features is not None:
            classes = [c for f, c in zip(all_features, all_classes) if f in features]
        else:
            features, classes = all_features.copy(), all_classes.copy()
        uclasses = np.sort(np.unique(classes))
        features_selected, feature_classes_selected = [list(item) for item in zip(*[(f, c) for f, c in zip(features, classes) if c <= delay_class])]
        u_feature_classes_selected = np.unique(feature_classes_selected)
        nfeatures = []
        dcs = []
        tbl_dc_combs = []
        for uc in all_uclasses:
            dcs.extend(str(uc))
            dc_comb = ','.join([item for item in dcs])
            tbl_dc_combs.append(dc_comb)
            uc_features = [f for f, c in zip(features, classes) if c == uc]
            if uc in u_feature_classes_selected:
                nfeatures.append(len(uc_features))
                model_dc_comb = dc_comb

        print(f'The following feature classes will be used to train the model: {model_dc_comb}')

        # load data

        values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features_selected)
        if values.shape[1] == len(features_selected):
            pass
        elif values.shape[1] == len(features_selected) * series_len:
            values = values.reshape(values.shape[0], series_len, len(features_selected))

        # model name

        model_type = f'{model_prefix}{feature_extractor}'
        model_name = f'{model_type}_{delay_class}'

        # create model directory

        m_path = osp.join(task_models_dir, args.mode, model_name)
        if not osp.isdir(m_path):
            os.mkdir(m_path)

        # load model

        if ntests == 1 and not args.retrain:
            try:
                model = tf.keras.models.load_model(m_path)
                have_to_create_model = False
                print(f'Model {model_name} has been loaded from {m_path}')
            except Exception as e:
                print(e)
                have_to_create_model = True
        else:
            have_to_create_model = True

        # prediction results

        reals, preds, errors, tsteps = [], [], [], []
        mean_errors = np.zeros(ntests)
        min_errors = np.zeros(ntests)
        max_errors = np.zeros(ntests)
        aucs = np.zeros(ntests)
        feature_importances = np.zeros((len(all_features), ntests))

        for k in range(ntests):
            print(f'Test {k + 1}/{ntests}:')

            # data split

            inds = np.arange(len(labels))
            inds_splitted = [[] for _ in stages]
            np.random.shuffle(inds)
            val, remaining = np.split(inds, [int(validation_share * len(inds))])
            tr, te = np.split(remaining, [int(train_test_ratio * len(remaining))])
            if ae:
                labels_ = np.zeros_like(labels)
                labels_[np.where((labels < br_min) | (labels > br_max))[0]] = 1
                outlier_ids = tr[np.where(labels[tr] == 1)[0]]
                val = np.append(val, outlier_ids)
                te = np.append(te, outlier_ids)
                tr = tr[np.where(labels_[tr] == 0)[0]]
            else:
                labels_ = labels.copy()
            if args.mode == 'production':
                tr = np.hstack([tr, val])
                val = te.copy()
                te = np.array([], dtype=int)
            inds_splitted[0] = tr
            inds_splitted[1] = val
            inds_splitted[2] = te
            timestamps_k, values_k, labels_k = {}, {}, {}
            for fi, stage in enumerate(stages):
                timestamps_k[stage], values_k[stage], labels_k[stage] = timestamps[inds_splitted[fi]], values[inds_splitted[fi], :], labels_[inds_splitted[fi]]

            # standardization coefficients

            if len(values_k[stages[0]].shape) == 2:
                xmin = np.nanmin(values_k[stages[0]], axis=0)[:np.sum(nfeatures)]
                xmax = np.nanmax(values_k[stages[0]], axis=0)[:np.sum(nfeatures)]
            elif len(values_k[stages[0]].shape) == 3:
                xmin = np.nanmin(values_k[stages[0]], axis=0)[:, :np.sum(nfeatures)]
                xmax = np.nanmax(values_k[stages[0]], axis=0)[:, :np.sum(nfeatures)]

            if args.ylimits:
                ymin = np.nanmin(labels_k[stages[0]])
                ymax = np.nanmax(labels_k[stages[0]])
            else:
                ymin = br_min
                ymax = br_max

            # create datasets by padding certain feature classes

            Xtv, Ytv = {}, {}
            for stage in stages[:-1]:
                Xtv[stage], Ytv[stage] = {}, {}
                Xtmp = values_k[stage]
                if len(Xtmp.shape) == 2:
                    for i, fs in enumerate(features_selected):
                        Xtv[stage][fs] = Xtmp[:, i]
                elif len(Xtmp.shape) == 3:
                    for i, fs in enumerate(features_selected):
                        Xtv[stage][fs] = Xtmp[:, :, i]
                Ytv[stage][br_key] = labels_k[stage]

            if args.mode == 'production':
                stage = stages[1]
            else:
                stage = stages[2]

            Xi = {}
            Xtmp = pad_data(values_k[stage], features_selected, features, classes, model_dc_comb)
            if len(Xtmp.shape) == 2:
                steps = 1
                for i, fs in enumerate(features_selected):
                    Xi[fs] = Xtmp[:, i]
            elif len(Xtmp.shape) == 3:
                steps = series_len
                for i, fs in enumerate(features_selected):
                    Xi[fs] = Xtmp[:, :, i]
            Yi = labels_k[stage]
            Ti = timestamps_k[stage]

            # create and train a new model if needed

            if have_to_create_model:

                print(f'Training new model {model_name}:')

                if ae:
                    extractor_type = locals()[feature_extractor]
                    model = extractor_type(features_selected, xmin, xmax, nfeatures, br_key)
                else:
                    inputs, inputs_processed = model_input(features_selected, xmin, xmax, steps)
                    hidden = split(inputs_processed, nfeatures)
                    extractor_type = locals()[feature_extractor]
                    hidden = extractor_type(hidden)
                    model = model_output(inputs, hidden, br_key, ymin, ymax)
                model_summary_lines = []
                model.summary(print_fn=lambda x: model_summary_lines.append(x))
                model_summary = "\n".join(model_summary_lines)
                if args.verbose and k == 0:
                    print(model_summary)

                if ae:
                    x_val = np.hstack([np.expand_dims(Xtv[stages[1]][f], 1) for f in features_selected])
                    y_val = Ytv[stages[1]][br_key]
                    es_callback = EarlyStoppingAtMaxAUC(validation_data=(Xtv[stages[1]], Ytv[stages[1]]), patience=patience)
                else:
                    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)

                model.fit(
                    Xtv[stages[0]], Ytv[stages[0]],
                    validation_data=(Xtv[stages[1]], Ytv[stages[1]]),
                    epochs=epochs,
                    verbose=args.verbose,
                    batch_size=batch_size,
                    callbacks=[es_callback]
                )

                # save model

                if args.update:
                    model.save(m_path)
                    with open(osp.join(m_path, summary_txt), 'w') as f:
                        f.write(model_summary)

            # calculate prediction error for non-permuted features of the class combination

            predictions = model.predict(Xi)

            if ae:

                aucs[k] = roc_auc(Yi, predictions)
                reals.extend(Yi)
                tsteps.extend(Ti)
                preds.extend(predictions)

                print(f'Anomaly detection ROC AUC (FPR = 10%): {roc_auc(Yi, predictions, fpr=0.1)}')
                print(f'Anomaly detection ROC AUC (FPR = 1%): {roc_auc(Yi, predictions, fpr=0.01)}')
                print(f'Anomaly detection ROC AUC (FPR = 0.1%): {roc_auc(Yi, predictions, fpr=0.001)}')

            else:

                predictions = predictions[br_key].flatten()
                min_errors[k] = np.min(np.abs(Yi - predictions))
                mean_errors[k] = np.mean(np.abs(Yi - predictions))
                max_errors[k] = np.max(np.abs(Yi - predictions))
                max_i = np.argmax(np.abs(Yi - predictions))
                print(f'Max error prediction: {predictions[max_i]}, the real value: {Yi[max_i]}')

                errors.extend(np.abs(Yi - predictions))
                reals.extend(Yi)
                tsteps.extend(Ti)
                preds.extend(predictions)

                print(f'Mean absolute prediction error for combination {model_dc_comb}: {mean_errors[k]}')
                print(f'Min absolute prediction error for combination {model_dc_comb}: {min_errors[k]}')
                print(f'Max absolute prediction error for combination {model_dc_comb}: {max_errors[k]}')

        # results tables

        if ae:

            mean_a_path = osp.join(results_mode_dir, anomaly_detection_mean_aucs_fname)
            r_path = osp.join(results_mode_dir, anomaly_detection_results_fname)

            try:
                auc_mean = pd.read_csv(mean_a_path)
            except:
                auc_mean = pd.DataFrame({
                    dc_combs_col_name: [comb for comb in tbl_dc_combs]
                })

            if model_type not in auc_mean.keys():
                auc_mean[model_type] = [np.nan for comb in tbl_dc_combs]

            try:
                pr = pd.read_csv(r_path)
                assert pr.shape[0] == len(tsteps), 'The dataset size has changed, the statistics table will be rewritten!'
            except:
                pr = pd.DataFrame({
                    ts_key: [value for value in tsteps],
                    br_key: [value for value in reals],
                })

        else:

            mean_e_path = osp.join(results_mode_dir, prediction_mean_errors_fname)
            min_e_path = osp.join(results_mode_dir, prediction_min_errors_fname)
            max_e_path = osp.join(results_mode_dir, prediction_max_errors_fname)
            r_path = osp.join(results_mode_dir, prediction_results_fname)

            try:
                p_e_mean = pd.read_csv(mean_e_path)
            except Exception:
                p_e_mean = pd.DataFrame({
                    dc_combs_col_name: [comb for comb in tbl_dc_combs]
                })

            if model_type not in p_e_mean.keys():
                p_e_mean[model_type] = [np.nan for comb in tbl_dc_combs]

            try:
                p_e_min = pd.read_csv(min_e_path)
            except:
                p_e_min = pd.DataFrame({
                    dc_combs_col_name: [comb for comb in tbl_dc_combs]
                })

            if model_type not in p_e_min.keys():
                p_e_min[model_type] = [np.nan for comb in tbl_dc_combs]

            try:
                p_e_max = pd.read_csv(max_e_path)
            except:
                p_e_max = pd.DataFrame({
                    dc_combs_col_name: [comb for comb in tbl_dc_combs]
                })

            if model_type not in p_e_max.keys():
                p_e_max[model_type] = [np.nan for comb in tbl_dc_combs]

            try:
                pr = pd.read_csv(r_path)
                assert pr.shape[0] == len(tsteps), 'The dataset size has changed, the statistics table will be rewritten!'

            except:
                pr = pd.DataFrame({
                    ts_key: [value for value in tsteps],
                    br_key: [value for value in reals],
                })

        # update prediction results

        if args.update:

            if ae:

                assert model_dc_comb in tbl_dc_combs
                idx = tbl_dc_combs.index(model_dc_comb)
                auc_mean[model_type].values[idx] = np.mean(aucs)
                auc_mean.to_csv(mean_a_path, index=None)

            else:
                assert model_dc_comb in tbl_dc_combs
                idx = tbl_dc_combs.index(model_dc_comb)
                p_e_mean[model_type].values[idx] = np.mean(mean_errors)
                p_e_mean.to_csv(mean_e_path, index=None)
                p_e_min[model_type].values[idx] = np.mean(min_errors)
                p_e_min.to_csv(min_e_path, index=None)
                p_e_max[model_type].values[idx] = np.mean(max_errors)
                p_e_max.to_csv(max_e_path, index=None)

            pr[model_name] = preds
            pr.to_csv(r_path, index=None)






