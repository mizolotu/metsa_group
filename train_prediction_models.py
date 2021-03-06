import os
import os.path as osp

import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from common.utils import set_seeds, load_meta, load_data, pad_data
from common.ml import model_input, split, model_output, mlp, cnn1, lstm, bilstm, cnn1lstm, EarlyStoppingAtMaxAUC, roc_auc

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-e', '--extractor', help='Feature extractor', default='mlp', choices=['mlp', 'cnn1', 'cnn1lstm', 'lstm', 'bilstm'])
    parser.add_argument('-c', '--classes', help='Delay class when prediction starts', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('-s', '--seed', help='Starting seed', type=int, default=seed)
    parser.add_argument('-g', '--gpu', help='GPU to use', default='0')
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-y', '--ylimits', help='Use bleach ratio limits from data?', default=False, type=bool)
    parser.add_argument('-r', '--retrain', help='Retrain model?', default=False, type=bool)
    parser.add_argument('-n', '--ntests', help='Number of tests', type=int, default=5)
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    parser.add_argument('-u', '--update', help='Update results?', default=True, type=bool)
    parser.add_argument('-f', '--features', help='Comma separated tuples: <csv file with feature importance values>,<column index>,<number of features>', nargs='+')
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

    # feature indexes

    if args.features is not None:
        assert len(args.features) == len(args.classes)
        feature_list, model_prefixes = [], []
        for features in args.features:
            try:
                fname, col, nfs = features[0], int(features[1]), int(features[2])
                df = pd.read_csv(osp.join(task_results_dir, fname))
                keys = list(df.keys())
                col_header = keys[col]
                f_importance = df[col_header].values
                idx = np.argsort(f_importance)[::-1][:nfs]
                feature_list.append(df['Features'][idx].tolist())
                model_prefixes.append(f"{'_'.join(fname.split('.json')[0].split('_')[:2])}_{col_header}_{nfs}")
            except:
                feature_list.append(None)
                model_prefixes.append('')
    else:
        feature_list = [None for _ in args.classes]
        model_prefixes = ['' for _ in args.classes]

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

        model_type = f'{model_prefix}{feature_extractor if feature_extractor is not None else default_feature_extractor_name}'
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

            # set seed for results reproduction

            set_seeds(args.seed + k)

            # data split

            inds = np.arange(len(labels))
            inds_splitted = [[] for _ in stages]
            np.random.shuffle(inds)
            val, remaining = np.split(inds, [int(validation_share * len(inds))])
            tr, te = np.split(remaining, [int(train_test_ratio * len(remaining))])
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

                if feature_extractor is None:
                    inputs, hidden = model_input(features_selected, xmin, xmax, steps)
                    extractor_type = locals()[default_feature_extractor]
                    hidden = extractor_type(hidden)
                    model = model_output(inputs, hidden, br_key, ymin, ymax)
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

        # results tables

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






