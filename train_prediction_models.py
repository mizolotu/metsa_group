import os
import scipy.stats as ss
import os.path as osp

import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *
from common.utils import set_seeds, load_meta, load_data, pad_data, get_best_distribution
from common.ml import model_input, model_output, baseline, split, mlp, cnn1, lstm, bilstm, cnn1lstm

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-i', '--input', help='Model input latent size', default='split', choices=['baseline', 'split'])
    parser.add_argument('-e', '--extractor', help='Feature extractor', default='mlp', choices=['mlp', 'cnn1', 'lstm', 'bilstm', 'cnn1lstm'])
    parser.add_argument('-c', '--classes', help='Delay class when prediction starts', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=seed)
    parser.add_argument('-g', '--gpu', help='GPU to use')
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-y', '--ylimits', help='Use bleach ratio limits from data?', default=False, type=bool)
    parser.add_argument('-r', '--retrain', help='Retrain model?', default=False, type=bool)
    parser.add_argument('-n', '--ntests', help='Number of tests', type=int, default=1)
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=modes)
    args = parser.parse_args()

    # model input layer

    if args.input == 'baseline':
        feature_extractor = 'mlp'
        print('Baseline model will use mlp feature extractor')
    else:
        feature_extractor = args.extractor

    # number of tests

    if args.mode == 'production':
        ntests = 1
    else:
        ntests = args.ntests

    # cuda

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seed for results reproduction

    set_seeds(args.seed)

    # walk through first and last classes

    for delay_class in args.classes:

        # load meta

        task_dir = osp.join(data_dir, args.task)
        meta = load_meta(osp.join(task_dir, meta_fname))
        features = meta['features']
        classes = meta['classes']
        uclasses = np.sort(np.unique(classes))
        features_selected, feature_classes_selected = [list(item) for item in zip(*[(f, c) for f, c in zip(features, classes) if c <= delay_class])]
        u_feature_classes_selected = np.unique(feature_classes_selected)
        nfeatures = []
        dcs = []
        tbl_dc_combs = []
        for uc in uclasses:
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

        # model name

        model_type = f'{args.input}_{feature_extractor}'
        model_name = f'{model_type}_{delay_class}'

        # create output directories

        task_models_dir = osp.join(models_dir, args.task)
        model_mode_dir = osp.join(task_models_dir, args.mode)
        m_path = osp.join(task_models_dir, args.mode, model_name)
        task_results_dir = osp.join(results_dir, args.task)
        results_mode_dir = osp.join(task_results_dir, args.mode)
        for d in [models_dir, task_models_dir, model_mode_dir, m_path, results_dir, task_results_dir, results_mode_dir]:
            if not osp.isdir(d):
                os.mkdir(d)

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

        reals, errors = [], []
        mean_errors = np.zeros(ntests)
        min_errors = np.zeros(ntests)
        max_errors = np.zeros(ntests)

        for k in range(ntests):
            print(f'Test {k + 1}/{ntests}:')

            # data split

            inds = np.arange(len(labels))
            inds_splitted = [[] for _ in stages]
            np.random.shuffle(inds)
            val, remaining = np.split(inds, [int(validation_share * len(inds))])
            tr, te = np.split(remaining, [int(train_test_ratio * len(remaining))])
            if args.mode == 'production':
                tr = np.hstack([tr, val])
                val = te.copy()
                te = np.array([], dtype=int)
            inds_splitted[0] = tr
            inds_splitted[1] = val
            inds_splitted[2] = te
            timestamps_k, values_k, labels_k = {}, {}, {}
            for fi, stage in enumerate(stages):
                timestamps_k[stage], values_k[stage], labels_k[stage] = timestamps[inds_splitted[fi]], values[inds_splitted[fi], :], labels[inds_splitted[fi]]

            # standardization coefficients

            xmin = np.nanmin(values_k[stages[0]], axis=0)[:np.sum(nfeatures)]
            xmax = np.nanmax(values_k[stages[0]], axis=0)[:np.sum(nfeatures)]
            if args.ylimits:
                ymin = np.nanmin(labels_k[stages[0]])
                ymax = np.nanmax(labels_k[stages[0]])
            else:
                ymin = br_min
                ymax = br_max

            # br ratio distribution

            ymean = np.mean(labels_k[stages[0]])
            ystd = np.std(labels_k[stages[0]])
            y_prob_thr = ss.norm.pdf(ymean + 5 * ystd, ymean, ystd)

            # create datasets by padding certain feature classes

            Xtv, Wtv, Ytv = {}, {}, {}
            for stage in stages[:-1]:
                Xtv[stage], Wtv[stage], Ytv[stage] = {}, {}, {}
                #Xtmp = pad_data(values_k[stage], features_selected, features, classes, model_dc_comb)
                Xtmp = values_k[stage]
                #Ytmp = labels_k[stage]
                #Xtmp = np.vstack(Xtmp)
                #Ytmp = np.hstack(Ytmp)
                for i, fs in enumerate(features_selected):
                    Xtv[stage][fs] = Xtmp[:, i]
                Ytv[stage][br_key] = labels_k[stage]
                Wtv[stage] = 1 / np.clip(ss.norm.pdf(labels_k[stage], ymean, ystd), y_prob_thr, np.inf)

            if args.mode == 'production':
                stage = stages[1]
            else:
                stage = stages[2]
            Xi = {}
            Xtmp = pad_data(values_k[stage], features_selected, features, classes, model_dc_comb)
            for i, fs in enumerate(features_selected):
                Xi[fs] = Xtmp[:, i]
            Yi = labels_k[stage]
            Ti = timestamps_k[stage]

            # create and train a new model if needed

            if have_to_create_model:

                print(f'Training new model {model_name}:')

                inputs, hidden = model_input(features_selected, xmin, xmax)
                input_type = locals()[args.input]
                hidden = input_type(hidden, nfeatures)
                extractor_type = locals()[feature_extractor]
                hidden = extractor_type(hidden)
                model = model_output(inputs, hidden, br_key, ymin, ymax)
                model_summary_lines = []
                model.summary(print_fn=lambda x: model_summary_lines.append(x))
                model_summary = "\n".join(model_summary_lines)
                if args.verbose and k == 0:
                    print(model_summary)

                model.fit(
                    Xtv[stages[0]], Ytv[stages[0]],
                    #sample_weight=Wtv[stages[0]],
                    validation_data=(Xtv[stages[1]], Ytv[stages[1]]),  # Wtv[stages[1]]
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

            # calculate prediction error for non-permuted features of the class combination

            predictions = model.predict(Xi)[br_key].flatten()
            assert len(predictions) == len(Yi)

            min_errors[k] = np.min(np.abs(Yi - predictions))
            mean_errors[k] = np.mean(np.abs(Yi - predictions))
            max_errors[k] = np.max(np.abs(Yi - predictions))

            errors.extend(np.abs(Yi - predictions))
            reals.extend(Yi)

            print(f'Mean absolute prediction error for combination {model_dc_comb}: {mean_errors[k]}')
            print(f'Min absolute prediction error for combination {model_dc_comb}: {min_errors[k]}')
            print(f'Max absolute prediction error for combination {model_dc_comb}: {max_errors[k]}')

            # calculate prediction error for features permuted

            # TO DO

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
            except:
                pr = pd.DataFrame({
                    ts_key: [value for value in Ti],
                    br_key: [value for value in Yi],
                })

            # save results

            assert model_dc_comb in tbl_dc_combs
            idx = tbl_dc_combs.index(model_dc_comb)
            p_e_mean[model_type].values[idx] = np.mean(mean_errors[:k + 1])
            p_e_mean.to_csv(mean_e_path, index=None)
            p_e_min[model_type].values[idx] = np.mean(min_errors[:k + 1])
            p_e_min.to_csv(min_e_path, index=None)
            p_e_max[model_type].values[idx] = np.mean(max_errors[:k + 1])
            p_e_max.to_csv(max_e_path, index=None)
            pr[model_name] = predictions
            pr.to_csv(r_path, index=None)