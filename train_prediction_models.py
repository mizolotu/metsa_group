import os, json
import os.path as osp

import pandas as pd
import tensorflow as tf
import numpy as np
import argparse as arp

from config import *

def load_meta(fpath):
    meta = None
    try:
        with open(fpath) as f:
            meta = json.load(f)
    except Exception as e:
        print(e)
    return meta

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def load_data(fpath, tags):
    df = pd.read_csv(fpath)
    values = df[tags].values
    labels = df[br_key].values
    timestamps = df[ts_key].values
    return values, labels, timestamps, tags.copy()

def pad_data(X, x_features, features, delay_classes, dc_comb):
    nan_cols = []
    dc_comb = [int(dc) for dc in dc_comb.split(',')]
    for i, xf in enumerate(x_features):
        dc = delay_classes[features.index(xf)]
        if dc not in dc_comb:
            nan_cols.append(i)
    X_padded = X.copy()
    X_padded[:, nan_cols] = np.nan
    return X_padded

def model_input(nfeatures, xmin, xmax, latent_dim=64, batchnorm=False):

    # input layer

    nfeatures_sum = np.sum(nfeatures)
    inputs = tf.keras.layers.Input(shape=(nfeatures_sum,))

    # deal with nans

    is_nan = tf.math.is_nan(inputs)
    masks = tf.dtypes.cast(-xmax, tf.float32)
    inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
    inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
    inputs_without_nan = tf.math.multiply_no_nan(inputs, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

    # standardize the input

    inputs_std = (inputs_without_nan - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std

    # split, stack and flatten

    hidden_spl = tf.split(hidden, nfeatures, axis=1)
    hidden = []
    for spl in hidden_spl:
        hidden.append(tf.keras.layers.Dense(latent_dim, activation='relu')(spl))
    hidden = tf.stack(hidden, axis=1)

    return inputs, hidden

def mlp(hidden, nhidden=2048):
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(nhidden, activation='relu')(hidden)
    return hidden

def cnn(hidden, nfilters=1024, kernel_size=3):
    hidden = tf.keras.layers.Conv1D(nfilters, kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Conv1D(nfilters, kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def attention_block(x, nh):
    q = tf.keras.layers.Dense(nh, use_bias=False)(x)
    k = tf.keras.layers.Dense(nh, use_bias=False)(x)
    v = tf.keras.layers.Dense(nh, use_bias=False)(x)
    a = tf.keras.layers.Multiply()([q, k])
    a = tf.keras.layers.Softmax(axis=-1)(a)
    h = tf.keras.layers.Multiply()([a, v])
    return h

def lstm(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden)(hidden)
    return hidden

class Attention(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",shape=(input_shape[-1], 1),initializer="normal")
        self.b = self.add_weight(name="att_bias",shape=(input_shape[1], 1),initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et=tf.squeeze(tf.tanh(tf.tensordot(x, self.W, 1) + self.b), axis=-1)
        at=tf.math.softmax(et)
        at=tf.expand_dims(at, axis=-1)
        output=x * at
        return tf.math.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

def lstm_att(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = Attention()(hidden)
    return hidden

def bilstm(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nhidden, activation='relu', return_sequences=False))(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def bilstm_att(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nhidden, activation='relu', return_sequences=True))(hidden)
    hidden = Attention()(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def model_output(inputs, hidden, ymin, ymax, layers=[2048, 2048], dropout=0.5, lr=2.5e-4):
    for nh in layers:
        hidden = tf.keras.layers.Dense(nh, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='linear')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

if __name__ == '__main__':

    # args

    parser = arp.ArgumentParser(description='Train classifiers')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-e', '--extractor', help='feature extractor', default='mlp', choices=['mlp', 'cnn', 'lstm', 'lstm_att', 'bilstm', 'bilstm_att'])
    parser.add_argument('-f', '--firstclass', help='Delay class when prediction starts', type=int, default=1)
    parser.add_argument('-l', '--lastclass', help='Delay class when prediction ends', type=int, default=5)
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=0)
    parser.add_argument('-c', '--cuda', help='Use CUDA', default=False, type=bool)
    parser.add_argument('-v', '--verbose', help='Verbose', default=True, type=bool)
    parser.add_argument('-y', '--ylimits', help='Use bleach ratio limits from data?', default=False, type=bool)
    parser.add_argument('-r', '--retrain', help='Retrain model?', default=False, type=bool)
    parser.add_argument('-n', '--ntests', help='Number of tests', type=int, default=1)
    parser.add_argument('-m', '--mode', help='Mode', default='development', choices=['development', 'production'])
    args = parser.parse_args()

    # number of tests

    if args.mode == 'production':
        ntests = 1
    else:
        ntests = args.ntests

    # cuda

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # set seed for results reproduction

    set_seeds(seed)

    # load meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']
    uclasses = np.sort(np.unique(classes))
    features_selected, feature_classes_selected = [list(item) for item in zip(*[(f, c) for f, c in zip(features, classes) if c <= args.lastclass])]
    u_feature_classes_selected = np.unique(feature_classes_selected)
    u_classes_selected = [c for c in feature_classes_selected if c >= args.firstclass]
    nfeatures = []
    dcs = []
    dc_combs = []
    tbl_dc_combs = []
    for uc in uclasses:
        dcs.extend(str(uc))
        dc_comb = ','.join([item for item in dcs])
        tbl_dc_combs.append(dc_comb)
        nf = len([c for c in classes if c == uc])
        if uc in u_feature_classes_selected:
            nfeatures.append(nf)
            if uc in u_classes_selected:
                dc_combs.append(dc_comb)

    print(f'The following feature classes will be used to train the model: {dc_combs}')

    # load data

    values, labels, timestamps, val_features = load_data(osp.join(task_dir, features_fname), features_selected)
    print(values.shape)

    # model name

    if args.firstclass == args.lastclass:
        model_name = f'{args.extractor}_{args.firstclass}'
    else:
        model_name = f'{args.extractor}_{args.firstclass}_{args.lastclass}'

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

    if args.mode == 'production' and not args.retrain:
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

    errors = np.zeros(ntests)
    for k in range(ntests):
        print(f'Test {k + 1}/{ntests}:')

        # data split

        inds = np.arange(len(labels))
        inds_splitted = [[] for _ in stages]
        np.random.shuffle(inds)
        val, remaining = np.split(inds, [int(validation_share * len(inds))])
        if args.mode == 'production':
            tr = remaining
            te = np.array([], dtype=int)
        else:
            tr, te = np.split(remaining, [int(train_test_ratio * len(remaining))])
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

        # create datasets by padding certain feature classes

        Xtv, Ytv = {}, {}

        for stage in stages[:-1]:
            Xtv[stage] = []
            Ytv[stage] = []
            for dc_comb in dc_combs:
                Xtv[stage].append(pad_data(values_k[stage], val_features, features, classes, dc_comb))
                Ytv[stage].append(labels_k[stage])
            Xtv[stage] = np.vstack(Xtv[stage])
            Ytv[stage] = np.hstack(Ytv[stage])

        if args.mode == 'production':
            stage = stages[1]
        else:
            stage = stages[2]
        Xi = {}
        for dc_comb in dc_combs:
            Xi[dc_comb] = pad_data(values_k[stage], val_features, features, classes, dc_comb)
        Yi = labels_k[stage]
        Ti = timestamps_k[stage]

        # create and train a new model if needed

        if have_to_create_model:

            print(f'Training new model {model_name}:')
            inputs, hidden = model_input(nfeatures, xmin, xmax)
            extractor_type = locals()[args.extractor]
            hidden = extractor_type(hidden)
            model = model_output(inputs, hidden, ymin, ymax)
            model_summary_lines = []
            model.summary(print_fn=lambda x: model_summary_lines.append(x))
            model_summary = "\n".join(model_summary_lines)
            if args.verbose and k == 0:
                print(model_summary)

            model.fit(
                Xtv[stages[0]], Ytv[stages[0]],
                validation_data=(Xtv[stages[1]], Ytv[stages[1]]),
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

        # results tables

        e_path = osp.join(results_mode_dir, prediction_errors_fname)
        r_path = osp.join(results_mode_dir, prediction_results_fname)

        try:
            pe = pd.read_csv(e_path)
        except:
            pe = pd.DataFrame({
                'Delay classes': [comb for comb in tbl_dc_combs]
            })

        if model_name not in pe.keys():
            pe[model_name] = [np.nan for comb in tbl_dc_combs]

        try:
            pr = pd.read_csv(r_path)
        except:
            pr = pd.DataFrame({
                ts_key: [value for value in Ti],
                br_key: [value for value in Yi],
            })

        # calculate prediction error for each class combination

        for dc_comb in dc_combs:
            predictions = model.predict(Xi[dc_comb]).flatten()
            assert len(predictions) == len(Yi)
            errors[k] = np.mean(np.abs(Yi - predictions))
            print(f'Prediction error for combination {dc_comb}: {errors[k]}')

            # save results

            assert dc_comb in tbl_dc_combs
            idx = tbl_dc_combs.index(dc_comb)
            pe[model_name].values[idx] = np.mean(errors[:k+1])
            pe.to_csv(e_path, index=None)
            model_comb = f'{model_name}_{dc_comb}'
            pr[model_comb] = predictions
            pr.to_csv(r_path, index=None)