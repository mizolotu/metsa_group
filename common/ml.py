import tensorflow as tf
import numpy as np

from config import nan_value
from sklearn.metrics import roc_auc_score

def model_input(features, xmin, xmax, steps=1, batchnorm=False, eps=1e-10):

    # input layer

    inputs = {colname: tf.keras.layers.Input(name=f'input_{colname}', shape=(steps,), dtype=tf.float32) for colname in features}
    hidden = tf.keras.layers.Concatenate(axis=-1)(list(inputs.values()))
    if steps > 1:
        hidden = tf.keras.layers.Reshape(target_shape=(steps, len(features)))(hidden)
    else:
        hidden = tf.keras.layers.Reshape(target_shape=(len(features),))(hidden)

    # deal with nans

    is_nan = tf.math.is_nan(hidden)
    masks = tf.dtypes.cast(-xmax, tf.float32)
    inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
    inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
    inputs_without_nan = tf.math.multiply_no_nan(hidden, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

    # standardize the input

    inputs_std = (inputs_without_nan - xmin) / (xmax - xmin + eps)
    if batchnorm:
        hidden = tf.keras.layers.BatchNormalization()(inputs_std)
    else:
        hidden = inputs_std

    return inputs, hidden

def split(hidden, nfeatures, latent_dim=256):
    hidden_spl = tf.split(hidden, nfeatures, axis=-1)
    hidden = []
    for spl in hidden_spl:
        hidden.append(tf.keras.layers.Dense(latent_dim, activation='relu')(spl))
    hidden = tf.stack(hidden, axis=-2)
    return hidden

def mlp(hidden, nhiddens=[2048, 2048], dropout=0.5):
    hidden = tf.keras.layers.Flatten()(hidden)
    for nhidden in nhiddens:
        hidden = tf.keras.layers.Dense(nhidden, activation='relu')(hidden)
        if dropout is not None:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
    return hidden

def cnn1(hidden, nhiddens=[256, 512], nfilters=1024, kernel_size=2):
    last_conv_kernel_size = hidden.shape[-2]
    for nhidden in nhiddens:
        hidden = tf.keras.layers.Conv1D(nhidden, kernel_size, padding='same', activation='relu')(hidden)
    hidden = tf.keras.layers.Conv1D(nfilters, last_conv_kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def lstm(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden)(hidden)
    return hidden

def bilstm(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nhidden, activation='relu', return_sequences=False))(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def cnn1lstm(hidden, nfilters=[1280, 1280], kernel_size=2, nhidden=640):
    for nf in nfilters:
        hidden = tf.keras.layers.Conv1D(nf, kernel_size, padding='same', activation='relu')(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, activation='relu', return_sequences=False)(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def som(features, xmin, xmax, nfeatures, target, layers=[64, 64], lr=1e-6):
    model = SOM(layers, features, xmin, xmax, nfeatures, target)
    model.build(input_shape={f: (None, 1) for f in features})
    model.compute_output_shape({f: (None, 1) for f in features})
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr))
    return model

def model_output(inputs, hidden, target, ymin, ymax, nhidden=2048, dropout=0.5, lr=1e-6, eps=1e-8):
    if dropout is not None:
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
    hidden = tf.keras.layers.Dense(nhidden, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(1, activation='linear')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    outputs = {target: outputs}
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model

class SOMLayer(tf.keras.layers.Layer):

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        self.initial_prototypes = prototypes
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.prototypes = self.add_weight(shape=(self.nprototypes, *input_dims), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        d = tf.reduce_mean(tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=-1), axis=-1)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.nprototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def som_loss(weights, distances):
    return tf.reduce_sum(weights * distances, axis=1)

class SOM(tf.keras.models.Model):

    def __init__(self, map_size, features, xmin, xmax, nfeatures, target, split_neurons=256, encoder_filters=[256, 512, 1024], decoder_filters=256, T_min=0.1, T_max=10.0, niterations=10000, nnn=4, batchnorm=False, eps=1e-10):
        super(SOM, self).__init__()

        self.map_size = map_size
        self.features = features
        self.target = target
        self.xmin = xmin
        self.xmax = xmax
        self.nfeatures = nfeatures
        self.batchnorm = batchnorm
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.eps = eps

        self.nprototypes = np.prod(map_size)
        ranges = [np.arange(m) for m in map_size]
        mg = np.meshgrid(*ranges, indexing='ij')
        self.prototype_coordinates = tf.convert_to_tensor(np.array([item.flatten() for item in mg]).T)
        self.split_layer = [tf.keras.layers.Dense(split_neurons, activation='relu') for _ in self.nfeatures]
        self.unsplit_layer = [tf.keras.layers.Dense(nf, activation='relu') for nf in self.nfeatures]
        self.cnn_encoder = [tf.keras.layers.Conv1D(nf, 2, activation='relu', padding='same') for nf in self.encoder_filters[:-1]]
        self.cnn_encoder.append(tf.keras.layers.Conv1D(self.encoder_filters[-1], len(self.nfeatures), activation='relu'))
        self.cnn_decoder = [tf.keras.layers.Conv1DTranspose(decoder_filters, 2, activation='relu') for _ in self.nfeatures[:-1]]
        self.som_layer = SOMLayer(map_size, name='som_layer')
        self.T_min = T_min
        self.T_max = T_max
        self.niterations = niterations
        self.current_iteration = 0
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.re_tracker = tf.keras.metrics.Mean(name='re')
        self.acc_tracker = ReconstructionAccuracy('acc')
        self.nnn = nnn

    @property
    def prototypes(self):
        return self.som_layer.get_weights()[0]

    def call(self, x):

        # input

        x = tf.keras.layers.Concatenate(axis=-1)([x[f] for f in self.features])

        # deal with nans

        is_nan = tf.math.is_nan(x)
        masks = tf.dtypes.cast(-self.xmax, tf.float32)
        inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
        inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
        inputs_without_nan = tf.math.multiply_no_nan(x, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

        # standardize the input

        x = (inputs_without_nan - self.xmin) / (self.xmax - self.xmin + self.eps)

        # split

        x_tmp = tf.split(x, self.nfeatures, axis=-1)
        x_spl = []
        for i, spl in enumerate(x_tmp):
            x_spl.append(self.split_layer[i](spl))
        x_spl = tf.stack(x_spl, axis=-2)

        # encoding

        for layer in self.cnn_encoder:
            x_spl = layer(x_spl)

        # decoding

        x_rec = x_spl
        for layer in self.cnn_decoder:
            x_rec = layer(x_rec)
        x_tmp = tf.split(x_rec, len(self.nfeatures), axis=1)
        x_rec = []
        for i, spl in enumerate(x_tmp):
            x_rec.append(tf.keras.layers.Flatten()(self.unsplit_layer[i](spl)))
        x_rec = tf.concat(x_rec, axis=-1)

        # cluster

        c = self.som_layer(x_spl)
        s = tf.sort(c, axis=1)
        spl = tf.split(s, [self.nnn, self.nprototypes - self.nnn], axis=1)

        # error

        cl_dists = tf.reduce_mean(spl[0], axis=1)
        rec_errors = tf.math.sqrt(tf.reduce_sum(tf.square(x - x_rec), axis=-1))

        return tf.math.add(cl_dists, rec_errors)

    def map_dist(self, y_pred):
        labels = tf.gather(self.prototype_coordinates, y_pred)
        mh = tf.reduce_sum(tf.math.abs(tf.expand_dims(labels, 1) - tf.expand_dims(self.prototype_coordinates, 0)), axis=-1)
        return tf.cast(mh, tf.float32)

    @staticmethod
    def neighborhood_function(d, T):
        return tf.math.exp(-(d ** 2) / (T ** 2))

    def train_step(self, data):

        # input

        x, y = data
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(x[f], -1) for f in self.features])
        y = y[self.target]

        with tf.GradientTape() as tape:

            # deal with nans

            is_nan = tf.math.is_nan(x)
            masks = tf.dtypes.cast(-self.xmax, tf.float32)
            inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
            inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
            inputs_without_nan = tf.math.multiply_no_nan(x, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

            # standardize the input

            x = (inputs_without_nan - self.xmin) / (self.xmax - self.xmin + self.eps)

            # split

            x_tmp = tf.split(x, self.nfeatures, axis=-1)
            x_spl = []
            for i, spl in enumerate(x_tmp):
                x_spl.append(self.split_layer[i](spl))
            x_spl = tf.stack(x_spl, axis=-2)

            # encoding

            for layer in self.cnn_encoder:
                x_spl = layer(x_spl)

            # decoding

            x_rec = x_spl
            for layer in self.cnn_decoder:
                x_rec = layer(x_rec)
            x_tmp = tf.split(x_rec, len(self.nfeatures), axis=1)
            x_rec = []
            for i, spl in enumerate(x_tmp):
                x_rec.append(tf.keras.layers.Flatten()(self.unsplit_layer[i](spl)))
            x_rec = tf.concat(x_rec, axis=-1)

            # compute cluster assignments for batches

            d = self.som_layer(x_spl)
            y_pred = tf.math.argmin(d, axis=1)

            # Update temperature parameter

            self.current_iteration += 1
            if self.current_iteration > self.niterations:
                self.current_iteration = self.niterations
            self.T = self.T_max * (self.T_min / self.T_max) ** (self.current_iteration / (self.niterations - 1))

            # Compute topographic weights batches

            w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)

            # calculate loss

            rec_errors = tf.math.sqrt(tf.reduce_sum(tf.square(x - x_rec), axis=-1))
            cl_dists = som_loss(w_batch, d)
            losses = tf.math.add(rec_errors, cl_dists)
            loss = tf.reduce_mean(losses)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.re_tracker.update_state(tf.reduce_mean(rec_errors))
        self.acc_tracker.update_state(y, losses)

        return {
            "loss": self.loss_tracker.result(),
            "re": self.re_tracker.result(),
            "acc": self.acc_tracker.result()
        }

    def test_step(self, data):

        # input

        x, y = data
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(x[f], -1) for f in self.features])
        y = y[self.target]

        # deal with nans

        is_nan = tf.math.is_nan(x)
        masks = tf.dtypes.cast(-self.xmax, tf.float32)
        inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
        inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
        inputs_without_nan = tf.math.multiply_no_nan(x, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

        # standardize the input

        x = (inputs_without_nan - self.xmin) / (self.xmax - self.xmin + self.eps)

        # split

        x_tmp = tf.split(x, self.nfeatures, axis=-1)
        x_spl = []
        for i, spl in enumerate(x_tmp):
            x_spl.append(self.split_layer[i](spl))
        x_spl = tf.stack(x_spl, axis=-2)

        # encoding

        for layer in self.cnn_encoder:
            x_spl = layer(x_spl)

        # decoding

        x_rec = x_spl
        for layer in self.cnn_decoder:
            x_rec = layer(x_rec)
        x_tmp = tf.split(x_rec, len(self.nfeatures), axis=1)
        x_rec = []
        for i, spl in enumerate(x_tmp):
            x_rec.append(tf.keras.layers.Flatten()(self.unsplit_layer[i](spl)))
        x_rec = tf.concat(x_rec, axis=-1)

        # clustering

        d = self.som_layer(x_spl)
        y_pred = tf.math.argmin(d, axis=1)
        w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)

        rec_errors = tf.math.sqrt(tf.reduce_sum(tf.square(x - x_rec), axis=-1))
        cl_dists = som_loss(w_batch, d)
        losses = tf.math.add(rec_errors, cl_dists)
        loss = tf.reduce_mean(losses)

        self.loss_tracker.update_state(loss)
        self.re_tracker.update_state(tf.reduce_mean(rec_errors))
        self.acc_tracker.update_state(y, losses)

        return {
            "loss": self.loss_tracker.result(),
            "re": self.re_tracker.result(),
            "acc": self.acc_tracker.result()
        }

class EarlyStoppingAtMaxMetric(tf.keras.callbacks.Callback):

    def __init__(self, validation_data, metric, patience=10, max_fpr=1.0):
        super(EarlyStoppingAtMaxMetric, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.metric = metric
        self.validation_data = validation_data
        self.current = -np.Inf
        self.max_fpr = max_fpr

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if np.greater(self.current, self.best):
            self.best = self.current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_test_end(self, logs):
        x, y = self.validation_data
        predictions = self.model.predict(x)
        probs = predictions.flatten()
        if self.metric == 'auc':
            self.current = roc_auc_score(y, probs, max_fpr=self.max_fpr)
        elif self.metric == 'acc':
            n = len(y)
            p0 = probs[np.where(y == 0)[0]]
            p1 = probs[np.where(y == 1)[0]]
            p0si = np.argsort(p0)
            p1si = np.argsort(p1)
            p0s = p0[p0si]
            p1s = p1[p1si]
            n0 = len(p0s)
            n1 = len(p1s)
            if p1s[0] > p0s[-1]:
                acc = [1]
            else:
                idx = np.where(p0s > p1s[0])[0]
                acc = [float(len(p0s) - len(idx) + len(p1s)) / n, *np.zeros(len(idx))]
                h = n0 - len(idx)
                n10 = 0
                for i, j in enumerate(idx):
                    thr = p0s[j]
                    thridx = np.where(p1s[n10:] < thr)[0]
                    n10 += len(thridx)
                    h += 1
                    acc[i + 1] = (h - n10 + n1) / n
            self.current = np.max(acc)
        else:
            raise NotImplemented
        print(f'\nValidation {self.metric}:', self.current)

class ReconstructionAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='reconstruction_accuracy', alpha=3, **kwargs):
        super(ReconstructionAccuracy, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.reconstruction_errors = tf.Variable([], shape=(None,), validate_shape=False)
        self.true_labels = tf.Variable([], shape=(None,), validate_shape=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(y_true, 0, 1)
        self.reconstruction_errors.assign(tf.concat([self.reconstruction_errors.value(), y_pred], axis=0))
        self.true_labels.assign(tf.concat([self.true_labels.value(), y_true], axis=0))

    def result(self):
        thr = tf.reduce_mean(self.reconstruction_errors) + self.alpha * tf.math.reduce_std(self.reconstruction_errors)
        predictions = tf.math.greater_equal(self.reconstruction_errors, thr)
        true_labels = tf.cast(self.true_labels, tf.bool)
        true_positives = tf.logical_and(tf.equal(predictions, True), tf.equal(true_labels, True))
        true_positives = tf.cast(true_positives, self.dtype)
        true_negatives = tf.logical_and(tf.equal(predictions, False), tf.equal(true_labels, False))
        true_negatives = tf.cast(true_negatives, self.dtype)
        false_positives = tf.logical_and(tf.equal(predictions, True), tf.equal(true_labels, False))
        false_positives = tf.cast(false_positives, self.dtype)
        false_negatives = tf.logical_and(tf.equal(predictions, False), tf.equal(true_labels, True))
        false_negatives = tf.cast(false_negatives, self.dtype)
        return (tf.reduce_sum(true_positives) + tf.reduce_sum(true_negatives))  / (tf.reduce_sum(true_positives) + tf.reduce_sum(true_negatives) + tf.reduce_sum(false_positives) + tf.reduce_sum(false_negatives))

    def reset_states(self):
        self.reconstruction_errors.assign([])
        self.true_labels.assign([])