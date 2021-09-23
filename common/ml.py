import tensorflow as tf
import numpy as np

from config import nan_value

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

def som(features, xmin, xmax, nfeatures, layers=[64, 64], lr=1e-6):
    model = SOM(layers, features, xmin, xmax, nfeatures)
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
    return tf.reduce_mean(tf.reduce_sum(weights * distances, axis=1))

class SOM(tf.keras.models.Model):

    def __init__(self, map_size, features, xmin, xmax, nfeatures, split_neurons=256, cnn_filters=[256, 512, 1024], T_min=0.1, T_max=10.0, niterations=10000, nnn=4, batchnorm=False, eps=1e-10):
        super(SOM, self).__init__()

        self.map_size = map_size
        self.features = features
        self.xmin = xmin
        self.xmax = xmax
        self.nfeatures = nfeatures
        self.batchnorm = batchnorm
        self.eps = eps

        self.nprototypes = np.prod(map_size)
        ranges = [np.arange(m) for m in map_size]
        mg = np.meshgrid(*ranges, indexing='ij')
        self.prototype_coordinates = tf.convert_to_tensor(np.array([item.flatten() for item in mg]).T)
        self.split_layer = [tf.keras.layers.Dense(split_neurons, activation='relu') for _ in self.nfeatures]
        self.som_layer = SOMLayer(map_size, name='som_layer')
        self.T_min = T_min
        self.T_max = T_max
        self.niterations = niterations
        self.current_iteration = 0
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.nnn = nnn

        self.inputs = {colname: tf.keras.layers.Input(name=f'input_{colname}', shape=(1,), dtype=tf.float32) for colname in self.features}
        self.built = True

    @property
    def prototypes(self):
        return self.som_layer.get_weights()[0]

    def call(self, x):

        # input

        x = self.inputs(x)
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(x[f], -1) for f in self.features])

        # deal with nans

        is_nan = tf.math.is_nan(x)
        masks = tf.dtypes.cast(-self.xmax, tf.float32)
        inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
        inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
        inputs_without_nan = tf.math.multiply_no_nan(x, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

        # standardize the input

        inputs_std = (inputs_without_nan - self.xmin) / (self.xmax - self.xmin + self.eps)

        # split

        x_tmp = tf.split(inputs_std, self.nfeatures, axis=-1)
        x_spl = []
        for i, spl in enumerate(x_tmp):
            x_spl.append(self.split_layer[i](spl))
        x_spl = tf.stack(x_spl, axis=-2)

        # cluster

        c = self.som_layer(x_spl)
        s = tf.sort(c, axis=1)
        spl = tf.split(s, [self.nnn, self.nprototypes - self.nnn], axis=1)

        return tf.reduce_mean(spl[0], axis=1)

    def map_dist(self, y_pred):
        labels = tf.gather(self.prototype_coordinates, y_pred)
        mh = tf.reduce_sum(tf.math.abs(tf.expand_dims(labels, 1) - tf.expand_dims(self.prototype_coordinates, 0)), axis=-1)
        return tf.cast(mh, tf.float32)

    @staticmethod
    def neighborhood_function(d, T):
        return tf.math.exp(-(d ** 2) / (T ** 2))

    def train_step(self, data):

        # input

        x, outputs = data
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(x[f], -1) for f in self.features])

        with tf.GradientTape() as tape:

            # deal with nans

            is_nan = tf.math.is_nan(x)
            masks = tf.dtypes.cast(-self.xmax, tf.float32)
            inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
            inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
            inputs_without_nan = tf.math.multiply_no_nan(x, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

            # standardize the input

            inputs_std = (inputs_without_nan - self.xmin) / (self.xmax - self.xmin + self.eps)

            # split

            x_tmp = tf.split(inputs_std, self.nfeatures, axis=-1)
            x_spl = []
            for i, spl in enumerate(x_tmp):
                x_spl.append(self.split_layer[i](spl))
            x_spl = tf.stack(x_spl, axis=-2)

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

            loss = som_loss(w_batch, d)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):

        # input

        x, outputs = data
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(x[f], -1) for f in self.features])

        # deal with nans

        is_nan = tf.math.is_nan(x)
        masks = tf.dtypes.cast(-self.xmax, tf.float32)
        inputs_nan = tf.dtypes.cast(is_nan, dtype=tf.float32)
        inputs_not_nan = tf.dtypes.cast(tf.math.logical_not(is_nan), dtype=tf.float32)
        inputs_without_nan = tf.math.multiply_no_nan(x, inputs_not_nan) + tf.math.multiply_no_nan(masks, inputs_nan)

        # standardize the input

        inputs_std = (inputs_without_nan - self.xmin) / (self.xmax - self.xmin + self.eps)

        # split

        x_tmp = tf.split(inputs_std, self.nfeatures, axis=-1)
        x_spl = []
        for i, spl in enumerate(x_tmp):
            x_spl.append(self.split_layer[i](spl))
        x_spl = tf.stack(x_spl, axis=-2)

        d = self.som_layer(x_spl)
        y_pred = tf.math.argmin(d, axis=1)
        w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)
        loss = som_loss(w_batch, d)
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }