import tensorflow as tf
import numpy as np

from config import nan_value

def model_input(features, steps, xmin, xmax, batchnorm=False, eps=1e-10):

    # input layer

    features_columns = [key for key in features]
    inputs = {colname: tf.keras.layers.Input(name=f'input_{colname}', shape=(steps,), dtype=tf.float32) for colname in features_columns}
    hidden = tf.keras.layers.Concatenate(axis=-1)(list(inputs.values()))
    hidden = tf.keras.layers.Reshape(target_shape=(steps, len(features)))(hidden)

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

def baseline(hidden, nfeatures, nfilters=[256,512,1024], ks=4, ss=4, nhidden=1024):
    for nf in nfilters:
        hidden = tf.keras.layers.Conv1D(nf, ks, strides=ss, padding='same', activation='relu')(hidden)
    return hidden

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

def cnn1(hidden, nhiddens=[1280, 1280], nfilters=1024, kernel_size=2):
    last_conv_kernel_size = hidden.shape[1]
    for nhidden in nhiddens:
        hidden = tf.keras.layers.Conv1D(nhidden, kernel_size, padding='same', activation='relu')(hidden)
    hidden = tf.keras.layers.Conv1D(nfilters, last_conv_kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def cnn1m(hidden, nhiddens=[1280, 1280], nfilters=1024):
    nstreams = len(nhiddens)
    nfilters_max = hidden.shape[1]
    last_conv_kernel_size = nfilters_max * nstreams
    hiddens = []
    for i in range(nstreams):
        nf = np.clip(nfilters_max - i, 1, nfilters_max).astype(int)
        hiddens.append(tf.keras.layers.Conv1D(nhiddens[i], (nf,), padding='same', activation='relu')(hidden))
    hidden = tf.concat(hiddens, axis=1)
    hidden = tf.keras.layers.Conv1D(nfilters, last_conv_kernel_size, activation='relu')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def attention_block(x, nh):
    q = tf.keras.layers.Dense(nh, activation='relu')(x)
    v = tf.keras.layers.Dense(nh, activation='relu')(x)
    a = tf.keras.layers.Dense(1)(tf.nn.tanh(q + v))
    a = tf.keras.layers.Softmax(axis=1)(a)
    h = tf.keras.layers.Multiply()([a, v])
    h = tf.reduce_sum(h, axis=1)
    return h

def lstm(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden)(hidden)
    return hidden

def lstmatt(hidden, nhidden=1280):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden)
    hidden = Attention()(hidden)
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

class Attention(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight1",shape=(input_shape[-1], 1),initializer="normal")
        self.b = self.add_weight(name="att_bias1",shape=(input_shape[1], 1),initializer="zeros")
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

class Attention2(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super(Attention2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name="att_weight1",shape=(input_shape[-1], 1),initializer="normal")
        self.b1 = self.add_weight(name="att_bias1",shape=(input_shape[1] // 2, 1),initializer="zeros")
        self.W2 = self.add_weight(name="att_weight2", shape=(input_shape[-1], 1), initializer="normal")
        self.b2 = self.add_weight(name="att_bias2", shape=(input_shape[1] // 2, 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
        et1=tf.squeeze(tf.tanh(tf.tensordot(x1, self.W1, 1) + self.b1), axis=-1)
        at1=tf.math.softmax(et1)
        at1=tf.expand_dims(at1, axis=-1)
        output1=x2 * at1
        et2 = tf.squeeze(tf.tanh(tf.tensordot(x2, self.W2, 1) + self.b2), axis=-1)
        at2 = tf.math.softmax(et2)
        at2 = tf.expand_dims(at2, axis=-1)
        output2 = x1 * at2
        output = tf.concat([output1, output2], axis=1)
        return tf.math.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention2, self).get_config()

def att(hidden, nhidden=640):
    hidden1 = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden1 = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden1)
    hidden2 = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden2 = tf.keras.layers.LSTM(nhidden, return_sequences=True)(hidden2)
    hidden = tf.concat([hidden1, hidden2], axis=1)
    hidden = Attention()(hidden)
    return hidden

def bilstm_att(hidden, nhidden=640):
    hidden = tf.keras.layers.Masking(mask_value=nan_value)(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nhidden, activation='relu', return_sequences=True))(hidden)
    hidden = Attention()(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    return hidden

def model_output(inputs, hidden, target, ymin, ymax, nhidden=2048, dropout=0.5, lr=1e-6, eps=1e-8):
    hidden = tf.keras.layers.Dense(nhidden, activation='relu')(hidden)
    if dropout is not None:
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation='linear')(hidden)
    outputs = outputs * (ymax - ymin) + ymin
    outputs = {target: outputs}
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    #loss = tf.keras.losses.MeanAbsoluteError()
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps), metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model