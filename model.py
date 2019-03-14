import tensorflow as tf


def clipped_relu(x):

    return tf.keras.activations.relu(x, max_value=20)


def ctc_lambda_func(args):

    y_pred, labels, input_length, label_length = args

    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):

    return y_pred


class SpeechModel(object):

    """

    TODO: Implement 2D convolution
    TODO: Add dropout
    TODO: Add different optimizers
    TODO: Test layer batch normalization (at every layer?)

    """

    def __init__(self, hparams):

        input_data = tf.keras.layers.Input(name='inputs', shape=[hparams['max_input_length'], 161])
        x = input_data

        if hparams['use_bn']:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.ZeroPadding1D(padding=(0, hparams['max_input_length']))(x)
        for i in range(len(hparams['conv_channels'])):
            x = tf.keras.layers.Conv1D(hparams['conv_channels'][i], hparams['conv_filters'][i],
                                       strides=hparams['conv_strides'][i], activation='relu', padding='same')(x)

        if hparams['use_bn']:
            x = tf.keras.layers.BatchNormalization()(x)

        for h_units in hparams['rnn_units']:
            if hparams['bidirectional_rnn']:
                h_units = int(h_units / 2)
            gru = tf.keras.layers.GRU(h_units, activation='relu', return_sequences=True)
            if hparams['bidirectional_rnn']:
                gru = tf.keras.layers.Bidirectional(gru, merge_mode='sum')
            x = gru(x)

        if hparams['use_bn']:
            x = tf.keras.layers.BatchNormalization()(x)

        if hparams['future_context'] > 0:
            if hparams['future_context'] > 1:
                x = tf.keras.layers.ZeroPadding1D(padding=(0, hparams['future_context'] - 1))(x)
            x = tf.keras.layers.Conv1D(100, hparams['future_context'], activation='relu')(x)

        y_pred = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hparams['vocab_size'] + 1,
                                                                       activation='sigmoid'))(x)

        labels = tf.keras.layers.Input(name='labels', shape=[None], dtype='int32')
        input_length = tf.keras.layers.Input(name='input_lengths', shape=[1], dtype='int32')
        label_length = tf.keras.layers.Input(name='label_lengths', shape=[1], dtype='int32')

        loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                                           labels,
                                                                                           input_length,
                                                                                           label_length])

        self.model = tf.keras.Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

        if hparams['verbose']:
            print(self.model.summary())

        optimizer = tf.keras.optimizers.Adam(lr=hparams['learning_rate'], beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8, clipnorm=5)

        self.model.compile(optimizer=optimizer, loss=ctc)

    def train_generator(self, generator, train_params):

        callbacks = []

        if train_params['tensorboard']:
            callbacks.append(tf.keras.callbacks.TensorBoard(train_params['log_dir'], write_images=True))

        self.model.fit_generator(generator, epochs=train_params['epochs'],
                                 steps_per_epoch=train_params['steps_per_epoch'],
                                 callbacks=callbacks)