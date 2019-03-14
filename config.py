preprocessing = {

    'data_dir': 'data',
    'window_size': 20,
    'step_size': 10,

}

model = {

    'verbose': 1,

    'conv_channels': [100],
    'conv_filters': [5],
    'conv_strides': [2],

    'rnn_units': [64],
    'bidirectional_rnn': True,

    'future_context': 2,

    'use_bn': True,

    'learning_rate': 0.001

}

training = {

    'tensorboard': False,
    'log_dir': './logs',

    'batch_size': 64,
    'epochs': 5,
    'validation_size': 0.2

}