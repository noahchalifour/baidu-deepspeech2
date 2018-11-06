from tensorflow.contrib.training import HParams

hparams = HParams(

    batch_size = 64,
    n_features = 161,
    n_classes = 30,
    learning_rate = 0.001,
    n_epochs = 1,

    num_conv_layers = 1,

    rnn_type = 'gru',
    rnn_layers = 3,
    rnn_size = 128

)