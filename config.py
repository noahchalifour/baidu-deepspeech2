from tensorflow.contrib.training import HParams

hparams = HParams(

    batch_size = 2,
    n_features = 161,
    learning_rate = 0.001,
    n_epochs = 30,
    steps_per_checkpoint = 10,
    checkpoints_path = 'model/checkpoints',
    max_data = 10,
    optimizer = 'sgd',

    # CNN
    num_conv_layers = 1,    # Best: 3

    # RNN
    rnn_type = 'gru',
    rnn_layers = 3,         # Best: 7
    rnn_size = 128,
    bidirectional_rnn = True,

    # Row convolution
    future_context = 2,

    # Decoder
    beam_width = 32

)