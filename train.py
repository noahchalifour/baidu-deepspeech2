from model import Model
from config import hparams
import utils

import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    sess = tf.InteractiveSession()

    print('Loading data...')

    (x_train, x_train_seq_len, y_train,
     x_test, x_test_seq_len, y_test) = utils.load_data('data', hparams.batch_size)

    hparams.input_max_len = max([max(x_train_seq_len), max(x_test_seq_len)])

    print("Max input length:", hparams.input_max_len)

    x_train = utils.pad_sequences(x_train, hparams.input_max_len)
    x_test = utils.pad_sequences(x_test, hparams.input_max_len)

    print('done.')
    print('Initializing model...')

    model = Model(hparams)

    tf.global_variables_initializer().run()

    print('done.')

    epoch = hparams.n_epochs
    batch_size = hparams.batch_size

    while epoch:

        for i in range(int(len(x_train)/batch_size)):

            batch_train_x = np.asarray(x_train[i*batch_size:(i+1)*batch_size], dtype=np.float32)
            batch_train_x_seq_len = np.asarray(x_train_seq_len[i * batch_size:(i + 1) * batch_size], dtype=np.int32)
            batch_train_y = utils.sparse_tuple_from(np.asarray(y_train[i * batch_size:(i + 1) * batch_size]))

            cost, _ = model.train(batch_train_x, batch_train_x_seq_len, batch_train_y, sess)
            print('cost:', cost)

        if epoch > 0: epoch -= 1