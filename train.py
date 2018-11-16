from model import Model
from config import hparams
import utils

from tensorboard import default as tb_default
from tensorboard import program as tb_program
import os
import time
import shutil
import threading
import tensorflow as tf
tf.reset_default_graph()
import numpy as np

if __name__ == '__main__':

    if os.path.exists(hparams.logdir):
        shutil.rmtree(hparams.logdir)

    os.makedirs(hparams.logdir)

    tb = tb_program.TensorBoard(tb_default.get_plugins(), tb_default.get_assets_zip_provider())
    tb.configure(argv=[None, '--logdir', hparams.logdir])
    tb_thread = threading.Thread(target=tb.main)
    tb_thread.daemon = True
    tb_thread.start()

    graph = tf.Graph()

    print('Loading data...')

    output_mapping = utils.load_output_mapping('data/output_space.txt')
    hparams.n_classes = len(output_mapping) + 1  # not entirely sure we +1 here

    x_train, y_train, x_test, y_test = utils.load_data('data', max_data=hparams.max_data)

    hparams.input_max_len = max([max([len(x) for x in x_train]), max([len(x) for x in x_test])])

    x_train = np.asarray(utils.pad_sequences(x_train, hparams.input_max_len))
    x_test = np.asarray(utils.pad_sequences(x_test, hparams.input_max_len))

    hparams.n_features = x_train.shape[2]

    print('Initializing model...')

    with graph.as_default():

        model = Model(hparams)
        variables_initializer = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter(os.path.join(hparams.logdir, 'train'), graph=graph)
    eval_writer = tf.summary.FileWriter(os.path.join(hparams.logdir, 'eval'), graph=graph)

    config = tf.ConfigProto()
    config.log_device_placement = hparams.log_device_placement
    config.allow_soft_placement = hparams.allow_soft_placement

    sess = tf.Session(graph=graph,
                      config=config)

    sess.run(variables_initializer)

    epoch = hparams.n_epochs
    batch_size = hparams.batch_size
    steps_per_checkpoint = hparams.steps_per_checkpoint
    checkpoints_path = hparams.checkpoints_path
    global_step = 0
    start_time = time.time()

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    checkpoints_path = os.path.join(checkpoints_path, 'checkpoint')

    model.save('model', sess, global_step=0)

    print('Training...')

    try:

        while epoch:

            current_epoch = hparams.n_epochs - epoch

            for i in range(int(len(x_train)/batch_size)):

                batch_train_x = np.asarray(x_train[i*batch_size:(i+1)*batch_size], dtype=np.float32)
                batch_train_y = utils.sparse_tuple_from(np.asarray(y_train[i * batch_size:(i + 1) * batch_size]))

                cost, _, summary = model.train(batch_train_x, batch_train_y, sess)

                global_step += batch_size

                train_writer.add_summary(summary, global_step=global_step)

                print('epoch: {}, global_step: {}, cost: {}, time: {}'.format(current_epoch, global_step, cost, time.time() - start_time))

                if global_step % steps_per_checkpoint == 0:

                    print('checkpointing... (global step = {})'.format(global_step))

                    checkpoint_path = model.saver.save(sess, checkpoints_path, global_step=global_step)

                    ler, summary = model.eval(batch_train_x, batch_train_y, sess)

                    eval_writer.add_summary(summary, global_step=global_step)

                    print('Eval --- LER: {} %'.format(ler*100))

                    decoded_ids = model.infer([batch_train_x[0]], sess)[0][0].values

                    original_text = utils.ids_to_text(y_train[i*batch_size], output_mapping)
                    decoded_text = utils.ids_to_text(decoded_ids, output_mapping)

                    print('GROUND TRUTH: {}'.format(original_text))
                    print('PREDICTION: {}'.format(decoded_text))

            if epoch > 0: epoch -= 1

    except KeyboardInterrupt:

        sess.close()

    sess.close()