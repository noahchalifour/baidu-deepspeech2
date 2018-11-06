from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell, GRUCell
import tensorflow as tf


class Model(object):

    def __init__(self, config):

        self.config = config

        self.inputs = tf.placeholder(tf.float32,
                                     shape=[None, None, self.config.n_features],
                                     name='inputs')
        self.input_lens = tf.placeholder(tf.int32,
                                         shape=[None],
                                         name='input_lens')
        self.labels = tf.sparse_placeholder(tf.int32,
                                            name='labels')

        self.conv_weights = {}
        self.conv_biases = {}

        # TODO: Add batch normalization to RNN

        def rnn_cell():

            # TODO: Add bidirectional

            if self.config.rnn_type == 'lstm':
                return BasicLSTMCell(self.config.rnn_size)
            elif self.config.rnn_type == 'gru':
                return GRUCell(self.config.rnn_size)
            else:
                raise Exception(f'Invalid rnn type: {self.config.rnn_type} (Must be lstm or gru)')

        if self.config.rnn_layers == 1:
            self.rnn_cell = rnn_cell()
        else:
            self.rnn_cell = MultiRNNCell([rnn_cell() for _ in range(self.config.rnn_layers)])

        conv_output = tf.reshape(self.inputs, [self.config.batch_size, self.config.input_max_len, self.config.n_features, 1])

        layer_output = 1
        for i in range(self.config.num_conv_layers):

            self.conv_weights[f'W_conv{i+1}'] = tf.Variable(tf.random_normal([5, 5, layer_output, 32*(i+1)]))
            self.conv_biases[f'b_conv{i+1}'] = tf.Variable(tf.random_normal([32*(i+1)]))
            layer_output = 32 * (i + 1)

            conv_output = tf.nn.conv2d(conv_output, self.conv_weights[f'W_conv{i+1}'], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = tf.layers.batch_normalization(conv_output)

        conv_output = tf.reshape(conv_output, [self.config.batch_size, self.config.input_max_len, -1])

        rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, conv_output, self.input_lens, dtype=tf.float32)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.config.rnn_size])

        # TODO: Implement Row Convolution
        
        # fc_W = tf.Variable(tf.truncated_normal([self.config.rnn_size, self.config.n_classes], stddev=0.1))
        # fc_b = tf.Variable(tf.constant(0., shape=[self.config.n_classes]))
        #
        # logits = tf.matmul(rnn_outputs, fc_W) + fc_b

        logits = tf.contrib.layers.fully_connected(rnn_outputs, self.config.n_classes)
        logits = tf.reshape(logits, [self.config.batch_size, -1, self.config.n_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(self.labels, logits, self.input_lens)
        self.cost = tf.reduce_mean(loss)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)

        # TODO: Choose decoder

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, self.input_lens)

    def train(self, inputs, input_len, targets, sess):

        return sess.run([self.cost, self.optimizer], feed_dict={
            self.inputs: inputs,
            self.input_lens: input_len,
            self.labels: targets
        })
