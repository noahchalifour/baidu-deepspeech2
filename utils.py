import os
import random
import numpy as np
import tensorflow as tf


def words_to_text(words):

    return ''.join(words)


def compute_seq_lengths(seq):

    used = tf.sign(tf.reduce_max(tf.abs(seq), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def ids_to_text(ids, mapping):

    words = [mapping[x] for x in ids]
    return words_to_text(words)


def load_output_mapping(filename):

    mapping = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    for _ in range(len(lines)):
        for line in lines:
            sections = line.split(' --> ')
            if int(sections[1].strip('\n')) == len(mapping):
                mapping.append(sections[0])

    return mapping


def wer(r, h):

    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def pad_sequences(sequences, length):

    new_sequences = []

    for seq in sequences:
        pad = [np.zeros(161) for _ in range(length-len(seq))]
        if len(pad) > 0:
            new_sequences.append(np.concatenate((seq, pad), axis=0))
        else:
            new_sequences.append(seq)

    return new_sequences


def load_data(filepath, max_data, test_size=0.2):

    all_data = []

    with open(os.path.join(filepath, 'transcriptions.txt'), 'r') as f:
        lines = f.readlines()
        for file in os.listdir(filepath):
            if max_data > 0 and len(all_data) >= max_data:
                all_data = all_data[:max_data]
                break
            if file not in ['transcriptions.txt', 'output_space.txt', '.DS_Store']:
                arr = np.load(os.path.join(filepath, file))
                for line in lines:
                    if line.split(' - ')[0] == file[:-4]:
                        ids = [int(x) for x in line.split(' - ')[1].strip('\n').split(' ')]
                        if len(arr) > len(ids):
                            all_data.append((arr, ids))
                        else:
                            print("error loading transcription: \"{}\"".format(line))
                        break

    random.shuffle(all_data)

    train_data = all_data[int(test_size * len(all_data)):]
    test_data = all_data[:int(test_size * len(all_data))]

    x_train = [x[0] for x in train_data]
    y_train = [x[1] for x in train_data]
    x_test = [x[0] for x in test_data]
    y_test = [x[1] for x in test_data]

    return x_train, y_train, x_test, y_test
