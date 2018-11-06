import os
import random
import numpy as np


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


def load_data(filepath, batch_size, test_size=0.2):

    all_data = []

    with open(os.path.join(filepath, 'transcriptions.txt'), 'r') as f:
        lines = f.readlines()
        for file in os.listdir(filepath):
            if file not in ['transcriptions.txt', 'output_space.txt', '.DS_Store']:
                arr = np.load(os.path.join(filepath, file))
                for line in lines:
                    if line.split(' - ')[0] == file[:-4]:
                        all_data.append((arr, line.split(' - ')[1].strip('\n').split(' ')))
                        break

    random.shuffle(all_data)

    test_data = all_data[int(test_size * len(all_data)):]
    train_data = all_data[:int(test_size * len(all_data))]

    x_train = [x[0] for x in train_data]
    x_train_seq_len = [len(x) for x in x_train]

    y_train = [x[1] for x in train_data]

    x_test = [x[0] for x in test_data]
    x_test_seq_len = [len(x) for x in x_test]

    y_test = [x[1] for x in test_data]

    return x_train, x_train_seq_len, y_train, x_test, x_test_seq_len, y_test
