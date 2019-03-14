from config import preprocessing as conf_preprocess
import utils

from scipy import signal
from tqdm import tqdm
import os
import argparse
import time
import csv
import soundfile as sf
import numpy as np


def log_linear_specgram(audio, sample_rate, window_size=20,
                        step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    _, _, spec = signal.spectrogram(audio, fs=sample_rate,
                                    window='hann', nperseg=nperseg, noverlap=noverlap,
                                    detrend=False)

    return np.log(spec.T.astype(np.float32) + eps)


def preprocess_librispeech(directory):

    # TODO: Maybe normalize data

    print("Pre-processing LibriSpeech corpus")

    start_time = time.time()

    character_mapping = utils.create_character_mapping()

    if not os.path.exists(conf_preprocess['data_dir']):
        os.makedirs(conf_preprocess['data_dir'])

    dir_walk = list(os.walk(directory))
    num_hours = 0

    with open(os.path.join(conf_preprocess['data_dir'], 'metadata.csv'), 'w', newline='') as metadata:
        metadata_writer = csv.DictWriter(metadata, fieldnames=['filename', 'spec_length', 'labels_length', 'labels'])
        metadata_writer.writeheader()
        for root, dirs, files in tqdm(dir_walk):
            for file in files:
                if file[-4:] == '.txt':
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f.readlines():
                            sections = line.split(' ')
                            audio, sr = sf.read(os.path.join(root, sections[0] + '.flac'))
                            num_hours += (len(audio) / sr) / 3600
                            spec = log_linear_specgram(audio, sr, window_size=conf_preprocess['window_size'],
                                                       step_size=conf_preprocess['step_size'])
                            np.save(os.path.join(conf_preprocess['data_dir'], sections[0] + '.npy'), spec)
                            ids = [character_mapping[c] for c in ' '.join(sections[1:]).lower()
                                   if c in character_mapping]
                            metadata_writer.writerow({
                                'filename': sections[0],
                                'spec_length': spec.shape[0],
                                'labels_length': len(ids),
                                'labels': ' '.join([str(i) for i in ids])
                            })

    print("Done!")
    print("Hours pre-processed: " + str(num_hours))
    print("Time: " + str(time.time() - start_time))


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, type=str, help='The directory of your data to be preprocessed.')
    ap.add_argument('--dataset', required=True, type=str, help='The type of dataset you are using (librispeech)')

    args = ap.parse_args()

    if args.dataset == 'librispeech':
        preprocess_librispeech(args.data_dir)
    else:
        raise Exception("Invalid dataset \"{}\" must be (librispeech)".format(args.dataset))
