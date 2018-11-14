from pydub import AudioSegment
from scipy import signal
from sklearn import preprocessing
import os
import numpy as np


def log_mel_spectrogram():

    # TODO: Implement

    pass


def log_linear_specgram(audio, sample_rate, window_size=10,
                        step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    _, _, spec = signal.spectrogram(audio, fs=sample_rate,
                                    window='hann', nperseg=nperseg, noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)


def preprocess_audio(audio):

    # TODO: Normalize audio
    # TODO: Maybe convert data to mono?

    arr_data = np.fromstring(audio.raw_data, dtype=np.int16)

    features = log_linear_specgram(arr_data, audio.frame_rate, window_size=20, step_size=10)
    features = np.asarray([preprocessing.scale(spec_bin.astype(float)) for spec_bin in features])

    return features


def preprocess_librispeech(master_directory):

    output_space = {'<null>': 0}

    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/transcriptions.txt', 'w') as transcriptions_f:

        for root, dirs, files in os.walk(master_directory):
            if files and root != master_directory:
                for file in files:
                    if file[-5:] == '.flac':
                        audio = AudioSegment.from_file(os.path.join(root, file))
                        pp_audio = preprocess_audio(audio)
                        np.save('data/' + file[:-5] + '.npy', pp_audio)
                    elif file[-4:] == '.txt':
                        with open(os.path.join(root, file), 'r') as f:
                            for line in f.readlines():
                                audio_file_id = line.split(' ')[0]
                                transcription = ' '.join(line.split(' ')[1:]).strip('\n').lower()
                                id_transcription = ""
                                for c in transcription:
                                    if c not in output_space.keys():
                                        output_space[c] = len(output_space.keys())
                                    id_transcription += str(output_space[c]) + " "
                                transcriptions_f.write(audio_file_id + ' - ' + id_transcription.strip() + '\n')

    with open('data/output_space.txt', 'w') as f:
        for k, v in output_space.items():
            f.write(k + ' --> ' + str(v) + '\n')


# if __name__ == '__main__':
#
#     preprocess_librispeech('librispeech_data/LibriSpeech')
