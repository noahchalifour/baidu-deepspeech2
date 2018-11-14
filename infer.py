from model import Model, ModelModes
import preprocess
import utils

import argparse
import time
from pydub import AudioSegment
import tensorflow as tf

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, type=str)

    args = ap.parse_args()

    infer_sess = tf.Session()

    model = Model.load('model/hparams', 'model/checkpoints/checkpoint-10', ModelModes.INFER, infer_sess)
    output_mapping = utils.load_output_mapping('data/output_space.txt')

    audio_segment = AudioSegment.from_file(args.file)
    pp_audio = preprocess.preprocess_audio(audio_segment.raw_data, audio_segment.frame_rate)
    pp_audio = utils.pad_sequences([pp_audio], model.config.input_max_len)

    start_time = time.time()
    decoded = model.infer(pp_audio, infer_sess)[0][0]
    prediction = utils.ids_to_text(decoded.values, output_mapping)

    print(f"Transcription: {prediction} (took {time.time() - start_time} seconds)")