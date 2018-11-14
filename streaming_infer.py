from model import Model, ModelModes
import preprocess
import utils
import pyaudio
import tensorflow as tf

if __name__ == '__main__':

    infer_sess = tf.Session()

    model = Model.load('model/hparams', 'model/checkpoints/checkpoint-10', ModelModes.STREAMING_INFER, infer_sess)
    output_mapping = utils.load_output_mapping('data/output_space.txt')

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    transcription = ""

    # TODO: Test with better performing model

    try:

        while True:

            pp_audio = preprocess.preprocess_audio(stream.read(1024), 16000)
            pp_audio = utils.pad_sequences([pp_audio], model.config.input_max_len)

            decoded = model.infer(pp_audio, infer_sess)[0][0]
            prediction = utils.ids_to_text(decoded.values, output_mapping)

            if prediction != transcription:
                transcription = prediction
                print("Transcription: {}".format(transcription))

    except KeyboardInterrupt:

        stream.stop_stream()
        stream.close()
        exit()
