# Baidu's Deep Speech 2 (Tensorflow)

(This is a work in progress)

This is a python implementation of Baidu's Deep Speech 2 paper https://arxiv.org/pdf/1512.02595.pdf using tensorflow

# TODO:

<ul>
  <li>Add batch normalization to RNN</li>
  <li>Implement row convolution layer</li>
  <li>Add other dataset support</li>
  <li>Create pretrained models</li>
</ul>

# Preprocessing

To preprocess your data you must first download the one of the datasets above and extract them to a folder. Then run the following script to preprocess the data (This might take a while depending on the amount of data you have)

`python preprocess.py --data-dir=<your data directory> --dataset=<dataset name>`

# Training

Now that you have preprocessed your data, you can train a model. To do this, you can edit the settings in the `config.py` file if you want. Then run the following command to train the model:

`python train.py`

# Testing your model

Now that you have trained a model, you can go ahead and start using it. We have created two scripts that can help you do this `infer.py` and `streaming_infer.py`. The `infer.py` script, transcribes a audio file that you give it

`python infer.py -f <your audio file name>`

The `streaming_infer.py` script uses PyAudio to record audio from your computer's microphone and transcribes it in real-time. To run it simply:

`python streaming_infer.py`
