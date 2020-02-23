#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras

nsamples = 16000 * 30

# Get files for training
clean_files = [str(path) for path in Path('LibriSpeech/dev-clean').glob('**/*.flac')]
noisy_files = [str(path) for path in Path('rnnoise_contributions').glob('*.wav')]

# We assume clean samples are 1-channel 16kHz
# This function returns a 30 second of clean audio.
def get_samples(files):
    data = np.empty(0)
    while len(files) > 0 and data.shape[0] < nsamples:
        file = files.pop()
        data2, samplerate = sf.read(file)
        assert samplerate == 16000
        data = np.append(data, data2)
    return data[:nsamples], files

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
#
# f(x) = sgn(x) * ln(u * |x| + 1) / ln(u + 1) where u = 255
def ulaw(xs):
    u = 255
    xs = np.sign(xs) * np.log(u * np.absolute(xs) + 1) / np.log(u + 1)
    return np.rint(xs * 256).astype(int)

# f^{-1}(y) = sgn(y) * (1 / u) * ((1 + u)^|y| - 1) where u = 255
def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) * (1 / u) * (np.power(1 + u, np.absolute(ys)) - 1)

# Create a keras model
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(32,)))
    model.add(keras.layers.Dense(1, input_shape=(32,), activation='relu'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    model.summary()
    return model

model = get_model()

def train(clean, noisy):
#    mixed = np.clip(clean + noisy, -1, 1)

    #    clean = ulaw(clean)
    #    noisy = ulaw(noisy)
    #    mixed = ulaw(mixed)

    print(clean[:32], clean[0])
    model.fit(x=np.reshape(clean[:32], (1, 32)), y=np.array([clean[0]]))

# Run train()
while len(clean_files) > 0 and len(noisy_files) > 0:
    print(datetime.now(), len(clean_files), len(noisy_files), flush=True)

    clean, clean_files = get_samples(clean_files)
    noisy, noisy_files = get_samples(noisy_files)

    if clean.shape[0] < noisy.shape[0]:
        noisy.resize(clean.shape)
    else:
        clean.resize(noisy.shape)

    train(clean, noisy)
