#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model

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
    xs = np.rint(xs * 256)
    xs = np.clip(xs, -128, 127)
    return np.rint(xs).astype(int)

# f^{-1}(y) = sgn(y) * (1 / u) * ((1 + u)^|y| - 1) where u = 255
def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) * (1 / u) * (np.power(1 + u, np.absolute(ys)) - 1)

# Create a keras model
def get_model():
    inputs = Input(shape=(nsamples,1))
    f = Conv1D(filters=512, kernel_size=2, padding='same', activation='tanh')(inputs)
    f = Conv1D(filters=512, kernel_size=2, padding='same', activation='tanh')(f)
    f = Conv1D(filters=256, kernel_size=1, activation='softmax')(f)

    model = Model(inputs = inputs, outputs=f)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

model = get_model()

def train(clean, noisy):
    #    mixed = np.clip(clean + noisy, -1, 1)

    #    clean = ulaw(clean)
    #    noisy = ulaw(noisy)
    #    mixed = ulaw(mixed)

    print(clean[:8], clean[0])
    model.fit(x=np.reshape(clean[:8], (1, 8)), y=np.array([clean[0]]))

x = np.empty(0)

# Run train()
while len(clean_files) > 0 and len(noisy_files) > 0:
    print(datetime.now(), len(clean_files), len(noisy_files), flush=True)

    clean, clean_files = get_samples(clean_files)
    noisy, noisy_files = get_samples(noisy_files)

    if clean.shape[0] != nsamples or noisy.shape[0] != nsamples:
        break

    x = np.append(x, ulaw(clean))
    break

x = np.reshape(x, (-1, nsamples, 1))
y = keras.utils.to_categorical(y=x, num_classes=256)
print('x=', x)
print('y=', y)

model.fit(x=x, y=y, batch_size=32, epochs=10)

print(list(x[0:1].flatten().astype(int))[:100])

z = np.argmax(model.predict(x[0:1])[0], axis=1)
z = z + np.where(z > 127, -256, 0)
print(list(z)[:100])
# print(x[1:2], np.argmax(model.predict(x[1:2])))
# print(x[2:3], np.argmax(model.predict(x[2:3])))
# print(x[3:4], np.argmax(model.predict(x[3:4])))
