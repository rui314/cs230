#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# We assume clean samples are 1-channel 16kHz
def get_sample():
    files = [str(path) for path in Path('LibriSpeech/dev-clean').glob('**/*.flac')]
    data = []
    limit = 16000 * 60 * 60
    total = 0
    while total < limit:
        file = files.pop()
        x, samplerate = sf.read(file)
        assert samplerate == 16000
        total += len(x)
        data.append(x)

    data = np.concatenate(data)
    return data[:limit]

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
    channels = 256
    num_layers = 14

    inputs = Input(shape=(None, 1))
    x = Conv1D(filters=channels, kernel_size=1)(inputs)

    for layer in range(num_layers):
        res = x
        x = Conv1D(filters=channels, kernel_size=2, padding='causal', dilation_rate=2**layer, activation='linear')(x)
        x1 = Conv1D(filters=channels, kernel_size=1, activation='tanh')(x)
        x2 = Conv1D(filters=channels, kernel_size=1, activation='sigmoid')(x)
        x = x1 * x2 + res
        x = Dropout(0.05)(x)

    x = Dense(256, activation='tanh')(x)
    x = Dense(256, activation='tanh')(x)
    x = Dense(256, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

x = ulaw(get_sample())
y = np.concatenate([[0], x[:-1]])

num_samples = 16000 * 30

x = np.reshape(x, (num_samples, -1, 1))
y = np.reshape(y, (num_samples, -1, 1))

y = keras.utils.to_categorical(y=y, num_classes=256)

checkpoint = ModelCheckpoint(filepath='./saved_model/model-{epoch:02d}.hdf5',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

model.fit(x=x, y=y, batch_size=32, epochs=10, callbacks=[checkpoint])
