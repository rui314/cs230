#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import re
import sys
import random
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Dropout, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 4
initial_epoch = 0

# We assume clean samples are 1-channel 16kHz
def generator():
    files = [str(path) for path in Path('LibriSpeech/train-other-500').glob('**/*.flac')]
    random.shuffle(files)

    sample_size = 16000 * 3
    total = batch_size * sample_size
    num_classes = 256

    data = np.empty(0)
    count = 0

    while True:
        file = files[count % len(files)]
        samples, samplerate = sf.read(file)
        assert samplerate == 16000
        data = np.append(data, samples)

        if len(data) >= total:
            x = ulaw(data[:total])
            data = data[total:]

            y = np.concatenate([x[1:], [0]])
            y = keras.utils.to_categorical(y=y, num_classes=num_classes)

            x = np.reshape(x, (batch_size, sample_size, 1))
            y = np.reshape(y, (batch_size, sample_size, num_classes))
            yield x, y

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
#
# f(x) = sgn(x) * ln(u * |x| + 1) / ln(u + 1) where u = 255
def ulaw(xs):
    u = 255
    xs = np.sign(xs) * np.log(u * np.abs(xs) + 1) / np.log(u + 1)
    xs = np.rint(xs * 256)
    xs = np.clip(xs, -128, 127)
    return np.rint(xs).astype(int)

# f^{-1}(y) = sgn(y) / u * ((1 + u)^|y| - 1) where u = 255
def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) / u * ((1 + u) ** np.abs(ys) - 1)

# Create a keras model
def get_model():
    channels = 256
    num_layers = 13

    inputs = Input(shape=(None, 1))
    x = Conv1D(filters=channels, kernel_size=1)(inputs)
    skip_connections = []

    for i in range(num_layers):
        res = x
        x = Conv1D(filters=channels, kernel_size=2, padding='causal', dilation_rate=2**i, activation='linear')(x)
        x1 = Conv1D(filters=channels, kernel_size=1, activation='tanh')(x)
        x2 = Conv1D(filters=channels, kernel_size=1, activation='sigmoid')(x)
        x = x1 * x2
        skip_connections.append(Dropout(0.05)(x))
        x = x + res
        x = Dropout(0.05)(x)

    x = Add()(skip_connections)
    x = Activation('relu')(x)
    x = Conv1D(filters=channels, kernel_size=1, activation='relu')(x)
    x = Dense(channels)(x)
    x = Dense(channels, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = None
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

if len(sys.argv) == 2:
    path = sys.argv[1]
    model.load_weights(path)
    initial_epoch = int(re.match('.*/model-(\d+).hdf5', path).group(1))

checkpoint = ModelCheckpoint(filepath='./saved_model/model-{epoch:03d}.hdf5',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             period=1)

model.save('./saved_model/saved_model', save_format='tf')

model.fit(x=generator(),
          steps_per_epoch=100,
          initial_epoch=initial_epoch,
          epochs=100000,
          callbacks=[checkpoint])
