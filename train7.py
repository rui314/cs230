#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import re
import os
import sys
import soundfile as sf
import tensorflow as tf
import librosa
import itertools
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Conv1D, MaxPooling1D, MaxPooling2D, UpSampling1D, UpSampling2D, Concatenate, Reshape, Dropout, LSTM, Activation, BatchNormalization, Add, LeakyReLU, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LambdaCallback
from tensorflow.keras.mixed_precision import experimental as mixed_precision

batch_size = 4
initial_epoch = 0
sample_rate = 16000
num_classes = 256
num_samples = 16000

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
#
# f(x) = sgn(x) * ln(u * |x| + 1) / ln(u + 1) where u = 255
def ulaw(xs):
    u = 255
    xs = np.sign(xs) * np.log(u * np.abs(xs) + 1) / np.log(u + 1)
    xs = np.rint(xs * 256)
    xs = np.clip(xs, -128, 127)
    return np.rint(xs).astype(int)

def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) / u * ((1 + u) ** np.abs(ys) - 1)

def permutation(num_frames):
    skew = np.random.RandomState(0).permutation(num_samples)
    perm = np.random.RandomState(1).permutation(num_frames // num_samples - 1)

    for i in itertools.cycle(skew):
        for j in perm:
            yield i + j * num_samples

def read_audio(filename, initial_epoch):
    num_frames = os.path.getsize(filename) // 2

    skew = np.random.RandomState(0).permutation(num_samples // 100) * 100
    start = np.random.RandomState(1).permutation(num_frames // num_samples - 1) * num_samples
    rand = itertools.cycle(0.6 + np.random.RandomState(10).rand(10001) * 0.4) # for data augmentation

    for sk in itertools.cycle(skew):
        for neg in [1, -1]: # for data augmentation
            for st in start:
                x, _ = sf.read(filename, format='RAW', subtype='PCM_16', samplerate=sample_rate, channels=1,
                               start=st, frames=num_samples)
                yield x * next(rand) * neg

# We assume clean samples are 1-channel 8kHz
def sample_generator(initial_epoch):
    speech_gen = read_audio('train-16k.raw', initial_epoch)
    noise_gen = read_audio('noise-16k.raw', initial_epoch)

    i = 0
    while True:
        speech = np.array(list(itertools.islice(speech_gen, batch_size)))
        noise = np.array(list(itertools.islice(noise_gen, batch_size)))
        mixed = speech * 0.8 + noise * 0.2

        if i % 100 == 0:
            sf.write('x.wav', ulaw_reverse(ulaw(mixed.flatten())), sample_rate)
            sf.write('y.wav', ulaw_reverse(ulaw(speech.flatten() * 0.8)), sample_rate)
        i += 1

        x = (ulaw(mixed)+128).reshape((batch_size, num_samples, 1))
        y = (ulaw(speech * 0.8)+128).reshape((batch_size, num_samples, 1))
        yield x, y

# Create a keras model
def get_model():
    x = Input(shape=(None, 1))
    y = x

    kernel_size = 3
    residual_channel = 512
    skip_channel = 512
    dilation_depth = 8
    repeat = 2
    dropout = 0.05
    skip_connections = []

    y = Conv1D(residual_channel, 1)(y)

    for dilation_rate in [3**i for i in range(dilation_depth)] * repeat:
        res = y
        y1 = Conv1D(residual_channel, kernel_size, padding='same', dilation_rate=dilation_rate, activation='tanh')(y)
        y2 = Conv1D(residual_channel, kernel_size, padding='same', dilation_rate=dilation_rate, activation='sigmoid')(y)
        y = y1 * y2
        skip_connections.append(Conv1D(skip_channel, 1)(y))
        y = y + res
        y = Dropout(dropout)(y)

    y = Add()(skip_connections)
    y = Activation('relu')(y)
    y = Conv1D(skip_channel, 1, activation='relu')(y)
    y = Conv1D(num_classes, 1, activation='softmax')(y)

    return Model(inputs=x, outputs=y)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

model.compile(keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

if len(sys.argv) == 2:
    path = sys.argv[1]
    model.load_weights(path, by_name=True)
    initial_epoch = int(re.match('.*/weights-(\d+).h5', path).group(1))

cp1 = ModelCheckpoint(filepath='./model/weights-{epoch:04d}.h5',
                      verbose=0,
                      save_best_only=False,
                      save_weights_only=False,
                      save_freq=100)

cp2 = CSVLogger('training.log', append=True)
cp3 = ReduceLROnPlateau('loss', patience=30)

model.fit(x=sample_generator(initial_epoch),
          steps_per_epoch=100,
          initial_epoch=initial_epoch,
          epochs=135000,
          callbacks=[cp1, cp2, cp3])
