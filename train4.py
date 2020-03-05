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
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Conv1D, MaxPooling1D, MaxPooling2D, UpSampling1D, UpSampling2D, Concatenate, Reshape, Dropout, LSTM, Activation, BatchNormalization, Add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

batch_size = 128
initial_epoch = 0
sample_rate = 8000
num_classes = 256
num_samples = 8192

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
#
# f(x) = sgn(x) * ln(u * |x| + 1) / ln(u + 1) where u = 255
def ulaw(xs):
    u = 255
    xs = np.sign(xs) * np.log(u * np.abs(xs) + 1) / np.log(u + 1)
    xs = np.rint(xs * 256)
    xs = np.clip(xs, -128, 127)
    return np.rint(xs).astype(int)

def permutation(num_frames):
    skew = np.random.RandomState(0).permutation(num_samples)
    perm = np.random.RandomState(1).permutation(num_frames // num_samples - 1)

    for i in itertools.cycle(skew):
        for j in itertools.cycle(perm):
            yield i + j * num_samples

def read_audio(filename, initial_epoch):
    num_frames = os.path.getsize(filename) // 2
    perm = permutation(num_frames)

    for _ in range(initial_epoch):
        next(perm)

    for i in perm:
        x, _ = sf.read(filename, format='RAW', subtype='PCM_16', samplerate=sample_rate, channels=1,
                       start=i, frames=num_samples)
        yield ulaw(x)

# We assume clean samples are 1-channel 8kHz
def sample_generator(initial_epoch):
    it = read_audio('train-8k.raw', initial_epoch)
    
    while True:
        x = np.array(list(itertools.islice(it, batch_size)))
        x = x + 128
        x = x.reshape((batch_size, num_samples, 1))
        yield x, x

def get_validation_data():
    it = read_audio('validate-8k.raw', 0)
    x = np.array(list(itertools.islice(it, batch_size)))
    x = x + 128
    x = x.reshape((batch_size, num_samples, 1))
    return x, x

# Create a keras model
def get_model():
    layers = 0
    units = 4

    x = Input(shape=(num_samples, 1))
    y = x

    def block(y, i, s):
        s += str(i)
        y = Conv1D(2**(units+i), 15, padding='same', activation='relu', name=s+'_conv1d_1')(y)
        y = Conv1D(2**(units+i), 15, padding='same', activation='relu', name=s+'_conv1d_2')(y)
        return BatchNormalization(name=s+'_norm')(y)

    # Encoder
    for i in range(layers):
        y = block(y, i, 'enc')
        y = MaxPooling1D(4)(y)

    y = block(y, layers, 'enc')

    # Decoder
    for i in reversed(range(layers)):
        y = UpSampling1D(4)(y)
        y = block(y, i, 'dec')

    y = Dense(256, activation='relu', name='final_relu')(y)
    y = Dense(256, activation='sigmoid', name='final_sigmoid')(y)
    y = Dense(256, activation='softmax', name='softmax_final')(y)

    return Model(inputs=x, outputs=y)

model = None
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

model.compile(keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()
model.save('./model/saved_model')

if len(sys.argv) == 2:
    path = sys.argv[1]
    model.load_weights(path, by_name=True)
    initial_epoch = int(re.match('.*/weights-(\d+).hdf5', path).group(1))

cp1 = ModelCheckpoint(filepath='./model/weights-{epoch:04d}.hdf5',
                      verbose=0,
                      validation_data=get_validation_data(),
                      save_best_only=False,
                      save_weights_only=True,
                      period=1)

cp2 = ReduceLROnPlateau('loss')
cp3 = CSVLogger('training.log')

model.fit(x=sample_generator(initial_epoch),
          steps_per_epoch=100,
          initial_epoch=initial_epoch,
          epochs=135000,
          callbacks=[cp1, cp2, cp3])