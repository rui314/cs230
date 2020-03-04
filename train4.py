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
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Conv1D, MaxPooling2D, UpSampling1D, UpSampling2D, Concatenate, Reshape, Dropout, LSTM, Activation, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 128
initial_epoch = 0
sample_rate = 8000
num_classes = 256
num_samples = 4096

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
#
# f(x) = sgn(x) * ln(u * |x| + 1) / ln(u + 1) where u = 255
def ulaw(xs):
    u = 255
    xs = np.sign(xs) * np.log(u * np.abs(xs) + 1) / np.log(u + 1)
    xs = np.rint(xs * 256)
    xs = np.clip(xs, -128, 127)
    return np.rint(xs).astype(int)

def read_audio(initial_epoch):
    filename = 'train-8k.raw'
    num_frames = os.path.getsize(filename) // 2
    perm = np.random.RandomState(0).permutation(num_frames // num_samples)

    while True:
        for i in perm:
            x, _ = sf.read(filename, format='RAW', subtype='PCM_16', samplerate=sample_rate, channels=1,
                           start=num_samples*i, frames=num_samples)
            yield ulaw(x)

# We assume clean samples are 1-channel 16kHz
def sample_generator(initial_epoch):
    filename = 'train-8k.raw'
    num_frames = os.path.getsize(filename) // 2
    it = read_audio(initial_epoch)
    
    while True:
        x = np.array(list(itertools.islice(it, batch_size)))
        y = keras.utils.to_categorical(x, num_classes)

        x = x.reshape((batch_size, num_samples))
        y = y.reshape((batch_size, num_samples, num_classes))
        yield x, y

# Create a keras model
def get_model():
    layers = 2
    units = 5

    x = Input(shape=(num_samples))
    y = Reshape((num_samples, 1))(x)

    # Encoder
    for i in range(layers):
        s = '_enc1_' + str(i)
        y = Conv1D(2**(units+i), 15, strides=4, padding='same', name='conv1d'+s)(y)
        y = Add()([Activation('tanh', name='tanh'+s)(y),
                   Activation('sigmoid', name='sigmoid'+s)(y)])
        y = BatchNormalization(name='norm'+s)(y)
        y = Dropout(0.05)(y)

        s = '_enc2_' + str(i)
        y = Conv1D(2**(units+i), 15, padding='same', name='conv1d'+s)(y)
        y = Add()([Activation('tanh', name='tanh'+s)(y),
                   Activation('sigmoid', name='sigmoid'+s)(y)])
        y = BatchNormalization(name='norm'+s)(y)
        y = Dropout(0.05)(y)

    # Decoder
    for i in reversed(range(layers)):
        y = UpSampling1D(4)(y)

        s = '_dec1_' + str(i)
        y = Conv1D(2**(units+i-1), 15, padding='same', name='conv1d'+s)(y)
        y = Add()([Activation('tanh', name='tanh'+s)(y),
                   Activation('sigmoid', name='sigmoid'+s)(y)])
        y = BatchNormalization(name='norm'+s)(y)
        y = Dropout(0.05)(y)

        s = '_dec2_' + str(i)
        y = Conv1D(2**(units+i-1), 15, padding='same', name='conv1d'+s)(y)
        y = Add()([Activation('tanh', name='tanh'+s)(y),
                   Activation('sigmoid', name='sigmoid'+s)(y)])
        y = BatchNormalization(name='norm'+s)(y)
        y = Dropout(0.05)(y)

    y = Dense(256, activation='softmax', name='final')(y)

    model = Model(inputs=x, outputs=y)
    model.compile(keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = None
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

if len(sys.argv) == 2:
    path = sys.argv[1]
    model.load_weights(path)
    initial_epoch = int(re.match('.*/weights-(\d+).hdf5', path).group(1))

checkpoint = ModelCheckpoint(filepath='./model/weights-{epoch:04d}.hdf5',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=True,
                             period=1)

model.save('./model/saved_model', save_format='tf')

model.fit(x=sample_generator(initial_epoch),
          steps_per_epoch=100,
          initial_epoch=initial_epoch,
          epochs=135000,
          callbacks=[checkpoint])
