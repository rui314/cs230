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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LambdaCallback
from tensorflow.keras.mixed_precision import experimental as mixed_precision

batch_size = 64
initial_epoch = 0
sample_rate = 16000
num_classes = 256
num_samples = 16384

model = None

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

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
    it = read_audio('train-16k.raw', initial_epoch)

    while True:
        x = np.array(list(itertools.islice(it, batch_size)))
        x = x.reshape((batch_size, num_samples, 1))
        yield x, x

def get_validation_data():
    it = read_audio('validate-16k.raw', 0)
    x = np.array(list(itertools.islice(it, batch_size*10)))
    x = x.reshape((-1, num_samples, 1))
    return x, x

# Create a keras model
def get_model():
    x = Input(shape=(None, 1))
    y = x

    def layer(y, dilation):
        y = Conv1D(64, 3, dilation_rate=2**dilation, padding='same')(y)
        y = LeakyReLU()(y)
        y = Conv1D(64, 3, dilation_rate=2**dilation, padding='same')(y)
        y = LeakyReLU()(y)
        return BatchNormalization()(y)

    for i in range(13):
        y = layer(y, i)
    y = layer(y, 0)
    y = Conv1D(1, 1)(y)

    return Model(inputs=x, outputs=y)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

model.compile(keras.optimizers.Adam(0.00001), loss='mse', metrics=['accuracy'])
model.summary()

if len(sys.argv) == 2:
    path = sys.argv[1]
    model.load_weights(path, by_name=True)
    initial_epoch = int(re.match('.*/weights-(\d+).h5', path).group(1))

cp1 = ModelCheckpoint(filepath='./model/weights-{epoch:04d}.h5',
                      verbose=0,
                      save_best_only=False,
                      save_weights_only=False,
                      save_freq=1)

cp2 = CSVLogger('training.log', append=True)

model.fit(x=sample_generator(initial_epoch),
          steps_per_epoch=200,
          initial_epoch=initial_epoch,
          validation_data=get_validation_data(),
          epochs=135000,
          callbacks=[cp1, cp2])
