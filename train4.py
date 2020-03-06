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

batch_size = 128
initial_epoch = 0
sample_rate = 8000
num_classes = 256
num_samples = 8192

model = None

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

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
                yield ulaw(x * next(rand) * neg)

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
    x = np.array(list(itertools.islice(it, batch_size*100)))
    np.random.RandomState(0).shuffle(x)
    x = x[:batch_size*25, :]
    x = x + 128
    x = x.reshape((batch_size*25, num_samples, 1))
    return x, x

# Create a keras model
def get_model():
    layers = 2
    units = 6
    force_trainable = True

    x = Input(shape=(num_samples, 1))
    y = x

    def block(y, i, s, trainable):
        s += str(i)
        u = 2**(units+i)
        trainable = (trainable or force_trainable)

        y = Conv1D(u, 15, padding='same', activation='relu', name=s+'_conv1d_1', trainable=trainable)(y)
        y = BatchNormalization(name=s+'_norm1', trainable=trainable)(y)
        y = Conv1D(u, 15, padding='same', activation='relu', name=s+'_conv1d_2', trainable=trainable)(y)
        y = BatchNormalization(name=s+'_norm2', trainable=trainable)(y)
        return Dropout(0.1)(y)

    # Encoder
    for i in range(layers):
        y = block(y, i, 'enc', i == layers - 1)
        y = MaxPooling1D(4)(y)

    y = block(y, layers, 'enc'+str(layers+100)+'_', True)

    # Decoder
    for i in reversed(range(layers)):
        y = UpSampling1D(4)(y)
        y = block(y, i, 'dec', i == layers - 1)

    trainable = (layers == 1 or force_trainable)
    y = Dense(256, activation='relu', name='final_relu1', trainable=trainable)(y)
    y = Dense(256, activation='relu', name='final_relu2', trainable=trainable)(y)
    y = Dense(256, activation='softmax', dtype='float32', name='softmax_final', trainable=trainable)(y)

    return Model(inputs=x, outputs=y)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model()

model.compile(keras.optimizers.Adam(0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.save('./model/saved_model')
model.summary()

if len(sys.argv) == 2:
    path = sys.argv[1]
    model.load_weights(path, by_name=True)
    initial_epoch = int(re.match('.*/weights-(\d+).h5', path).group(1))

cp1 = ModelCheckpoint(filepath='./model/weights-{epoch:04d}.h5',
                      verbose=0,
                      save_best_only=False,
                      save_weights_only=True,
                      save_freq=3)

cp2 = CSVLogger('training.log', append=True)
cp3 = ReduceLROnPlateau(patience=50)

model.fit(x=sample_generator(initial_epoch),
          steps_per_epoch=100,
          initial_epoch=initial_epoch,
          validation_data=get_validation_data(),
          epochs=135000,
          callbacks=[cp1, cp2, cp3])
