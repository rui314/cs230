#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import re
import os
import sys
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Multiply, Dropout, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 4
initial_epoch = 0
num_classes = 256

# We assume clean samples are 1-channel 16kHz
def sample_generator(initial_epoch):
    data, samplerate = sf.read('train.wav')
    assert samplerate == 16000
    
    sample_size = 16000 * 2
    total = batch_size * sample_size
    start = total * initial_epoch

    while True:
        end = start + total
        if len(data) <= end:
            start = 0
            end = total
            
        x = data[start:end]
        start += total

        x = ulaw(x)

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

# Create a keras model
def get_model():
    channels = 256
    num_layers = 15

    inputs = Input(shape=(None, 1), name='input')
    x = inputs
    skip_connections = []

    for i in range(num_layers):
        residual = x
        x1 = Conv1D(channels, 2, padding='causal', dilation_rate=2**i, activation='tanh', name='tanh_'+str(i))(x)
        x2 = Conv1D(channels, 2, padding='causal', dilation_rate=2**i, activation='sigmoid', name='sigmoid_'+str(i))(x)
        x = Multiply(name='layer_mul_'+str(i))([x1, x2])
        skip = Conv1D(channels, 1, activation='relu', name='skip_'+str(i))(x)
        skip_connections.append(Dropout(0.05, name='drop_'+str(i))(skip))
        x = Add(name='layer_out_'+str(i))([x, residual])
        x = Dropout(0.05, name='layer_out_drop_'+str(i))(x)

    x = Add(name='add_skip_connections')(skip_connections)
    x = Activation('relu')(x)
    x = Conv1D(channels, 1, activation='relu', name='final_conv1d')(x)
    x = Conv1D(num_classes, 1, activation='softmax', name='final_softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
          epochs=100000,
          callbacks=[checkpoint])
