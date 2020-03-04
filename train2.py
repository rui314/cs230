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
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 8
initial_epoch = 0

# We assume clean samples are 1-channel 16kHz
def sample_generator(initial_epoch):
    filename = 'train.raw'
    num_frames = os.path.getsize(filename) // 2

    frames = 16000 * 40
    start = frames * initial_epoch
    num_classes = 16

    while True:
        xs = []
        ys = []

        for _ in range(batch_size):
            if num_frames <= start + frames:
                start = 0

            audio, _ = sf.read(filename, format='RAW', subtype='PCM_16', samplerate=16000, channels=1,
                               start=start, frames=frames)
            start += frames

            x = librosa.core.stft(audio)
            x = x[:1024, :1024]

            magnitude = np.abs(x)
            magnitude = np.clip(np.log(x.T), -5, 5) + 5

            x = magnitude
            y = keras.utils.to_categorical(np.round(magnitude / 5 * (num_classes - 1)), num_classes)

            xs.append(np.reshape(x, (1024, 1024, 1)))
            ys.append(y)

        yield np.array(xs), np.array(ys)

# Create a keras model
def get_model():
    kernel_size = 3

    x = Input(shape=(1024, 1024, 1))

    # Encoder-decoder
    y = x
    for i in range(6):
        y = Conv2D(2**(4+i), kernel_size, padding='same', activation='relu')(y)
        y = MaxPooling2D(2, padding='same')(y)
        y = Dropout(0.05)(y)

    for i in reversed(range(6)):
        y = Conv2DTranspose(2**(4+i), kernel_size, strides=2, padding='same', activation='relu')(y)
        y = Dropout(0.05)(y)

    y = Conv2D(16, kernel_size, padding='same', activation='softmax')(y)

    model = Model(inputs=x, outputs=y)
    model.compile('adadelta',
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
          steps_per_epoch=50,
          initial_epoch=initial_epoch,
          epochs=135000,
          callbacks=[checkpoint])
