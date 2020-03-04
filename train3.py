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
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Conv1D, MaxPooling2D, UpSampling2D, Concatenate, Reshape, Dropout, LSTM, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 16
initial_epoch = 0

# We assume clean samples are 1-channel 16kHz
def sample_generator(initial_epoch):
    filename = 'train.raw'
    num_frames = os.path.getsize(filename) // 2

    frames = 16000 * 5
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
            x = np.abs(x).T
            x = (np.clip(np.log(x+0.0000000001), -5, 5) + 5) / 10
            x = x[:64, :1024]

            y = keras.utils.to_categorical(np.reshape(x, (64, 1024, 1)) * (num_classes - 1), num_classes)

            xs.append(x)
            ys.append(y)

        yield np.array(xs), np.array(ys)

# Create a keras model
def get_model():
    x = Input(shape=(None, 1024))
    y = x

    # Encoder-decoder
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.05)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.05)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.05)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.05)(y)

    y = Dense(256, activation='relu')(y)
    y = Dropout(0.05)(y)

    y = Dense(512, activation='relu')(y)
    y = Dropout(0.05)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.05)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.05)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.05)(y)

    y = Dense(1024*16, activation='relu')(y)
    y = Reshape((-1, 1024, 16))(y)
    y = Activation('softmax')(y)

    model = Model(inputs=x, outputs=y)
    model.compile(keras.optimizers.Adam(.001), loss='categorical_crossentropy', metrics=['accuracy'])
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
          steps_per_epoch=400,
          initial_epoch=initial_epoch,
          epochs=135000,
          callbacks=[checkpoint])
