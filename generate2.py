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
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

model = keras.models.load_model('./model/saved_model')
model.load_weights(sys.argv[1])

frames = 16000 * 40
audio, _ = sf.read('train.raw', format='RAW', subtype='PCM_16', samplerate=16000, channels=1,
                   start=16000*30, frames=frames)

x = librosa.core.stft(audio)
x = x[:1024, :1024]
x = np.abs(x).T
x = (np.clip(np.log(x+0.0000000001), -5, 5) + 5) / 10
x = np.reshape(x, (1, 1024, 1024))

y = model.predict(x, verbose=1)[0]
y = np.argmax(y, axis=-1)
y = np.clip(np.exp(y * 10 - 5), -5, 5)

z = np.zeros((1024, 1025))
z[:, :1024] = y
z = z.T

# z = librosa.core.griffinlim(z, n_iter=1024)
z = librosa.core.griffinlim(z)
sf.write(b'out.wav', z, 16000)
