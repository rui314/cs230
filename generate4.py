#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import random
import soundfile as sf
import tensorflow as tf
import librosa
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from pathlib import Path

num_samples = 4096

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

model = keras.models.load_model('./model/saved_model')
model.load_weights(sys.argv[1])

ys = []

for i in range(10):
    x, _ = sf.read('train-8k.raw', format='RAW', subtype='PCM_16', samplerate=8000, channels=1,
                   start=8000*300+num_samples*i, frames=num_samples)
    print('x', x)
    x = ulaw(x).reshape(1, num_samples)

    y = model.predict(x)[0]
    y = np.argmax(y, axis=-1)
    y = y - np.where(y > 127, 256, 0)
    y = ulaw_reverse(y)
    print('y', y)
    ys.append(y)

sf.write(b'out.wav', np.concatenate(ys), 8000)
