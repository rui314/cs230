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

num_samples = 8192

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

xs = []
ys = []

for i in range(30):
    x, _ = sf.read('validate-8k.raw', format='RAW', subtype='PCM_16', samplerate=8000, channels=1,
                   start=8000*500+num_samples*i, frames=num_samples)

    x = ulaw(x)
    xs.append(ulaw_reverse(x))
    print('x', x[1000:1050].tolist())
    x = x + 128
    x = x.reshape(1, num_samples)

    y = model.predict(x)[0]
    y = np.argmax(y, axis=-1) - 128
    print('y', y[1000:1050].tolist())
    print()
    y = ulaw_reverse(y)
    ys.append(y)

sf.write(b'out.wav', np.concatenate(ys), 8000)
sf.write(b'out2.wav', np.concatenate(xs), 8000)
