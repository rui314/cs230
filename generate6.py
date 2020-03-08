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

np.set_printoptions(precision=4, suppress=True, edgeitems=10, linewidth=300)

num_samples = 16384
rate = 16000

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


model = keras.models.load_model(sys.argv[1])

for i in range(10):
    sound, _ = sf.read('validate-clean-16k.raw', format='RAW', subtype='PCM_16', samplerate=rate, channels=1,
                       start=rate*i*60, frames=rate*5)

    noise, _ = sf.read('validate-noise-16k.raw', format='RAW', subtype='PCM_16', samplerate=rate, channels=1,
                       start=rate*i*60, frames=rate*5)

    mixed = sound * 0.8 + noise * 0.2

    x = ulaw(mixed) + 128

    y = model.predict(x.reshape(1, -1, 1))[0]
    y = np.argmax(y, axis=-1)

    print(x[4000:4050])
    print(y[4000:4050])
    print(np.abs((x - y)[4000:4050]))
    print()

    t = ulaw(sound * 0.8)
    print(t[4000:4050])
    print(y[4000:4050])
    print(np.abs((t - y)[4000:4050]))
    print()
    print()

    sf.write('generated-files/input-%d.wav' % i, ulaw_reverse(x-128), rate)
    sf.write('generated-files/output-%d.wav' % i, ulaw_reverse(y-128), rate)
