#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import random
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

sample_size = 16384

# f^{-1}(y) = sgn(y) * (1 / u) * ((1 + u)^|y| - 1) where u = 255
def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) * (1 / u) * (np.power(1 + u, np.absolute(ys)) - 1)

def choose(dist):
    for i in range(len(dist)):
        if random.uniform(0, 1) < dist[i]:
            return i
    return np.argmax(dist)

model = tf.keras.models.load_model(sys.argv[1])
size = 16000 * 10
ys = np.zeros(size)

x = np.round(np.random.rand(1, sample_size, 1) * 256) - 128

for i in range(size):
    preds = model.predict(x, verbose=0)[0]
    y = choose(preds[-1])
    if y > 127:
        y = y - 256

    x[0, 0:(sample_size-1), 0] = x[0, 1:, 0]
    x[0, -1, 0] = y
    ys[i] = y

    print(y, flush=True)

    if i % 100 == 0:
        print(i, flush=True)
        sf.write(f'out.wav', ulaw_reverse(ys), 16000)
