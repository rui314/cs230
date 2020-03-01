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

# f^{-1}(y) = sgn(y) / u * ((1 + u)^|y| - 1) where u = 255
def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) / u * ((1 + u) ** np.abs(ys) - 1)

model = keras.models.load_model('./model/saved_model')
model.load_weights(sys.argv[1])

size = 16000 * 10
ys = np.zeros(size)

x = np.round(np.random.rand(1, sample_size, 1) * 256) - 128
x = np.clip(x, -128, 127)
# x = np.zeros((1, sample_size, 1))

for i in range(size):
    preds = model.predict(x, verbose=0)[0]
    y = np.random.choice(256, p=preds[-1])
    if y > 127:
        y = y - 256

    x[0, 0:(sample_size-1), 0] = x[0, 1:, 0]
    x[0, -1, 0] = y
    ys[i] = y

    if y >= 0:
        print('%s%s' % (' ' * 128, '*' * y), flush=True)
    else:
        print('%s%s' % (' ' * (128 + y), '*' * abs(y)), flush=True)

    if i % 100 == 0:
        print('*** ', i, flush=True)
        sf.write(f'out.wav', ulaw_reverse(ys), 16000)
