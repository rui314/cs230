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

num_samples = 16384
samplerate = 16000

model = keras.models.load_model(sys.argv[1])

xs = []
ys = []

for i in range(30):
    x, _ = sf.read('validate-16k.raw', format='RAW', subtype='PCM_16', samplerate=samplerate, channels=1,
                   start=samplerate*250+num_samples*i, frames=num_samples)

    print('x', x[4000:4010].tolist())
    x = x.reshape(1, num_samples)

    y = model.predict(x)[0].flatten()
    print('y', y[4000:4010].tolist())
    ys.append(y)

sf.write(b'out.wav', np.concatenate(ys), samplerate)
sf.write(b'out2.wav', np.concatenate(xs), samplerate)
