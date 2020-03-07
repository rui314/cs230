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

model = keras.models.load_model(sys.argv[1])

xs = []
ys = []

x, _ = sf.read('train-16k.raw', format='RAW', subtype='PCM_16', samplerate=rate, channels=1,
               start=rate*250, frames=rate*10)

y = model.predict(x.reshape(1, -1, 1))[0].flatten().astype('float32')
print(np.abs(x - y))

sf.write(b'out.wav', y, rate)
sf.write(b'out2.wav', x, rate)
