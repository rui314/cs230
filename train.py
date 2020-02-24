#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Dropout
from tensorflow.keras.models import Model

sample_size = 16000 * 5

# Get files for training
clean_files = [str(path) for path in Path('LibriSpeech/dev-clean').glob('**/*.flac')]
noisy_files = [str(path) for path in Path('rnnoise_contributions').glob('*.wav')]

# We assume clean samples are 1-channel 16kHz
# This function returns a 5 second of clean audio.
def get_samples(files):
    data = np.empty(0)
    while len(files) > 0 and data.shape[0] < sample_size:
        file = files.pop()
        data2, samplerate = sf.read(file)
        assert samplerate == 16000
        data = np.append(data, data2)
    return data[:sample_size], files

# https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
#
# f(x) = sgn(x) * ln(u * |x| + 1) / ln(u + 1) where u = 255
def ulaw(xs):
    u = 255
    xs = np.sign(xs) * np.log(u * np.absolute(xs) + 1) / np.log(u + 1)
    xs = np.rint(xs * 256)
    xs = np.clip(xs, -128, 127)
    return np.rint(xs).astype(int)

# f^{-1}(y) = sgn(y) * (1 / u) * ((1 + u)^|y| - 1) where u = 255
def ulaw_reverse(ys):
    u = 255
    ys = ys.astype(float) / 256
    return np.sign(ys) * (1 / u) * (np.power(1 + u, np.absolute(ys)) - 1)

# Create a keras model
def get_model():
    channels = 256

    inputs = Input(shape=(sample_size,1))
    f = Conv1D(filters=channels, kernel_size=1)(inputs)

    for _ in range(2): # number of stacks
        skip_connections = []
        for layer in range(10): # number of layers
            f = Dropout(.05)(f)
            res = f
            f = Conv1D(filters=channels, kernel_size=2, padding='same', dilation_rate=2**layer, activation='linear')(f)
            f1 = Conv1D(filters=channels, kernel_size=1, activation='tanh')(f)
            f2 = Conv1D(filters=channels, kernel_size=1, activation='sigmoid')(f)
            f = f1 * f2
            skip_connections.append(f)
            f = f + res

        skip_connections.append(f)            
        f = Add()(skip_connections)
        f = Conv1D(filters=256, kernel_size=1, activation='tanh')(f)

    f = Conv1D(filters=256, kernel_size=1, activation='tanh')(f)
    f = Conv1D(filters=256, kernel_size=1, activation='softmax')(f)

    model = Model(inputs=inputs, outputs=f)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model()
x = []
y = []

i = 0
while len(clean_files) > 0 and len(noisy_files) > 0:
    print(datetime.now(), len(clean_files), len(noisy_files), flush=True)
    clean, clean_files = get_samples(clean_files)
    noisy, noisy_files = get_samples(noisy_files)
    if clean.shape[0] != sample_size or noisy.shape[0] != sample_size:
        break
    x.append(ulaw(np.clip(clean + noisy, -1, 1)))
    y.append(ulaw(clean))
    i += 1
    if i == 500:
        break

x = np.concatenate(x)
y = np.concatenate(y)

if sys.argv[1] == 'train':
    # x = x[:x.shape[0] // 2]
    # y = y[:y.shape[0] // 2]

    x = np.reshape(x, (-1, sample_size, 1))
    y = np.reshape(y, (-1, sample_size, 1))

    y = keras.utils.to_categorical(y=y, num_classes=256)

    model.fit(x=x, y=y, batch_size=1, epochs=100)
    model.save('./saved_model/my_model')
    exit(0)

if sys.argv[1] == 'gen':
    model = tf.keras.models.load_model('saved_model/my_model')

    x = x[x.shape[0] // 2:]
    y = y[y.shape[0] // 2:]

    for i in range(50):
        x2 = x[i * sample_size : (i + 1) * sample_size]
        y2 = y[i * sample_size : (i + 1) * sample_size]

        x3 = np.reshape(x2, (-1, sample_size, 1))
        z = model.predict(x3)
        z = np.argmax(z[0], axis=1)
        z = z + np.where(z > 127, -256, 0)

        sf.write(f'out/out-mixed{i}.wav', ulaw_reverse(x2), 16000)
        sf.write(f'out/out-clean{i}.wav', ulaw_reverse(y2), 16000)
        sf.write(f'out/out-denoised{i}.wav', ulaw_reverse(z), 16000)

    exit(0)

print('argv[1] must be either "train" or "gen"')
exit(1)
