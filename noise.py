#!/usr/bin/python3
import glob
import soundfile as sf
import numpy as np

samplerate = 16000

xs = []

for file in glob.glob('rnnoise_contributions/**/*.wav', recursive=True):
    x, rate = sf.read(file)
    assert rate == 16000
    x = x / np.max(x) * 0.3
    print(file, np.max(x))
    xs.append(x)

sf.write(b'out.wav', np.concatenate(xs), samplerate)
