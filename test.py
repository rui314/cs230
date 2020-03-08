#!/usr/bin/python3
import numpy as np
import soundfile as sf
import os
import glob

for file in glob.glob('/home/ruiu/rnnoise_contributions/*.wav'):
    x, _ = sf.read(file)
    print('%.5f %s' % (np.average(np.abs(x)), file))
