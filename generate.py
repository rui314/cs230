#!/usr/bin/python3
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Input, Add, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

model = tf.keras.models.load_model('./saved_model/model-50.hdf5')
audio = []

x = np.zeros((1, 16000, 1))
preds = model.predict(x_pred, verbose=0)[0]
print(preds)
