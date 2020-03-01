#!/usr/bin/python3
import librosa
import soundfile

y, sr = librosa.core.load('LibriSpeech/dev-clean/6241/66616/6241-66616-0010.flac')
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, win_length=256, hop_length=64)
y2 = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, n_fft=256, win_length=256, hop_length=64)
print(y.shape)
print(spectrogram.shape)
print(y2.shape)
soundfile.write('out.wav', y2, sr)
