#!/usr/bin/python3
import librosa
import soundfile

y, sr = librosa.core.load('LibriSpeech/dev-clean/6241/66616/6241-66616-0010.flac')
spectrogram = librosa.core.stft(y=y, win_length=256, hop_length=64)
y2 = librosa.core.istft(spectrogram, win_length=256, hop_length=64)
print(y.shape)
print(spectrogram.shape)
print(y2.shape)
soundfile.write('out.wav', y2, sr)
