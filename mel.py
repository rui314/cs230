#!/usr/bin/python3
import librosa
import soundfile

y, sr = librosa.core.load('LibriSpeech/dev-clean/6241/66616/6241-66616-0010.flac')
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
y2 = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, n_fft=512, n_iter=2048)
soundfile.write('out.wav', y2, sr)
