import torchaudio
print(f"Torchaudio version: {torchaudio.__version__}")
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
print(f"n_fft: {mel.n_fft}")
print(f"hop_length: {mel.hop_length}")
