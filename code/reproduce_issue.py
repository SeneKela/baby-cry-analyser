import numpy as np
import soundfile as sf
import torch
from inference import predict_audio, waveform_to_mel, load_wav

# Create a dummy wav file (5 seconds of white noise)
sr = 16000
duration = 5
audio = np.random.uniform(-1, 1, sr * duration)
sf.write('dummy.wav', audio, sr)

print("Created dummy.wav")

# Run inference
try:
    pred, conf, probs = predict_audio('dummy.wav')
    print(f"Prediction: {pred}, Confidence: {conf}")
    
    # Check mel spec shape
    waveform = load_wav('dummy.wav')
    mel_spec = waveform_to_mel(waveform)
    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    
except Exception as e:
    print(f"Inference failed: {e}")
