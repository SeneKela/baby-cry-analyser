# inference.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio

#CONFIG
SAMPLE_RATE = 16000
MAX_LEN = 500  # Matches train.py (500 time frames)
N_CLASSES = 5
CLASS_NAMES = ["hungry", "sleepy", "diaper", "pain", "discomfort"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cry_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HELPER FUNCTION: PAD/TRUNCATE (from train.py)
def pad_truncate(mel_spec, max_len=MAX_LEN):
    if mel_spec.shape[0] < max_len:
        pad_amount = max_len - mel_spec.shape[0]
        mel_spec = F.pad(mel_spec, (0, 0, 0, pad_amount))
    else:
        mel_spec = mel_spec[:max_len, :]
    return mel_spec

#WAV LOADER
def load_wav(path, target_sr=SAMPLE_RATE):
    try:
        # Try soundfile first (handles WAV, FLAC, OGG)
        waveform, sr = sf.read(path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
    except Exception as e:
        # Fall back to torchaudio for other formats (MP3, WebM, etc.)
        try:
            # For WebM and other formats, try using ffmpeg backend
            waveform, sr = torchaudio.load(path, backend="ffmpeg")
        except:
            # If ffmpeg not available, try default backend
            waveform, sr = torchaudio.load(path)
    
    # Convert stereo to mono if needed
    if waveform.ndim == 2:
        if waveform.shape[0] == 2:  # (channels, samples)
            waveform = waveform.mean(dim=0, keepdim=True)  # Average channels
        elif waveform.shape[1] == 2:  # (samples, channels)
            waveform = waveform.mean(dim=1).unsqueeze(0)  # Average channels
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # shape (1, N)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        
    return waveform

# MODEL
class CryModel(nn.Module):
    def __init__(self):
        super(CryModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, seq)
        x = self.cnn(x)
        x = x.mean(dim=2)
        return self.fc(x)

# MEL SPECTOGRAM
def waveform_to_mel(waveform):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=64
    )
    mel_spec = mel_transform(waveform)
    # mel_spec shape is (1, n_mels, time)
    mel_spec = mel_spec.squeeze(0).transpose(0, 1) # (time, n_mels)
    mel_spec = pad_truncate(mel_spec, MAX_LEN)
    return mel_spec

#LOAD MODEL
model = CryModel().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

#PREDICTION FUNCTION
def predict_audio(file_path):
    waveform = load_wav(file_path)
    mel_spec = waveform_to_mel(waveform)
    mel_spec = mel_spec.unsqueeze(0).to(DEVICE)  # batch dim (1, time, mel)
    
    with torch.no_grad():
        logits = model(mel_spec)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    pred_class = CLASS_NAMES[idx.item()]
    return pred_class, conf.item(), probs.cpu().numpy()
