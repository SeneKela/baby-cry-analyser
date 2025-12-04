import requests
import numpy as np
import soundfile as sf

# Create a stereo test audio file
sr = 16000
duration = 3
# Create stereo audio (2 channels)
audio_left = np.random.uniform(-0.5, 0.5, sr * duration).astype(np.float32)
audio_right = np.random.uniform(-0.5, 0.5, sr * duration).astype(np.float32)
audio_stereo = np.stack([audio_left, audio_right], axis=1)

test_file = 'test_stereo.wav'
sf.write(test_file, audio_stereo, sr)

print(f"Created stereo test audio file: {test_file}")
print(f"Shape: {audio_stereo.shape}")

# Test the API
url = 'http://localhost:8000/analyze'
with open(test_file, 'rb') as f:
    files = {'file': (test_file, f, 'audio/wav')}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✅ Stereo audio handled successfully!")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"❌ Error: {response.text}")
