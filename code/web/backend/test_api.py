import requests
import numpy as np
import soundfile as sf

# Create a test audio file
sr = 16000
duration = 3
audio = np.random.uniform(-0.5, 0.5, sr * duration).astype(np.float32)
test_file = 'test_audio.wav'
sf.write(test_file, audio, sr)

print(f"Created test audio file: {test_file}")

# Test the API
url = 'http://localhost:8000/analyze'
with open(test_file, 'rb') as f:
    files = {'file': (test_file, f, 'audio/wav')}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Report URL: {result['report_url']}")
else:
    print(f"Error: {response.text}")
