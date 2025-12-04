import requests

test_file = 'test.webm'
url = 'http://localhost:8000/analyze'

print(f"Testing WebM file: {test_file}")

with open(test_file, 'rb') as f:
    files = {'file': (test_file, f, 'audio/webm')}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✅ WebM audio handled successfully!")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"❌ Error: {response.text}")
