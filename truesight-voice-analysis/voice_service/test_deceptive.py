"""Test the deceptive sample"""
import urllib.request
import json

with open("sample_audio/deceptive.wav", "rb") as f:
    audio_bytes = f.read()

boundary = "FormBoundary12345"
disp = b"Content-Disposition: form-data; name=\"audio\"; filename=\"deceptive.wav\"\r\n"
body = (
    ("--" + boundary + "\r\n").encode()
    + disp
    + b"Content-Type: audio/wav\r\n"
    + b"\r\n"
    + audio_bytes
    + b"\r\n"
    + ("--" + boundary + "--\r\n").encode()
)

req = urllib.request.Request(
    "http://127.0.0.1:8001/analyze",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    method="POST",
)
r = urllib.request.urlopen(req)
result = json.loads(r.read())
print("Deceptive sample result:")
print(json.dumps(result, indent=2))
