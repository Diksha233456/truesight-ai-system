"""Quick API test script"""
import urllib.request
import urllib.parse
import json

# --- Health check ---
print("Testing GET / ...")
r = urllib.request.urlopen("http://127.0.0.1:8001/")
health = json.loads(r.read())
print("Health:", health)
assert health["status"] == "running", "Health check failed!"
print("PASS: Health check")

# --- Analyze endpoint ---
print("\nTesting POST /analyze with sample_audio/truthful.wav ...")

with open("sample_audio/truthful.wav", "rb") as f:
    audio_bytes = f.read()

boundary = "FormBoundary12345"
crlf = b"\r\n"
body = (
    ("--" + boundary + "\r\n").encode()
    + b"Content-Disposition: form-data; name=\"audio\"; filename=\"truthful.wav\"\r\n"
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

try:
    r = urllib.request.urlopen(req)
    result = json.loads(r.read())
    print("Response:")
    print(json.dumps(result, indent=2))

    assert "verdict" in result
    assert "score" in result
    assert "confidence" in result
    assert "explanation" in result
    assert result["verdict"] in ("Truthful", "Deceptive")
    assert 0.0 <= result["score"] <= 1.0
    assert 2 <= len(result["explanation"]) <= 5

    print("\nPASS: /analyze returned valid response!")
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.read().decode())
