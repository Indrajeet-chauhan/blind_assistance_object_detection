import os
import time
import threading
from pathlib import Path

import cv2

from flask import Flask, Response, jsonify, render_template_string

from detector import ObjectDetector

app = Flask(__name__)
detector = ObjectDetector()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SPEECH_COOLDOWN = float(os.getenv("SPEECH_COOLDOWN", "3.0"))

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind Assistance — Object Detection</title>
<style>
:root { color-scheme: dark; }
body { font-family: system-ui, Arial, sans-serif; margin:0; background:#101827; color:#f7fafc; }
main { max-width:900px; margin:auto; padding:22px; }
h1 { margin-bottom:6px; }
p { line-height:1.5; color:#cbd5e1; }
.panel { background:#1e293b; border-radius:16px; padding:16px; margin:16px 0; }
img { display:block; width:100%; border-radius:12px; background:#000; }
button { border:0; border-radius:10px; padding:12px 16px; font-size:1rem; font-weight:700; cursor:pointer; margin:4px; }
.primary { background:#38bdf8; color:#082f49; }
.secondary { background:#cbd5e1; color:#0f172a; }
#status { font-weight:700; }
li { margin:8px 0; }
.warning { border-left:4px solid #fbbf24; padding-left:12px; }
:focus-visible { outline:3px solid #fbbf24; outline-offset:3px; }
</style>
</head>
<body>
<main>
<h1>Object Detection for Blind Assistance</h1>
<p>Camera-based prototype using computer vision and spoken object labels.</p>
<div class="panel">
  <h2>Live camera</h2>
  <img src="/video_feed" alt="Live camera feed with detected objects outlined">
  <p id="status" role="status" aria-live="polite">Camera feed starting…</p>
  <button class="primary" onclick="speakLatest()">Speak detected objects</button>
  <button class="secondary" onclick="toggleSpeech()" id="speechButton">Pause automatic speech</button>
  <button class="secondary" onclick="location.reload()">Restart camera view</button>
</div>
<div class="panel">
  <h2>Detected objects</h2>
  <ul id="objects"><li>Waiting for detections…</li></ul>
</div>
<div class="panel warning">
  <h2>Safety notice</h2>
  <p>This is an educational prototype, not a certified mobility aid. It can miss objects, misclassify them, and cannot reliably determine safe paths or distances. Do not rely on it to cross roads, avoid stairs, or navigate independently. Use a cane/guide dog and trained assistance as appropriate.</p>
</div>
</main>
<script>
let autoSpeech = true;
let lastSpoken = "";
async function update() {
  try {
    const response = await fetch('/detections');
    const data = await response.json();
    document.getElementById('status').textContent = data.camera_ok ? 'Camera active' : 'Camera unavailable — check camera permissions';
    const ul = document.getElementById('objects');
    ul.replaceChildren();
    if (!data.objects.length) {
      const li = document.createElement('li'); li.textContent = 'No objects detected right now.'; ul.appendChild(li);
      return;
    }
    for (const obj of data.objects) {
      const li = document.createElement('li');
      li.textContent = `${obj.label} — ${Math.round(obj.confidence * 100)}% confidence`;
      ul.appendChild(li);
    }
    const summary = data.objects.map(o => o.label).join(', ');
    if (autoSpeech && summary && summary !== lastSpoken && 'speechSynthesis' in window) {
      lastSpoken = summary;
      const utterance = new SpeechSynthesisUtterance('Objects detected: ' + summary);
      utterance.rate = 0.92;
      speechSynthesis.cancel();
      speechSynthesis.speak(utterance);
    }
  } catch (e) {
    document.getElementById('status').textContent = 'Could not read detections.';
  }
}
function speakLatest() {
  if (!('speechSynthesis' in window)) { alert('Text-to-speech is not supported in this browser.'); return; }
  fetch('/detections').then(r => r.json()).then(data => {
    const msg = data.objects.length ? 'Objects detected: ' + data.objects.map(o => o.label).join(', ') : 'No objects detected right now.';
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(msg));
  });
}
function toggleSpeech() {
  autoSpeech = !autoSpeech;
  document.getElementById('speechButton').textContent = autoSpeech ? 'Pause automatic speech' : 'Resume automatic speech';
  if (!autoSpeech && 'speechSynthesis' in window) speechSynthesis.cancel();
}
setInterval(update, 1800);
update();
</script>
</body>
</html>
"""

class CameraService:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.objects = []
        self.camera_ok = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        while self.running:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok:
                with self.lock:
                    self.camera_ok = False
                    self.frame = None
                    self.objects = []
                time.sleep(0.4)
                # Retry camera opening in case it was temporarily unavailable.
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(CAMERA_INDEX)
                continue
            annotated, detections = detector.detect(frame)
            with self.lock:
                self.camera_ok = True
                self.frame = annotated
                self.objects = detections
            time.sleep(0.02)

    def get_snapshot(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            return frame, list(self.objects), self.camera_ok

    def stream(self):
        while self.running:
            frame, _, _ = self.get_snapshot()
            if frame is None:
                time.sleep(0.1)
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(0.04)

camera = CameraService()

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    return Response(camera.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    _, objects, camera_ok = camera.get_snapshot()
    return jsonify({"camera_ok": camera_ok, "objects": objects})

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.")
    print("Allow camera access if prompted. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
