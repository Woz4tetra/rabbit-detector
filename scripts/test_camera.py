#!/usr/bin/env python3
"""Interactive camera tuner — keeps the camera open while you adjust settings live.

Run on the Pi:
    python scripts/test_camera.py [--width W] [--height H] [--port 7000]

Then open http://<pi-ip>:7000 in a browser. Sliders for every picamera2 control
update the camera instantly without restart. "Save frame" downloads the current frame.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

# ─── shared state ─────────────────────────────────────────────────────────────

_cam = None
_lock = threading.Lock()
_jpeg: bytes = b""
_stats: dict = {}

# ─── capture loop ─────────────────────────────────────────────────────────────

def capture_loop() -> None:
    global _jpeg, _stats
    while True:
        try:
            req = _cam.capture_request()
            frame = req.make_array("main")
            meta = req.get_metadata()
            req.release()

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue

            cg = meta.get("ColourGains", (1.0, 1.0))
            with _lock:
                _jpeg = buf.tobytes()
                _stats = {
                    "shape": f"{frame.shape[1]}x{frame.shape[0]}",
                    "mean": round(float(frame.mean()), 1),
                    "std": round(float(frame.std()), 1),
                    "exposure_us": int(meta.get("ExposureTime", 0)),
                    "gain": round(float(meta.get("AnalogueGain", 1.0)), 2),
                    "colour_gains": [round(float(cg[0]), 3), round(float(cg[1]), 3)],
                    "lux": round(float(meta.get("Lux", 0.0)), 1),
                }
        except Exception as exc:
            print(f"capture error: {exc}")
            time.sleep(0.5)

# ─── HTML ─────────────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Camera Tuner</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
h1 { color: #e94560; padding: 10px 16px; font-size: 1.2em; border-bottom: 1px solid #0f3460; flex-shrink: 0; }
.layout { display: flex; flex: 1; overflow: hidden; }
.feed-panel { flex: 1; display: flex; flex-direction: column; padding: 10px; gap: 8px; min-width: 0; overflow: hidden; }
#feed { max-width: 100%; max-height: calc(100vh - 100px); border-radius: 4px; object-fit: contain; display: block; background: #000; }
#stats { font-size: 0.75em; color: #888; padding: 5px 8px; background: #16213e; border-radius: 4px; flex-shrink: 0; }
.ctrl-panel { width: 300px; flex-shrink: 0; overflow-y: auto; background: #16213e; border-left: 1px solid #0f3460; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
.group { background: #1a1a2e; border-radius: 5px; padding: 9px 10px; }
.group h3 { font-size: 0.7em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 7px; }
.row { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
.row:last-child { margin-bottom: 0; }
.row label { width: 90px; font-size: 0.78em; color: #bbb; flex-shrink: 0; }
.row .val { width: 68px; font-size: 0.76em; color: #e94560; text-align: right; flex-shrink: 0; font-weight: bold; }
input[type=range] { flex: 1; accent-color: #e94560; min-width: 0; }
input[type=checkbox] { accent-color: #e94560; width: 15px; height: 15px; flex-shrink: 0; }
select { background: #0f3460; color: #e0e0e0; border: 1px solid #1a3a60; padding: 2px 5px; border-radius: 3px; font-family: monospace; font-size: 0.78em; }
.disabled { opacity: 0.35; pointer-events: none; }
button { background: #e94560; color: #fff; border: none; padding: 7px; border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 0.85em; width: 100%; }
button:hover { background: #c73652; }
</style>
</head>
<body>
<h1>Camera Tuner</h1>
<div class="layout">
  <div class="feed-panel">
    <img id="feed" src="/stream.mjpeg">
    <div id="stats">connecting&hellip;</div>
  </div>
  <div class="ctrl-panel">

    <div class="group">
      <h3>Exposure</h3>
      <div class="row">
        <label>Auto</label>
        <input type="checkbox" id="ae" checked onchange="onAe()">
      </div>
      <div id="exp-group">
        <div class="row">
          <label>Exp (&#x03bc;s)</label>
          <input type="range" id="exp" min="2" max="6.5" step="0.005" value="4.3" oninput="onLog('exp','exp-val','&#x03bc;s')">
          <span class="val" id="exp-val">20000&#x03bc;s</span>
        </div>
        <div class="row">
          <label>Gain</label>
          <input type="range" id="gain" min="1" max="16" step="0.1" value="1" oninput="onSlider('gain','gain-val','x')">
          <span class="val" id="gain-val">1.0x</span>
        </div>
      </div>
      <div class="row" title="Maximum exposure the AE algorithm may use">
        <label>AE max exp</label>
        <input type="range" id="maxexp" min="2" max="6.5" step="0.005" value="6.477" oninput="onLog('maxexp','maxexp-val','&#x03bc;s')">
        <span class="val" id="maxexp-val">3000k&#x03bc;s</span>
      </div>
    </div>

    <div class="group">
      <h3>White Balance</h3>
      <div class="row">
        <label>Auto</label>
        <input type="checkbox" id="awb" checked onchange="onAwb()">
      </div>
      <div class="row">
        <label>AWB mode</label>
        <select id="awb-mode" onchange="apply()">
          <option value="0">Auto</option>
          <option value="1">Incandescent</option>
          <option value="2">Tungsten</option>
          <option value="3">Fluorescent</option>
          <option value="4">Indoor</option>
          <option value="5">Daylight</option>
          <option value="6">Cloudy</option>
        </select>
      </div>
      <div id="gains-group">
        <div class="row">
          <label>Red gain</label>
          <input type="range" id="rgain" min="0.5" max="4" step="0.01" value="1.5" oninput="onSlider('rgain','rgain-val','')">
          <span class="val" id="rgain-val">1.50</span>
        </div>
        <div class="row">
          <label>Blue gain</label>
          <input type="range" id="bgain" min="0.5" max="4" step="0.01" value="1.5" oninput="onSlider('bgain','bgain-val','')">
          <span class="val" id="bgain-val">1.50</span>
        </div>
      </div>
    </div>

    <div class="group">
      <h3>Image</h3>
      <div class="row">
        <label>Brightness</label>
        <input type="range" id="brightness" min="-1" max="1" step="0.01" value="0" oninput="onSlider('brightness','brightness-val','')">
        <span class="val" id="brightness-val">0.00</span>
      </div>
      <div class="row">
        <label>Contrast</label>
        <input type="range" id="contrast" min="0" max="32" step="0.1" value="1" oninput="onSlider('contrast','contrast-val','')">
        <span class="val" id="contrast-val">1.0</span>
      </div>
      <div class="row">
        <label>Saturation</label>
        <input type="range" id="saturation" min="0" max="32" step="0.1" value="1" oninput="onSlider('saturation','saturation-val','')">
        <span class="val" id="saturation-val">1.0</span>
      </div>
      <div class="row">
        <label>Sharpness</label>
        <input type="range" id="sharpness" min="0" max="16" step="0.1" value="1" oninput="onSlider('sharpness','sharpness-val','')">
        <span class="val" id="sharpness-val">1.0</span>
      </div>
      <div class="row">
        <label>Denoise</label>
        <select id="denoise" onchange="apply()">
          <option value="0">Off</option>
          <option value="1" selected>Fast</option>
          <option value="2">High quality</option>
        </select>
      </div>
    </div>

    <button onclick="saveFrame()">&#x1f4f7; Save frame</button>

    <div class="group">
      <h3>Config YAML</h3>
      <textarea id="yaml-out" readonly rows="10" style="width:100%;background:#0f3460;color:#e0e0e0;border:none;border-radius:3px;padding:6px;font-family:monospace;font-size:0.74em;resize:vertical"></textarea>
      <button onclick="copyYaml()" style="margin-top:4px">Copy</button>
    </div>

  </div>
</div>
<script>
let debounce = null;

// log-scale slider: slider value is log10(actual); range 2-6.5 = 100-3.16M
function onLog(id, valId, unit) {
  const v = Math.round(Math.pow(10, parseFloat(document.getElementById(id).value)));
  document.getElementById(valId).textContent = v >= 100000 ? (v/1000).toFixed(0)+'k'+unit : v+unit;
  apply();
}

function logSliderFor(actual) {
  return Math.log10(Math.max(100, actual)).toFixed(3);
}

function logActual(id) {
  return Math.round(Math.pow(10, parseFloat(document.getElementById(id).value)));
}

function onSlider(id, valId, unit) {
  const v = parseFloat(document.getElementById(id).value);
  document.getElementById(valId).textContent = v.toFixed(2) + unit;
  apply();
}

function onAe() {
  const on = document.getElementById('ae').checked;
  document.getElementById('exp-group').classList.toggle('disabled', on);
  if (!on) {
    fetch('/stats').then(r => r.json()).then(s => {
      document.getElementById('exp').value = logSliderFor(s.exposure_us);
      document.getElementById('exp-val').textContent = s.exposure_us + 'μs';
      document.getElementById('gain').value = Math.min(s.gain, 16);
      document.getElementById('gain-val').textContent = s.gain.toFixed(2) + 'x';
    });
  }
  apply();
}

function onAwb() {
  const on = document.getElementById('awb').checked;
  document.getElementById('gains-group').classList.toggle('disabled', on);
  if (!on) {
    fetch('/stats').then(r => r.json()).then(s => {
      document.getElementById('rgain').value = Math.min(s.colour_gains[0], 4);
      document.getElementById('rgain-val').textContent = s.colour_gains[0].toFixed(3);
      document.getElementById('bgain').value = Math.min(s.colour_gains[1], 4);
      document.getElementById('bgain-val').textContent = s.colour_gains[1].toFixed(3);
    });
  }
  apply();
}

function apply() {
  clearTimeout(debounce);
  debounce = setTimeout(sendControls, 200);
}

function sendControls() {
  const ae = document.getElementById('ae').checked;
  const awb = document.getElementById('awb').checked;
  const controls = {
    AeEnable: ae,
    AwbEnable: awb,
    AwbMode: parseInt(document.getElementById('awb-mode').value),
    Brightness: parseFloat(document.getElementById('brightness').value),
    Contrast: parseFloat(document.getElementById('contrast').value),
    Saturation: parseFloat(document.getElementById('saturation').value),
    Sharpness: parseFloat(document.getElementById('sharpness').value),
    NoiseReductionMode: parseInt(document.getElementById('denoise').value),
    FrameDurationLimits: [33333, logActual('maxexp')],
  };
  if (!ae) {
    controls.ExposureTime = logActual('exp');
    controls.AnalogueGain = parseFloat(document.getElementById('gain').value);
  }
  if (!awb) {
    controls.ColourGains = [
      parseFloat(document.getElementById('rgain').value),
      parseFloat(document.getElementById('bgain').value),
    ];
  }
  fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(controls)})
    .catch(() => {});
  generateYaml();
}

function saveFrame() {
  const a = document.createElement('a');
  a.href = '/snapshot.jpg';
  a.download = 'frame_' + Date.now() + '.jpg';
  a.click();
}

function generateYaml() {
  const ae = document.getElementById('ae').checked;
  const awb = document.getElementById('awb').checked;
  const maxexp_s = (logActual('maxexp') / 1e6).toFixed(1);
  const lines = ['camera:'];
  lines.push('  max_exposure_seconds: ' + maxexp_s);
  lines.push('  ae_enable: ' + ae);
  if (!ae) {
    lines.push('  exposure_time_us: ' + logActual('exp'));
    lines.push('  analogue_gain: ' + parseFloat(document.getElementById('gain').value).toFixed(2));
  }
  lines.push('  awb_enable: ' + awb);
  lines.push('  awb_mode: ' + document.getElementById('awb-mode').value +
    '  # 0=Auto 1=Incandescent 2=Tungsten 3=Fluorescent 4=Indoor 5=Daylight 6=Cloudy');
  if (!awb) {
    lines.push('  red_gain: ' + parseFloat(document.getElementById('rgain').value).toFixed(3));
    lines.push('  blue_gain: ' + parseFloat(document.getElementById('bgain').value).toFixed(3));
  }
  lines.push('  brightness: ' + parseFloat(document.getElementById('brightness').value).toFixed(2));
  lines.push('  contrast: ' + parseFloat(document.getElementById('contrast').value).toFixed(1));
  lines.push('  saturation: ' + parseFloat(document.getElementById('saturation').value).toFixed(1));
  lines.push('  sharpness: ' + parseFloat(document.getElementById('sharpness').value).toFixed(1));
  lines.push('  noise_reduction_mode: ' + document.getElementById('denoise').value +
    '  # 0=Off 1=Fast 2=HighQuality');
  document.getElementById('yaml-out').value = lines.join('\n');
}

function copyYaml() {
  const ta = document.getElementById('yaml-out');
  ta.select();
  navigator.clipboard.writeText(ta.value).catch(() => document.execCommand('copy'));
}

// Poll stats and update auto-value displays
setInterval(() => {
  fetch('/stats').then(r => r.json()).then(s => {
    document.getElementById('stats').textContent =
      s.shape + '  mean=' + s.mean + '  std=' + s.std +
      '  exp=' + s.exposure_us + 'μs  gain=' + s.gain + '×' +
      '  R=' + s.colour_gains[0] + '  B=' + s.colour_gains[1] + '  lux=' + s.lux;
    if (document.getElementById('ae').checked) {
      document.getElementById('exp-val').textContent = s.exposure_us + 'μs (auto)';
      document.getElementById('gain-val').textContent = s.gain + '× (auto)';
    }
    if (document.getElementById('awb').checked) {
      document.getElementById('rgain-val').textContent = s.colour_gains[0] + ' (auto)';
      document.getElementById('bgain-val').textContent = s.colour_gains[1] + ' (auto)';
    }
  }).catch(() => {});
}, 1000);

onAe(); onAwb(); generateYaml();
</script>
</body>
</html>
"""

# ─── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML.encode())
        elif path == "/stream.mjpeg":
            self._stream()
        elif path == "/stats":
            with _lock:
                body = json.dumps(_stats).encode()
            self._send(200, "application/json", body)
        elif path == "/snapshot.jpg":
            with _lock:
                body = _jpeg
            self._send(200, "image/jpeg", body, extra={"Content-Disposition": "attachment"})
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/set":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            controls = json.loads(body)
            _apply(controls)
            self._send(200, "application/json", b'{"ok":true}')
        except Exception as exc:
            self._send(400, "application/json", json.dumps({"error": str(exc)}).encode())

    def _send(self, code: int, ct: str, body: bytes, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _lock:
                    j = _jpeg
                if j:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                    )
                    self.wfile.flush()
                time.sleep(0.12)  # ~8 fps to the browser
        except (BrokenPipeError, ConnectionResetError):
            pass


def _apply(controls: dict) -> None:
    ae_on = bool(controls.get("AeEnable", True))
    awb_on = bool(controls.get("AwbEnable", True))
    mapped: dict = {}

    for key, val in controls.items():
        if key in ("AeEnable", "AwbEnable"):
            mapped[key] = bool(val)
        elif key in ("AwbMode", "NoiseReductionMode"):
            mapped[key] = int(val)
        elif key in ("Brightness", "Contrast", "Saturation", "Sharpness"):
            mapped[key] = float(val)
        elif key == "FrameDurationLimits":
            mapped[key] = (int(val[0]), int(val[1]))
        elif key == "ExposureTime" and not ae_on:
            mapped[key] = int(val)
        elif key == "AnalogueGain" and not ae_on:
            mapped[key] = float(val)
        elif key == "ColourGains" and not awb_on:
            mapped[key] = (float(val[0]), float(val[1]))

    if mapped:
        try:
            _cam.set_controls(mapped)
        except Exception as exc:
            print(f"set_controls error: {exc}")


# ─── main ─────────────────────────────────────────────────────────────────────

def _local_ips() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if not ip.startswith("127") and ":" not in ip:
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips) or ["<pi-ip>"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive camera tuner")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    global _cam
    from picamera2 import Picamera2

    _cam = Picamera2()
    cfg = _cam.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 3_000_000)},
    )
    _cam.configure(cfg)
    _cam.start()
    print(f"Camera started at {args.width}x{args.height}")

    threading.Thread(target=capture_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    for ip in _local_ips():
        print(f"  http://{ip}:{args.port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _cam.stop()
        _cam.close()
        print("Camera closed.")


if __name__ == "__main__":
    main()
