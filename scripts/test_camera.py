#!/usr/bin/env python3
"""Interactive camera tuner — keeps the camera open while you adjust settings live.

Reads config.yaml for default resolution and camera settings. Run on the Pi:
    python scripts/test_camera.py [--port 7000]

Then open http://<pi-ip>:7000 in a browser.
"""
from __future__ import annotations

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

            raw_cg = meta.get("ColourGains")
            try:
                cg = [round(float(raw_cg[0]), 3), round(float(raw_cg[1]), 3)]
            except (TypeError, IndexError):
                cg = [1.0, 1.0]

            with _lock:
                _jpeg = buf.tobytes()
                _stats = {
                    "shape": f"{frame.shape[1]}x{frame.shape[0]}",
                    "mean": round(float(frame.mean()), 1),
                    "std": round(float(frame.std()), 1),
                    "exposure_us": int(meta.get("ExposureTime", 0)),
                    "gain": round(float(meta.get("AnalogueGain", 1.0)), 2),
                    "colour_gains": cg,
                    "lux": round(float(meta.get("Lux", 0.0)), 1),
                }
        except Exception as exc:
            print(f"capture error: {exc}")
            time.sleep(0.5)

# ─── HTML template ────────────────────────────────────────────────────────────
# __CFG__ is replaced at startup with a JSON object of initial values from config.yaml

_HTML_TEMPLATE = """\
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
.row .val { width: 72px; font-size: 0.76em; color: #e94560; text-align: right; flex-shrink: 0; font-weight: bold; white-space: nowrap; overflow: hidden; }
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
        <input type="checkbox" id="ae">
      </div>
      <div id="exp-group">
        <div class="row">
          <label>Exp (&mu;s)</label>
          <input type="range" id="exp" min="2" max="6.5" step="0.005">
          <span class="val" id="exp-val"></span>
        </div>
        <div class="row">
          <label>Gain</label>
          <input type="range" id="gain" min="1" max="16" step="0.1">
          <span class="val" id="gain-val"></span>
        </div>
      </div>
      <div class="row" title="Maximum exposure the AE algorithm may use">
        <label>AE max exp</label>
        <input type="range" id="maxexp" min="2" max="6.5" step="0.005">
        <span class="val" id="maxexp-val"></span>
      </div>
    </div>

    <div class="group">
      <h3>White Balance</h3>
      <div class="row">
        <label>Auto</label>
        <input type="checkbox" id="awb">
      </div>
      <div class="row">
        <label>AWB mode</label>
        <select id="awb-mode">
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
          <input type="range" id="rgain" min="0.5" max="4" step="0.01">
          <span class="val" id="rgain-val"></span>
        </div>
        <div class="row">
          <label>Blue gain</label>
          <input type="range" id="bgain" min="0.5" max="4" step="0.01">
          <span class="val" id="bgain-val"></span>
        </div>
      </div>
    </div>

    <div class="group">
      <h3>Image</h3>
      <div class="row">
        <label>Brightness</label>
        <input type="range" id="brightness" min="-1" max="1" step="0.01">
        <span class="val" id="brightness-val"></span>
      </div>
      <div class="row">
        <label>Contrast</label>
        <input type="range" id="contrast" min="0" max="32" step="0.1">
        <span class="val" id="contrast-val"></span>
      </div>
      <div class="row">
        <label>Saturation</label>
        <input type="range" id="saturation" min="0" max="32" step="0.1">
        <span class="val" id="saturation-val"></span>
      </div>
      <div class="row">
        <label>Sharpness</label>
        <input type="range" id="sharpness" min="0" max="16" step="0.1">
        <span class="val" id="sharpness-val"></span>
      </div>
      <div class="row">
        <label>Denoise</label>
        <select id="denoise">
          <option value="0">Off</option>
          <option value="1">Fast</option>
          <option value="2">High quality</option>
        </select>
      </div>
    </div>

    <button id="save-btn">Save frame</button>

    <div class="group">
      <h3>Config YAML</h3>
      <textarea id="yaml-out" readonly rows="10" style="width:100%;background:#0f3460;color:#e0e0e0;border:none;border-radius:3px;padding:6px;font-family:monospace;font-size:0.74em;resize:vertical"></textarea>
      <button id="copy-btn" style="margin-top:4px">Copy</button>
    </div>

  </div>
</div>
<script>
var CFG = __CFG__;

// ── helpers ──────────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

// Log-scale helpers: slider value = log10(actual), range 2..6.5 = 100us..3.16Mus
function logToActual(v) { return Math.round(Math.pow(10, parseFloat(v))); }
function actualToLog(v) { return Math.log10(Math.max(100, v)).toFixed(4); }
function fmtUs(us) { return us >= 100000 ? (us/1000).toFixed(0)+'kμs' : us+'μs'; }

function setSlider(id, valId, value, fmt) {
  el(id).value = value;
  el(valId).textContent = fmt(value);
}

// ── init sliders from config ──────────────────────────────────────────────────

el('ae').checked = CFG.ae_enable;
el('awb').checked = CFG.awb_enable;
el('awb-mode').value = CFG.awb_mode;
el('denoise').value = CFG.noise_reduction_mode;

el('exp').value = actualToLog(CFG.exposure_time_us);
el('exp-val').textContent = fmtUs(CFG.exposure_time_us);
el('gain').value = CFG.analogue_gain;
el('gain-val').textContent = CFG.analogue_gain.toFixed(2)+'x';
el('maxexp').value = actualToLog(CFG.max_exposure_seconds * 1e6);
el('maxexp-val').textContent = fmtUs(CFG.max_exposure_seconds * 1e6);
el('rgain').value = CFG.red_gain;
el('rgain-val').textContent = CFG.red_gain.toFixed(3);
el('bgain').value = CFG.blue_gain;
el('bgain-val').textContent = CFG.blue_gain.toFixed(3);
el('brightness').value = CFG.brightness;
el('brightness-val').textContent = CFG.brightness.toFixed(2);
el('contrast').value = CFG.contrast;
el('contrast-val').textContent = CFG.contrast.toFixed(1);
el('saturation').value = CFG.saturation;
el('saturation-val').textContent = CFG.saturation.toFixed(1);
el('sharpness').value = CFG.sharpness;
el('sharpness-val').textContent = CFG.sharpness.toFixed(1);

// disable manual groups based on initial config
el('exp-group').classList.toggle('disabled', CFG.ae_enable);
el('gains-group').classList.toggle('disabled', CFG.awb_enable);

// ── event wiring ──────────────────────────────────────────────────────────────

var debounce = null;
function apply() {
  clearTimeout(debounce);
  debounce = setTimeout(sendControls, 200);
}

el('ae').addEventListener('change', function() {
  var on = this.checked;
  el('exp-group').classList.toggle('disabled', on);
  if (!on) {
    // seed manual sliders from last known camera values
    fetch('/stats').then(function(r){ return r.json(); }).then(function(s) {
      if (!s || !s.exposure_us) return;
      el('exp').value = actualToLog(s.exposure_us);
      el('exp-val').textContent = fmtUs(s.exposure_us);
      el('gain').value = Math.min(s.gain, 16);
      el('gain-val').textContent = s.gain.toFixed(2)+'x';
    }).catch(function(){});
  }
  apply();
});

el('awb').addEventListener('change', function() {
  var on = this.checked;
  el('gains-group').classList.toggle('disabled', on);
  if (!on) {
    fetch('/stats').then(function(r){ return r.json(); }).then(function(s) {
      if (!s || !s.colour_gains) return;
      el('rgain').value = Math.min(s.colour_gains[0], 4);
      el('rgain-val').textContent = s.colour_gains[0].toFixed(3);
      el('bgain').value = Math.min(s.colour_gains[1], 4);
      el('bgain-val').textContent = s.colour_gains[1].toFixed(3);
    }).catch(function(){});
  }
  apply();
});

el('awb-mode').addEventListener('change', apply);
el('denoise').addEventListener('change', apply);

function wireLog(id, valId) {
  el(id).addEventListener('input', function() {
    el(valId).textContent = fmtUs(logToActual(this.value));
    apply();
  });
}
function wireLinear(id, valId, digits, suffix) {
  el(id).addEventListener('input', function() {
    el(valId).textContent = parseFloat(this.value).toFixed(digits) + (suffix||'');
    apply();
  });
}

wireLog('exp', 'exp-val');
wireLog('maxexp', 'maxexp-val');
wireLinear('gain', 'gain-val', 2, 'x');
wireLinear('rgain', 'rgain-val', 3, '');
wireLinear('bgain', 'bgain-val', 3, '');
wireLinear('brightness', 'brightness-val', 2, '');
wireLinear('contrast', 'contrast-val', 1, '');
wireLinear('saturation', 'saturation-val', 1, '');
wireLinear('sharpness', 'sharpness-val', 1, '');

// ── send controls ─────────────────────────────────────────────────────────────

function sendControls() {
  var ae = el('ae').checked;
  var awb = el('awb').checked;
  var ctrl = {
    AeEnable: ae,
    AwbEnable: awb,
    AwbMode: parseInt(el('awb-mode').value),
    Brightness: parseFloat(el('brightness').value),
    Contrast: parseFloat(el('contrast').value),
    Saturation: parseFloat(el('saturation').value),
    Sharpness: parseFloat(el('sharpness').value),
    NoiseReductionMode: parseInt(el('denoise').value),
    FrameDurationLimits: [33333, logToActual(el('maxexp').value)],
  };
  if (!ae) {
    ctrl.ExposureTime = logToActual(el('exp').value);
    ctrl.AnalogueGain = parseFloat(el('gain').value);
  }
  if (!awb) {
    ctrl.ColourGains = [parseFloat(el('rgain').value), parseFloat(el('bgain').value)];
  }
  fetch('/set', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(ctrl)
  }).catch(function(){});
  generateYaml();
}

// ── stats polling ─────────────────────────────────────────────────────────────

setInterval(function() {
  fetch('/stats').then(function(r){ return r.json(); }).then(function(s) {
    if (!s || !s.shape) return;
    var cg = s.colour_gains || [0, 0];
    el('stats').textContent = s.shape + '  mean=' + s.mean + '  std=' + s.std
      + '  exp=' + s.exposure_us + 'μs  gain=' + s.gain + '×'
      + '  R=' + cg[0] + '  B=' + cg[1] + '  lux=' + s.lux;
    if (el('ae').checked) {
      el('exp-val').textContent = fmtUs(s.exposure_us) + ' (auto)';
      el('gain-val').textContent = s.gain + '× (auto)';
    }
    if (el('awb').checked) {
      el('rgain-val').textContent = cg[0] + ' (auto)';
      el('bgain-val').textContent = cg[1] + ' (auto)';
    }
  }).catch(function(){});
}, 1000);

// ── YAML output ───────────────────────────────────────────────────────────────

function generateYaml() {
  var ae = el('ae').checked;
  var awb = el('awb').checked;
  var maxexp_s = (logToActual(el('maxexp').value) / 1e6).toFixed(1);
  var lines = ['camera:'];
  lines.push('  max_exposure_seconds: ' + maxexp_s);
  lines.push('  ae_enable: ' + ae);
  if (!ae) {
    lines.push('  exposure_time_us: ' + logToActual(el('exp').value));
    lines.push('  analogue_gain: ' + parseFloat(el('gain').value).toFixed(2));
  }
  lines.push('  awb_enable: ' + awb);
  lines.push('  awb_mode: ' + el('awb-mode').value
    + '  # 0=Auto 1=Incandescent 2=Tungsten 3=Fluorescent 4=Indoor 5=Daylight 6=Cloudy');
  if (!awb) {
    lines.push('  red_gain: ' + parseFloat(el('rgain').value).toFixed(3));
    lines.push('  blue_gain: ' + parseFloat(el('bgain').value).toFixed(3));
  }
  lines.push('  brightness: ' + parseFloat(el('brightness').value).toFixed(2));
  lines.push('  contrast: ' + parseFloat(el('contrast').value).toFixed(1));
  lines.push('  saturation: ' + parseFloat(el('saturation').value).toFixed(1));
  lines.push('  sharpness: ' + parseFloat(el('sharpness').value).toFixed(1));
  lines.push('  noise_reduction_mode: ' + el('denoise').value
    + '  # 0=Off 1=Fast 2=HighQuality');
  el('yaml-out').value = lines.join('\n');
}

generateYaml();

el('save-btn').addEventListener('click', function() {
  var a = document.createElement('a');
  a.href = '/snapshot.jpg';
  a.download = 'frame_' + Date.now() + '.jpg';
  a.click();
});

el('copy-btn').addEventListener('click', function() {
  var ta = el('yaml-out');
  ta.select();
  navigator.clipboard.writeText(ta.value).catch(function() {
    document.execCommand('copy');
  });
});
</script>
</body>
</html>
"""

# ─── HTTP handler ─────────────────────────────────────────────────────────────

_html_cache: str = ""   # filled in main() once config is known


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _html_cache.encode())
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
                time.sleep(0.12)
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
    global _cam, _html_cache
    import argparse
    parser = argparse.ArgumentParser(description="Interactive camera tuner")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    from rabbit_deterrent.config import load_config
    cfg = load_config(args.config)
    cam_cfg = cfg.camera
    width = cfg.detection.capture_width
    height = cfg.detection.capture_height

    cfg_dict = {
        "max_exposure_seconds": cam_cfg.max_exposure_seconds,
        "ae_enable": cam_cfg.ae_enable,
        "exposure_time_us": cam_cfg.exposure_time_us,
        "analogue_gain": cam_cfg.analogue_gain,
        "awb_enable": cam_cfg.awb_enable,
        "awb_mode": cam_cfg.awb_mode,
        "red_gain": cam_cfg.red_gain,
        "blue_gain": cam_cfg.blue_gain,
        "brightness": cam_cfg.brightness,
        "contrast": cam_cfg.contrast,
        "saturation": cam_cfg.saturation,
        "sharpness": cam_cfg.sharpness,
        "noise_reduction_mode": cam_cfg.noise_reduction_mode,
    }
    _html_cache = _HTML_TEMPLATE.replace("__CFG__", json.dumps(cfg_dict))

    from picamera2 import Picamera2
    _cam = Picamera2()
    pi_cfg = _cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, int(cam_cfg.max_exposure_seconds * 1_000_000))},
    )
    _cam.configure(pi_cfg)
    _cam.start()
    print(f"Camera started at {width}x{height}")

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
