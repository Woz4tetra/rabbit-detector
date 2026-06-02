from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

from zoneinfo import ZoneInfo

from .config import PROJECT_ROOT, ServerSettings, load_server_settings
from .motion import MotionDetector

_EASTERN = ZoneInfo("America/New_York")


def _fmt_ts(ts: str) -> str:
    if not ts:
        return ts
    try:
        dt = datetime.datetime.fromisoformat(ts.rstrip("Z")).replace(
            tzinfo=datetime.timezone.utc
        )
        return dt.astimezone(_EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")
    except ValueError:
        return ts
from .moondream_loader import load_moondream

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = asyncio.Lock()
_settings: ServerSettings | None = None
_notifier = None
_recent_frames: Deque[dict] = deque(maxlen=5)   # last few frames for status bar
_rabbit_detections: list[dict] = []              # all rabbit detections, persisted to disk
_detections_log_path: Path | None = None
_latest_frame_path: Path | None = None
_last_detection_frame_path: Path | None = None
_motion: MotionDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _settings, _notifier, _latest_frame_path, _last_detection_frame_path
    global _detections_log_path, _rabbit_detections, _motion

    _settings, email_cfg = load_server_settings()

    if _settings.motion_enabled:
        _motion = MotionDetector(
            threshold=_settings.motion_threshold,
            min_area_frac=_settings.motion_min_area_frac,
            bg_alpha=_settings.motion_bg_alpha,
            pad_frac=_settings.motion_pad_frac,
            min_crop_px=_settings.motion_min_crop_px,
            warmup_frames=_settings.motion_warmup_frames,
            max_regions=_settings.motion_max_regions,
        )

    frames_dir = PROJECT_ROOT / _settings.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)

    latest = frames_dir / "latest.jpg"
    detection = frames_dir / "last_detection.jpg"
    _latest_frame_path = latest if latest.exists() else None
    _last_detection_frame_path = detection if detection.exists() else None

    log_path = PROJECT_ROOT / "data" / "detections.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _detections_log_path = log_path
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        _rabbit_detections.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        logger.info("Loaded %d rabbit detections from %s", len(_rabbit_detections), log_path)

    if email_cfg.enabled:
        from rabbit_deterrent.notifier import EmailNotifier
        _notifier = EmailNotifier(email_cfg)

    logger.info("Loading Moondream2 (%s @ %s) on %s …", _settings.model_id, _settings.model_revision, _settings.device)
    _model, _tokenizer = load_moondream(
        model_id=_settings.model_id,
        revision=_settings.model_revision,
        device=_settings.device,
    )
    logger.info("Model loaded. Server ready.")
    yield


app = FastAPI(title="Rabbit Detector Server", lifespan=lifespan)


class DetectionResponse(BaseModel):
    rabbit: bool
    confidence: float
    raw_response: str


def _detect_count(image: Image.Image) -> int:
    """Return the number of target objects Moondream localizes in the image."""
    total = 0
    for obj in _settings.detection_objects:
        try:
            result = _model.detect(image, obj)
        except Exception:
            logger.exception("detect() failed for %r", obj)
            continue
        total += len(result.get("objects", []))
    return total


def _run_inference(image_bytes: bytes) -> DetectionResponse:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # No motion gating: run detect() on the whole frame.
    if _motion is None:
        count = _detect_count(image)
        return DetectionResponse(
            rabbit=count > 0,
            confidence=1.0 if count > 0 else 0.0,
            raw_response=f"detect: {count} object(s) (full frame)",
        )

    frame_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])
    motion = _motion.update(frame_bgr)

    # While the background model seeds, fall back to whole-frame detection so we
    # don't silently miss anything during warmup.
    if motion.warming:
        count = _detect_count(image)
        return DetectionResponse(
            rabbit=count > 0,
            confidence=1.0 if count > 0 else 0.0,
            raw_response=f"detect: {count} object(s) (warmup, full frame)",
        )

    if not motion.regions:
        return DetectionResponse(rabbit=False, confidence=0.0, raw_response="no motion")

    total = 0
    hit_region: tuple[int, int, int, int] | None = None
    for (x, y, w, h) in motion.regions:
        count = _detect_count(image.crop((x, y, x + w, y + h)))
        if count:
            total += count
            if hit_region is None:
                hit_region = (x, y, w, h)

    if total > 0:
        raw = f"animal in motion region {hit_region}: {total} object(s) across {len(motion.regions)} region(s)"
        return DetectionResponse(rabbit=True, confidence=1.0, raw_response=raw)
    return DetectionResponse(
        rabbit=False,
        confidence=0.0,
        raw_response=f"motion only, no animal in {len(motion.regions)} region(s)",
    )


def _save_frame(image_bytes: bytes, is_detection: bool) -> str:
    global _latest_frame_path, _last_detection_frame_path
    frames_dir = PROJECT_ROOT / _settings.frames_dir

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"{ts}.jpg"
    (frames_dir / name).write_bytes(image_bytes)

    latest = frames_dir / "latest.jpg"
    latest.write_bytes(image_bytes)
    _latest_frame_path = latest

    if is_detection:
        detection = frames_dir / "last_detection.jpg"
        detection.write_bytes(image_bytes)
        _last_detection_frame_path = detection

    _prune_frames(frames_dir)
    return name


def _prune_frames(frames_dir: Path) -> None:
    if _settings.max_frames <= 0:
        return
    skip = {"latest.jpg", "last_detection.jpg"}
    frames = sorted(f for f in frames_dir.glob("*.jpg") if f.name not in skip)
    for old in frames[: -_settings.max_frames]:
        old.unlink(missing_ok=True)


def _send_email(image_bytes: bytes, timestamp: str, raw_response: str) -> None:
    if _notifier is None:
        return
    arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bgr = arr[:, :, ::-1].copy()
    _notifier.send(
        subject="Rabbit detected!",
        body=f"Detected at {timestamp}\nModel response: {raw_response}",
        image=bgr,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "moondream2"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(frame: UploadFile = File(...)):
    data = await frame.read()
    if not data:
        raise HTTPException(400, "Empty frame")

    loop = asyncio.get_event_loop()
    async with _lock:
        result = await loop.run_in_executor(None, _run_inference, data)

    ts = datetime.datetime.utcnow().isoformat() + "Z"
    frame_name = _save_frame(data, result.rabbit)
    record: dict = {"timestamp": ts, "rabbit_present": result.rabbit, "confidence": result.confidence, "raw_response": result.raw_response}
    _recent_frames.append(record)

    if result.rabbit:
        record["frame"] = frame_name
        _rabbit_detections.append(record)
        if _detections_log_path is not None:
            with open(_detections_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        threading.Thread(target=_send_email, args=(data, ts, result.raw_response), daemon=True).start()

    return result


@app.get("/latest-frame")
async def latest_frame():
    if _latest_frame_path is None or not _latest_frame_path.exists():
        raise HTTPException(404, "No frames received yet")
    return Response(content=_latest_frame_path.read_bytes(), media_type="image/jpeg")


@app.get("/last-detection-frame")
async def last_detection_frame():
    if _last_detection_frame_path is None or not _last_detection_frame_path.exists():
        raise HTTPException(404, "No rabbit detections yet")
    return Response(content=_last_detection_frame_path.read_bytes(), media_type="image/jpeg")


@app.get("/frame/{name}")
async def get_frame(name: str):
    if "/" in name or "\\" in name or not name.endswith(".jpg"):
        raise HTTPException(400, "Invalid name")
    frames_dir = PROJECT_ROOT / _settings.frames_dir
    path = frames_dir / name
    if not path.exists():
        raise HTTPException(404, "Frame not found")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@app.get("/api/state")
async def api_state():
    if _recent_frames:
        last = _recent_frames[-1]
        status_text = f'RABBIT DETECTED at {_fmt_ts(last["timestamp"])}' if last["rabbit_present"] else f'Clear at {_fmt_ts(last["timestamp"])}'
        status_color = "#ff6b6b" if last["rabbit_present"] else "#51cf66"
    elif _rabbit_detections:
        last = _rabbit_detections[-1]
        status_text = f'Last rabbit: {_fmt_ts(last["timestamp"])}'
        status_color = "#ff6b6b"
    else:
        status_text, status_color = "Waiting for first frame…", "#888"
    return {
        "status_text": status_text,
        "status_color": status_color,
        "detections": [
            {"timestamp": d.get("timestamp", 0.0), "rabbit_present": d.get("rabbit_present", False), "frame": d.get("frame", "")}
            for d in reversed(_rabbit_detections) if d.get("rabbit_present")
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    rows = ""
    for d in reversed(_rabbit_detections):
        if not d.get("rabbit_present"):
            continue
        frame = d.get("frame", "")
        data_attr = f'data-frame="{frame}"' if frame else ""
        dim = "" if frame else " no-frame-row"
        rows += (
            f'<tr class="det-row{dim}" {data_attr} style="cursor:pointer">'
            f'<td>{_fmt_ts(d.get("timestamp", ""))}</td>'
            f'<td>{"Rabbit detected" if d.get("rabbit_present") else "Clear"}</td>'
            f'</tr>\n'
        )

    if _recent_frames:
        last = _recent_frames[-1]
        status_text = f'RABBIT DETECTED at {_fmt_ts(last["timestamp"])}' if last["rabbit_present"] else f'Clear at {_fmt_ts(last["timestamp"])}'
        status_color = "#ff6b6b" if last["rabbit_present"] else "#51cf66"
    elif _rabbit_detections:
        last = _rabbit_detections[-1]
        status_text = f'Last rabbit: {_fmt_ts(last["timestamp"])}'
        status_color = "#ff6b6b"
    else:
        status_text, status_color = "Waiting for first frame…", "#888"

    no_rows = '<tr><td colspan="2" style="text-align:center;color:#555">No rabbit detections yet</td></tr>'

    has_detection = _last_detection_frame_path is not None
    detection_src = "/last-detection-frame" if has_detection else ""
    detection_display = "block" if has_detection else "none"
    no_detection_display = "none" if has_detection else "block"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rabbit Detector</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #e94560; margin-bottom: 12px; font-size: 1.6em; }}
  .status {{ font-size: 1.1em; color: {status_color}; padding: 10px 14px; background: #16213e;
             border-radius: 4px; margin-bottom: 18px; border-left: 3px solid {status_color}; }}
  .frames {{ display: flex; gap: 16px; margin-bottom: 16px; }}
  .frame-box {{ flex: 1; min-width: 0; background: #16213e; padding: 14px; border-radius: 6px; }}
  .frame-box h2 {{ font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 1px;
                   margin-bottom: 10px; }}
  .frame-box img {{ width: 100%; border-radius: 4px; display: block; }}
  .no-frame {{ color: #555; font-style: italic; padding: 60px 0; text-align: center; }}
  table {{ width: 100%; border-collapse: collapse; background: #16213e; border-radius: 6px;
           overflow: hidden; font-size: 0.88em; }}
  th {{ background: #0f3460; padding: 8px 12px; text-align: left; color: #aaa;
        text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; }}
  td {{ padding: 7px 12px; border-top: 1px solid #0f3460; }}
  tr.det-row {{ cursor: pointer; }}
  tr.det-row:hover td {{ background: #0f3460; }}
  tr.det-row.selected td {{ background: #1a3a60; border-left: 3px solid #ff6b6b; }}
  tr.no-frame-row td {{ color: #777; }}
  @media (max-width: 700px) {{
    .frames {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
<h1>Rabbit Detector</h1>
<div class="status">{status_text}</div>
<div class="frames">
  <div class="frame-box">
    <h2>Live Feed</h2>
    <img id="live" src="/latest-frame?t=0"
         onerror="this.style.display='none';document.getElementById('live-placeholder').style.display='block'">
    <div id="live-placeholder" class="no-frame" style="display:none">No frames yet</div>
  </div>
  <div class="frame-box">
    <h2 id="det-label">Last Detection</h2>
    <img id="det-img" src="{detection_src}" style="display:{detection_display};width:100%;border-radius:4px">
    <div id="det-placeholder" class="no-frame" style="display:{no_detection_display}">No detections yet</div>
  </div>
</div>
<table>
  <thead><tr><th>Time</th><th>Detection</th></tr></thead>
  <tbody id="det-tbody">{rows or no_rows}</tbody>
</table>
<script>
  (function() {{
    function fmtTs(utcStr) {{
      try {{
        return new Date(utcStr).toLocaleString('en-US', {{
          timeZone: 'America/New_York',
          year: 'numeric', month: '2-digit', day: '2-digit',
          hour: '2-digit', minute: '2-digit', second: '2-digit',
          hour12: false
        }}) + ' ET';
      }} catch(e) {{ return utcStr; }}
    }}
    var liveImg = document.getElementById('live');
    var livePh = document.getElementById('live-placeholder');
    setInterval(function() {{
      var next = new Image();
      next.onload = function() {{
        liveImg.src = next.src;
        liveImg.style.display = 'block';
        livePh.style.display = 'none';
      }};
      next.src = '/latest-frame?t=' + Date.now();
    }}, 2000);

    var detImg = document.getElementById('det-img');
    var detPh = document.getElementById('det-placeholder');
    var detLabel = document.getElementById('det-label');
    var selected = null;

    document.getElementById('det-tbody').addEventListener('click', function(e) {{
      var row = e.target.closest('tr.det-row');
      if (!row) return;
      if (selected) selected.classList.remove('selected');
      row.classList.add('selected');
      selected = row;
      var frame = row.getAttribute('data-frame');
      var ts = row.cells[0] ? row.cells[0].textContent : '';
      detLabel.textContent = ts ? 'Detection: ' + ts : 'Selected Detection';
      if (frame) {{
        detImg.src = '/frame/' + frame + '?t=' + Date.now();
        detImg.style.display = 'block';
        detPh.style.display = 'none';
      }} else {{
        detImg.style.display = 'none';
        detPh.textContent = 'No image saved for this detection';
        detPh.style.display = 'block';
      }}
    }});

    var statusEl = document.querySelector('.status');
    setInterval(function() {{
      fetch('/api/state').then(function(r) {{ return r.json(); }}).then(function(data) {{
        statusEl.textContent = data.status_text;
        statusEl.style.color = data.status_color;
        statusEl.style.borderLeftColor = data.status_color;

        var tbody = document.getElementById('det-tbody');
        if (data.detections.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="2" style="text-align:center;color:#555">No rabbit detections yet</td></tr>';
          return;
        }}
        var html = '';
        data.detections.forEach(function(d) {{
          var attr = d.frame ? ' data-frame="' + d.frame + '"' : '';
          var dim = d.frame ? '' : ' no-frame-row';
          var sel = (selected && selected.getAttribute('data-frame') === d.frame) ? ' selected' : '';
          html += '<tr class="det-row' + dim + sel + '"' + attr + ' style="cursor:pointer"><td>' + fmtTs(d.timestamp) + '</td><td>' + (d.rabbit_present ? 'Rabbit detected' : 'Clear') + '</td></tr>';
        }});
        tbody.innerHTML = html;
      }}).catch(function() {{}});
    }}, 5000);
  }})();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
