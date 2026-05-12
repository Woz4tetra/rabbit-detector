from __future__ import annotations

import asyncio
import datetime
import io
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

from .config import PROJECT_ROOT, ServerSettings, load_server_settings
from .moondream_loader import load_moondream

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = asyncio.Lock()
_settings: ServerSettings | None = None
_notifier = None
_recent_detections: Deque[dict] = deque(maxlen=20)
_latest_frame_path: Path | None = None
_last_detection_frame_path: Path | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _settings, _notifier, _latest_frame_path, _last_detection_frame_path

    _settings, email_cfg = load_server_settings()

    frames_dir = PROJECT_ROOT / _settings.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)

    latest = frames_dir / "latest.jpg"
    detection = frames_dir / "last_detection.jpg"
    _latest_frame_path = latest if latest.exists() else None
    _last_detection_frame_path = detection if detection.exists() else None

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


def _run_inference(image_bytes: bytes) -> DetectionResponse:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    enc = _model.encode_image(image)
    raw: str = _model.query(enc, _settings.detection_prompt)["answer"]
    cleaned = raw.strip().lower().rstrip(".,!? \t\n")
    if cleaned.startswith("yes"):
        rabbit, confidence = True, 1.0
    elif cleaned.startswith("no"):
        rabbit, confidence = False, 0.0
    else:
        logger.warning("Ambiguous Moondream2 response: %r", raw)
        rabbit, confidence = False, 0.0
    return DetectionResponse(rabbit=rabbit, confidence=confidence, raw_response=raw)


def _save_frame(image_bytes: bytes, is_detection: bool) -> None:
    global _latest_frame_path, _last_detection_frame_path
    frames_dir = PROJECT_ROOT / _settings.frames_dir

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (frames_dir / f"{ts}.jpg").write_bytes(image_bytes)

    latest = frames_dir / "latest.jpg"
    latest.write_bytes(image_bytes)
    _latest_frame_path = latest

    if is_detection:
        detection = frames_dir / "last_detection.jpg"
        detection.write_bytes(image_bytes)
        _last_detection_frame_path = detection

    _prune_frames(frames_dir)


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
    _save_frame(data, result.rabbit)
    _recent_detections.append(
        {"timestamp": ts, "rabbit": result.rabbit, "confidence": result.confidence, "raw_response": result.raw_response}
    )

    if result.rabbit:
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


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    rows = ""
    for d in reversed(list(_recent_detections)):
        status = "RABBIT" if d["rabbit"] else "clear"
        color = "#ff6b6b" if d["rabbit"] else "#51cf66"
        rows += f'<tr><td>{d["timestamp"]}</td><td style="color:{color};font-weight:bold">{status}</td><td>{d["raw_response"]}</td></tr>\n'

    if _recent_detections:
        last = _recent_detections[-1]
        status_text = f'RABBIT DETECTED at {last["timestamp"]}' if last["rabbit"] else f'Clear at {last["timestamp"]}'
        status_color = "#ff6b6b" if last["rabbit"] else "#51cf66"
    else:
        status_text, status_color = "Waiting for first frame…", "#888"

    no_rows = '<tr><td colspan="3" style="text-align:center;color:#555">No detections logged yet</td></tr>'
    detection_panel = (
        '<img src="/last-detection-frame" style="max-width:100%;border-radius:4px">'
        if _last_detection_frame_path is not None
        else '<div class="no-frame">No detections yet</div>'
    )

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
  .frames {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .frame-box {{ flex: 1; min-width: 300px; background: #16213e; padding: 14px; border-radius: 6px; }}
  .frame-box h2 {{ font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 1px;
                   margin-bottom: 10px; }}
  .frame-box img {{ max-width: 100%; border-radius: 4px; display: block; }}
  .no-frame {{ color: #555; font-style: italic; padding: 30px 0; text-align: center; }}
  table {{ width: 100%; border-collapse: collapse; background: #16213e; border-radius: 6px;
           overflow: hidden; font-size: 0.88em; }}
  th {{ background: #0f3460; padding: 8px 12px; text-align: left; color: #aaa;
        text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; }}
  td {{ padding: 7px 12px; border-top: 1px solid #0f3460; }}
  tr:hover td {{ background: #0f3460; }}
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
    <h2>Last Detection</h2>
    {detection_panel}
  </div>
</div>
<table>
  <thead><tr><th>Time</th><th>Result</th><th>Model Response</th></tr></thead>
  <tbody>{rows or no_rows}</tbody>
</table>
<script>
  (function() {{
    var img = document.getElementById('live');
    var ph = document.getElementById('live-placeholder');
    setInterval(function() {{
      var next = new Image();
      next.onload = function() {{
        img.src = next.src;
        img.style.display = 'block';
        ph.style.display = 'none';
      }};
      next.src = '/latest-frame?t=' + Date.now();
    }}, 2000);
    setTimeout(function() {{ location.reload(); }}, 30000);
  }})();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
