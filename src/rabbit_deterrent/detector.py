from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class OnnxRabbitDetector:
    def __init__(self, model_path: Path, confidence_threshold: float, image_size: int) -> None:
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        logger.info("Loading ONNX model from %s", model_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Model loaded. Input: %s", self.input_name)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        blob, scale, (pad_x, pad_y) = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], scale, pad_x, pad_y, image_bgr.shape)

    def _preprocess(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        import cv2

        h, w = image_bgr.shape[:2]
        scale = min(self.image_size / w, self.image_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])
        return blob, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        original_shape: tuple,
    ) -> list[Detection]:
        # raw shape from YOLOv8n ONNX (no end2end): [1, num_classes+4, num_anchors]
        raw = raw[0].T  # -> [num_anchors, num_classes+4]
        boxes_xywh = raw[:, :4]
        scores = raw[:, 4:]

        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= self.confidence_threshold
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert xywh (center) -> xyxy, undo letterbox
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        # NMS
        import cv2

        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=confidences.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=0.45,
        )
        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        return [
            Detection(
                x1=float(x1[i]),
                y1=float(y1[i]),
                x2=float(x2[i]),
                y2=float(y2[i]),
                confidence=float(confidences[i]),
                class_id=int(class_ids[i]),
            )
            for i in indices
        ]
