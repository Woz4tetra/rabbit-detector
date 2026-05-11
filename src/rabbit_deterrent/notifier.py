from __future__ import annotations

import logging
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np

from .config import EmailConfig

logger = logging.getLogger(__name__)


class EmailNotifier:
    def __init__(self, config: EmailConfig) -> None:
        self._config = config
        self._last_sent: float = 0.0

    def _on_cooldown(self) -> bool:
        return (time.time() - self._last_sent) < self._config.cooldown_seconds

    def send(self, subject: str, body: str, image: np.ndarray | None = None) -> bool:
        if not self._config.enabled:
            logger.debug("Email disabled")
            return False
        if self._on_cooldown():
            logger.debug("Email suppressed (cooldown)")
            return False

        import cv2

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self._config.from_addr
        msg["To"] = self._config.to_addr
        msg.attach(MIMEText(body, "plain"))

        if image is not None:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                img_part = MIMEImage(buf.tobytes(), _subtype="jpeg")
                img_part.add_header("Content-Disposition", "attachment", filename="detection.jpg")
                msg.attach(img_part)

        try:
            with smtplib.SMTP(self._config.host, self._config.port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(self._config.username, self._config.password)
                smtp.sendmail(self._config.from_addr, self._config.to_addr, msg.as_string())
            self._last_sent = time.time()
            logger.info("Email sent: %s", subject)
            return True
        except Exception as exc:
            logger.error("Failed to send email: %s", exc)
            return False
