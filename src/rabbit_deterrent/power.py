from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def apply_power_optimizations() -> None:
    _run("tvservice -o", "Disable HDMI")
    _run(
        "echo 0 > /sys/class/leds/led0/brightness",
        "Disable ACT LED",
        shell=True,
    )
    _run("iwconfig wlan0 power off", "Disable WiFi power management")


def _run(cmd: str, label: str, shell: bool = False) -> None:
    try:
        if shell:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        else:
            subprocess.run(cmd.split(), check=True, capture_output=True)
        logger.debug("%s: OK", label)
    except subprocess.CalledProcessError as exc:
        logger.warning("%s failed (non-fatal): %s", label, exc.stderr.decode().strip())
