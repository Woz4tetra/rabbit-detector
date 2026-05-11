from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def apply_power_optimizations() -> None:
    # tvservice is gone on Trixie; vcgencmd display_power 0 is the replacement
    _run("vcgencmd display_power 0", "Disable HDMI")
    # LED sysfs name is ACT on Trixie (was led0 on older Pi OS)
    _run(
        "echo 0 > /sys/class/leds/ACT/brightness",
        "Disable ACT LED",
        shell=True,
    )
    # WiFi power save is disabled permanently by optimize_pi.sh via NetworkManager conf.
    # Nothing to do at runtime.


def _run(cmd: str, label: str, shell: bool = False) -> None:
    try:
        if shell:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        else:
            subprocess.run(cmd.split(), check=True, capture_output=True)
        logger.debug("%s: OK", label)
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("%s failed (non-fatal): %s", label, exc)
