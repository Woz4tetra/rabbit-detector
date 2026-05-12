from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

_CHECK_INTERVAL = 5.0


def _has_ip(iface: str = "wlan0") -> bool:
    try:
        r = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            capture_output=True, text=True, timeout=5,
        )
        return "inet " in r.stdout
    except Exception:
        return False


def _start_hotspot(ssid: str, password: str) -> bool:
    cmd = [
        "nmcli", "device", "wifi", "hotspot",
        "ifname", "wlan0",
        "ssid", ssid,
        "password", password,
    ]
    for attempt in [cmd, ["sudo"] + cmd]:
        try:
            subprocess.run(attempt, check=True, timeout=30, capture_output=True)
            logger.info("Hotspot '%s' started — connect and SSH to 192.168.4.1", ssid)
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning("Hotspot attempt failed: %s", exc.stderr.decode().strip())
        except FileNotFoundError:
            logger.error("nmcli not found — cannot start hotspot")
            return False
    return False


def wait_for_network(
    timeout: float = 120.0,
    ssid: str = "RabbitDetector",
    password: str = "rabbitdet",
) -> bool:
    """Wait up to timeout seconds for a WiFi IP. Start a hotspot if none arrives.

    Returns True if a normal network connection was obtained, False if hotspot mode.
    """
    if _has_ip():
        logger.debug("Network already up")
        return True

    logger.info("No network — waiting up to %.0fs before starting hotspot…", timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_CHECK_INTERVAL)
        if _has_ip():
            logger.info("Network connection obtained")
            return True

    logger.warning("No network after %.0fs", timeout)
    _start_hotspot(ssid, password)
    return False
