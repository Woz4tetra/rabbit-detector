"""Send a test email notification to verify SMTP credentials.

Usage:
    python scripts/test_notify.py [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    from rabbit_deterrent.config import load_config
    from rabbit_deterrent.notifier import EmailNotifier

    config = load_config(args.config)
    notifier = EmailNotifier(config.email)

    print(f"Sending test email from {config.email.from_addr} to {config.email.to_addr}...")
    ok = notifier.send(
        subject="[rabbit-deterrent] Test notification",
        body="If you received this, email notifications are working correctly.",
    )
    if ok:
        print("Email sent. Check your inbox.")
    else:
        print("Failed to send email. Check credentials in config.yaml.")
        sys.exit(1)


if __name__ == "__main__":
    main()
