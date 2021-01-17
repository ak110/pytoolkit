#!/usr/bin/env python3
"""tk.notification.post()をするスクリプト。"""
import argparse
import pathlib
import sys

try:
    import pytoolkit as tk
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    import pytoolkit as tk

logger = tk.log.get(__name__)


def main():
    tk.utils.better_exceptions()
    tk.log.init(None)

    parser = argparse.ArgumentParser(description="tk.notification.post()をするスクリプト。")
    parser.add_argument("body", type=str, help="メッセージ")
    parser.add_argument("subject", type=str, nargs="?", default=None, help="件名(省略可)")
    args = parser.parse_args()
    tk.notifications.post(body=args.body, subject=args.subject)


if __name__ == "__main__":
    main()
