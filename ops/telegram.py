"""Telegram 通知（可选）：.env 里有 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 才发，否则只打日志。"""
from __future__ import annotations

import logging
import os

import requests

from data.collectors.common import load_env

log = logging.getLogger("qvt.tg")


def send(text: str) -> bool:
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        log.info("telegram not configured; message:\n%s", text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text[:4000], "parse_mode": "Markdown"},
            timeout=30,
        )
        r.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 — 通知失败不阻断主流程
        log.warning("telegram send failed: %s", exc)
        return False
