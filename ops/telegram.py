"""Telegram 通知（可选）：.env 里有 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 才发，否则只打日志。"""
from __future__ import annotations

import logging
import os

import requests

from data.collectors.common import load_env

log = logging.getLogger("qvt.tg")


def send(text: str) -> bool:
    """先按 Markdown 发；遇 400（内容实体解析失败，如奇数个下划线——账本名
    donchian_ensemble 等本身含 `_`，动态内容随时可能凑成奇数）自动降级为纯文本
    重发一次：通知必达优先于样式。失败时记录响应体，便于从日志直接定位内容问题。"""
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        log.info("telegram not configured; message:\n%s", text)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for payload in (
        {"chat_id": chat, "text": text[:4000], "parse_mode": "Markdown"},
        {"chat_id": chat, "text": text[:4000]},          # plain-text fallback
    ):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 400 and "parse_mode" in payload:
                log.warning("telegram markdown rejected, retrying plain: %s",
                            r.text[:300])
                continue
            r.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001 — 通知失败不阻断主流程
            log.warning("telegram send failed: %s", exc)
            return False
    return False
