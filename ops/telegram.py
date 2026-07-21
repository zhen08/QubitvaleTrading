"""Telegram 通知（可选）：.env 里有 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 才发，否则只打日志。"""
from __future__ import annotations

import logging
import os

import requests

from data.collectors.common import load_env

log = logging.getLogger("qvt.tg")


def send(text: str) -> bool:
    """纯文本发送（不用 parse_mode）：账本名 donchian_ensemble 等本身含 `_`，
    legacy Markdown 会把动态内容解析成实体——凑成奇数个就整条 400 丢失
    （2026-07-15 事故），凑成偶数则渲染成意外斜体。通知必达优先于样式。
    失败时记录响应体，便于从日志直接定位问题。"""
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        log.info("telegram not configured; message:\n%s", text)
        return False

    def _redact(s: str) -> str:  # the request URL embeds the bot token — never log it raw
        return s.replace(token, "***") if token else s

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text[:4000]},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 — 通知失败不阻断主流程
        log.warning("telegram send failed: %s", _redact(str(exc)))
        return False
    if not r.ok:
        # Telegram's JSON error body (error_code/description) is safe to log and tells us
        # WHY it failed; the raised exception / URL would carry the token, so log neither.
        log.warning("telegram send rejected: HTTP %s %s", r.status_code, _redact(r.text[:300]))
        return False
    return True
