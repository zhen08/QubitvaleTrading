"""Entrypoint for the interactive Telegram command bot.

Usage: python -m scripts.telegram_bot
Normally run via the qubitvale-telegram systemd user service (long-running).
"""
from __future__ import annotations

from ops.telegram_bot import run

if __name__ == "__main__":
    run()
