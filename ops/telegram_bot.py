"""Interactive Telegram bot: long-polls getUpdates and replies to commands.

Only the authorized TELEGRAM_CHAT_ID (from .env) is served; messages from any
other chat are ignored. Replies are sent as plain text (no Markdown) so book
names containing '_' never trigger Telegram parse errors.

Commands:
  /help                 list commands
  /status               per-book equity, return, positions, cash
  /positions            current positions per book
  /history [book] [N]   last N trades (default 10) for a book (default: all books)
  /risk                 current news risk flags

Run: python -m scripts.telegram_bot   (or the qubitvale-telegram systemd service)

Note: only ONE process may long-poll getUpdates for a given bot token. The daily
job only uses sendMessage, so it does not conflict with this listener.
"""
from __future__ import annotations

import json
import logging
import os
import time

import requests

from data import storeio
from data.collectors.common import load_env, load_settings, setup_logging
from execution.paper.ledger import Ledger
from intel.news_scorer import load_risk_flags

log = logging.getLogger("qvt.tgbot")
API = "https://api.telegram.org"

HELP = (
    "QVT paper bot — commands:\n"
    "/status — per-book equity, return, positions, cash\n"
    "/positions — current positions per book\n"
    "/history [book] [N] — last N trades (default 10); book optional (partial match)\n"
    "/pnl — per-book PnL, return, fees, trade counts\n"
    "/signal — latest target weights (decision) per book\n"
    "/risk — current news risk flags\n"
    "/help — this message"
)


def _api(token: str, method: str, **params) -> dict:
    r = requests.get(f"{API}/bot{token}/{method}", params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def _send(token: str, chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"{API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=30,
        ).raise_for_status()
    except Exception as exc:  # noqa: BLE001 — a failed reply must not kill the loop
        log.warning("send failed: %s", exc)


def _ledger(settings: dict, store, book: str) -> Ledger:
    bcfg = settings["paper"]["books"][book]
    return Ledger.load_or_init(store, float(bcfg["initial_capital_usdt"]),
                               str(bcfg["start_date"]), book=book)


def cmd_status(settings: dict, store) -> str:
    lines = ["QVT status"]
    for book, bcfg in settings["paper"]["books"].items():
        led = _ledger(settings, store, book)
        eq = led.equity_series()
        last = float(eq.iloc[-1]) if len(eq) else led.initial_capital
        ret = 100.0 * (last / led.initial_capital - 1.0)
        pos = ", ".join(f"{k} {v:.5f}" for k, v in led.positions.items()) or "flat"
        asof = f" @ {eq.index[-1].date()}" if len(eq) else ""
        lines.append(f"\n{book}\n  equity ${last:,.2f} ({ret:+.2f}%){asof}"
                     f"\n  {pos} | cash ${led.cash:,.2f}")
    return "\n".join(lines)


def cmd_positions(settings: dict, store) -> str:
    lines = ["QVT positions"]
    for book in settings["paper"]["books"]:
        led = _ledger(settings, store, book)
        pos = ", ".join(f"{k} {v:.6g}" for k, v in led.positions.items()) or "flat"
        lines.append(f"{book}: {pos} | cash ${led.cash:,.2f}")
    return "\n".join(lines)


def cmd_history(settings: dict, store, args: list[str]) -> str:
    books = settings["paper"]["books"]
    book_filter, n = None, 10
    for a in args:
        if a.isdigit():
            n = max(1, min(50, int(a)))
        else:
            book_filter = a
    targets = [b for b in books if book_filter is None or book_filter.lower() in b.lower()]
    if not targets:
        return f"no book matching '{book_filter}'. books: {', '.join(books)}"
    out = ["QVT trade history"]
    for book in targets:
        led = _ledger(settings, store, book)
        df = led.trades_df()
        out.append(f"\n{book} ({len(df)} trade{'s' if len(df) != 1 else ''}"
                   + (f", last {min(n, len(df))}" if len(df) > n else "") + ")")
        if len(df) == 0:
            out.append("  (none)")
            continue
        for r in df.tail(n).itertuples():
            out.append(f"  {r.day} {str(r.side).upper()} {r.symbol} {r.qty:.6g} @ {r.price:.2f}"
                       f" (${r.notional:,.2f} fee ${r.fee:.2f}) [{r.mode}]")
    return "\n".join(out)


def cmd_pnl(settings: dict, store) -> str:
    lines = ["QVT PnL"]
    for book in settings["paper"]["books"]:
        led = _ledger(settings, store, book)
        eq = led.equity_series()
        last = float(eq.iloc[-1]) if len(eq) else led.initial_capital
        pnl = last - led.initial_capital
        ret = 100.0 * (last / led.initial_capital - 1.0)
        df = led.trades_df()
        fees = float(df["fee"].sum()) if len(df) else 0.0
        n_live = int((df["mode"] == "live").sum()) if len(df) else 0
        n_catch = int((df["mode"] == "catchup").sum()) if len(df) else 0
        lines.append(f"\n{book}\n  equity ${last:,.2f}  PnL ${pnl:+,.2f} ({ret:+.2f}%)"
                     f"\n  fees ${fees:.2f} | trades {len(df)} (live {n_live}/catchup {n_catch})")
    return "\n".join(lines)


def cmd_signal(settings: dict, store) -> str:
    lines = ["QVT signals (latest decision)"]
    for book in settings["paper"]["books"]:
        p = store / "signals" / f"{book}.latest.json"
        if not p.exists():
            lines.append(f"\n{book}: (no signal yet)")
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        tw = ", ".join(f"{k} {100 * v:.2f}%" for k, v in s.get("target_weights", {}).items()) or "flat"
        lines.append(f"\n{book}\n  decision {s.get('decision_date')} -> effective {s.get('effective_from')}"
                     f"\n  {tw}\n  gross {100 * s.get('gross_exposure', 0.0):.2f}%")
    return "\n".join(lines)


def cmd_risk(settings: dict) -> str:
    f = load_risk_flags(settings)
    keys = ("generated_at", "scorer", "asset_neg_severity", "market_neg_severity", "stale")
    return "QVT risk flags\n" + json.dumps({k: f.get(k) for k in keys},
                                           ensure_ascii=False, indent=1)


def handle(cmd: str, args: list[str], settings: dict, store) -> str:
    if cmd in ("help", "start"):
        return HELP
    if cmd == "status":
        return cmd_status(settings, store)
    if cmd == "positions":
        return cmd_positions(settings, store)
    if cmd == "history":
        return cmd_history(settings, store, args)
    if cmd == "pnl":
        return cmd_pnl(settings, store)
    if cmd == "signal":
        return cmd_signal(settings, store)
    if cmd == "risk":
        return cmd_risk(settings)
    return f"unknown command /{cmd}\n\n{HELP}"


def run() -> None:
    setup_logging()
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    auth = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not auth:
        raise SystemExit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")
    settings = load_settings()
    store = storeio.store_dir(settings)

    # Drain any backlog so commands sent before startup are not replayed.
    offset = None
    try:
        d = _api(token, "getUpdates", timeout=0)
        if d.get("ok") and d["result"]:
            offset = d["result"][-1]["update_id"] + 1
    except Exception as exc:  # noqa: BLE001
        log.warning("initial drain failed: %s", exc)

    log.info("telegram bot online (authorized chat %s)", auth)
    _send(token, auth, "QVT bot online. Send /help for commands.")

    while True:
        try:
            d = _api(token, "getUpdates", timeout=30, offset=offset)
        except Exception as exc:  # noqa: BLE001
            log.warning("getUpdates: %s", exc)
            time.sleep(5)
            continue
        if not d.get("ok"):
            time.sleep(5)
            continue
        for u in d.get("result", []):
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if chat_id != str(auth):
                log.info("ignoring message from unauthorized chat %s", chat_id)
                continue
            if not text.startswith("/"):
                _send(token, auth, "Send /help for commands.")
                continue
            parts = text.split()
            cmd = parts[0].lstrip("/").split("@")[0].lower()  # tolerate /cmd@botname
            log.info("command /%s args=%s", cmd, parts[1:])
            try:
                reply = handle(cmd, parts[1:], settings, store)
            except Exception as exc:  # noqa: BLE001
                log.exception("command failed")
                reply = f"error handling /{cmd}: {exc}"
            _send(token, auth, reply)
            log.info("replied to /%s (%d chars)", cmd, len(reply))


if __name__ == "__main__":
    run()
