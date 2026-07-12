"""事件门：排期宏观事件（CPI/FOMC…）窗口内禁开新仓（允许减仓/平仓）。

规则（settings.intel）：now ∈ [event − lookahead_hours, event + cooldown_hours] → 禁新仓。
日频调仓下 lookahead 取 36h，覆盖"事件前不加仓"的精神（调研报告 §7.1）。
事件表：config/calendar.yaml（人工核实后录入；宁缺毋滥）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from data.collectors.common import REPO_ROOT

log = logging.getLogger("qvt.gate")


def load_events(path: Path | None = None) -> list[dict]:
    p = path or REPO_ROOT / "config" / "calendar.yaml"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    events = []
    for e in data.get("events", []):
        try:
            events.append({"name": e["name"], "kind": e.get("kind", "event"),
                           "utc": pd.Timestamp(e["utc"])})
        except (KeyError, ValueError) as exc:
            log.warning("calendar entry skipped: %s (%s)", e, exc)
    return events


def entries_blocked(now: pd.Timestamp, settings: dict,
                    events: list[dict] | None = None) -> tuple[bool, str | None]:
    cfg = settings.get("intel", {})
    look = pd.Timedelta(hours=float(cfg.get("event_gate_lookahead_hours", 36)))
    cool = pd.Timedelta(hours=float(cfg.get("event_gate_cooldown_hours", 1)))
    for e in (events if events is not None else load_events()):
        if e["utc"] - look <= now <= e["utc"] + cool:
            return True, f"{e['name']} @ {e['utc']}"
    return False, None
