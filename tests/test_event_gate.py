"""事件门窗口逻辑。"""
import pandas as pd

from intel.event_gate import entries_blocked

SETTINGS = {"intel": {"event_gate_lookahead_hours": 36, "event_gate_cooldown_hours": 1}}
EVENTS = [{"name": "CPI", "kind": "cpi", "utc": pd.Timestamp("2026-07-14T12:30:00Z")}]


def _blocked(ts):
    b, _ = entries_blocked(pd.Timestamp(ts), SETTINGS, EVENTS)
    return b


def test_gate_windows():
    assert not _blocked("2026-07-12T12:00:00Z")   # 事件前 48.5h：放行
    assert _blocked("2026-07-13T04:00:00Z")       # 事件前 32.5h：禁新仓
    assert _blocked("2026-07-14T12:00:00Z")       # 事件前 30min
    assert _blocked("2026-07-14T13:20:00Z")       # 事件后 50min（冷却内）
    assert not _blocked("2026-07-14T13:31:00Z")   # 事件后 61min：解除


def test_no_events_never_blocks():
    b, why = entries_blocked(pd.Timestamp("2026-07-14T12:30:00Z"), SETTINGS, [])
    assert not b and why is None
