"""§4 portfolio rule, shared verbatim by the GRU variants and R0/R1 (§5).

Daily event loop over decision dates: rank members by score, hold the top 5
equal-weight (20% each), enter only from rank <= 5, exit on rank > 10 or
score below the confidence threshold or membership loss; a slot whose
replacement lacks confidence stays in cash. 2% rebalance threshold. Costs on
one-way turnover at each asset's own cost rate. Position decided at D earns
ret_next[D]; a held asset with no next bar earns 0 and is force-exited with
cost (amendment 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_SLOTS = 5
SLOT_WEIGHT = 0.20
EXIT_RANK = 10
REBALANCE_THRESHOLD = 0.02


@dataclass
class SimResult:
    net: pd.Series
    gross: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)   # date -> tuple(symbols)

    @property
    def ann_turnover(self) -> float:
        return float(self.turnover.mean() * 365)


def simulate(dates: pd.DatetimeIndex, score: pd.DataFrame, member: pd.DataFrame,
             ret_next: pd.DataFrame, cost_rate: pd.DataFrame,
             confidence: pd.DataFrame | None = None,
             equal_weight_all: bool = False) -> SimResult:
    """confidence: boolean DataFrame — True where the score is trusted enough
    to hold (GRU: p >= 0.5; R1: mom21 > 0). None -> always confident (R0).
    equal_weight_all=True implements R0 (1/N over all members, no slots)."""
    held: dict[str, float] = {}
    net, gross, tover = [], [], []
    holdings = {}
    for d in dates:
        mem = member.loc[d]
        sc = score.loc[d].where(mem)
        conf = (confidence.loc[d] if confidence is not None
                else pd.Series(True, index=sc.index))

        if equal_weight_all:
            names = list(sc.dropna().index)
            target = {s: 1.0 / len(names) for s in names} if names else {}
        else:
            ranks = sc.rank(ascending=False, method="first")
            target = {}
            # keep: held names still member, rank <= EXIT_RANK, still confident
            for s, w in held.items():
                if bool(mem.get(s, False)) and ranks.get(s, np.inf) <= EXIT_RANK \
                        and bool(conf.get(s, False)):
                    target[s] = SLOT_WEIGHT
            # fill empty slots only from the top-N_SLOTS ranked, confident names
            top = ranks[ranks <= N_SLOTS].sort_values().index
            for s in top:
                if len(target) >= N_SLOTS:
                    break
                if s not in target and bool(conf.get(s, False)):
                    target[s] = SLOT_WEIGHT

        # 2% rebalance threshold per name
        final = {}
        for s in set(held) | set(target):
            cur, tgt = held.get(s, 0.0), target.get(s, 0.0)
            final_w = cur if abs(tgt - cur) <= REBALANCE_THRESHOLD else tgt
            if final_w > 0:
                final[s] = final_w

        day_turn, day_cost = 0.0, 0.0
        for s in set(held) | set(final):
            dw = abs(final.get(s, 0.0) - held.get(s, 0.0))
            if dw > 0:
                day_turn += dw
                cr = cost_rate.at[d, s] if s in cost_rate.columns else np.nan
                day_cost += dw * (cr if np.isfinite(cr) else 0.0011)

        day_gross = 0.0
        forced_exit = []
        for s, w in final.items():
            r = ret_next.at[d, s] if s in ret_next.columns else np.nan
            if np.isfinite(r):
                day_gross += w * r
            else:                       # delisted overnight: 0 return, exit w/ cost
                forced_exit.append(s)
        for s in forced_exit:
            cr = cost_rate.at[d, s]
            day_cost += final[s] * (cr if np.isfinite(cr) else 0.0011)
            del final[s]

        net.append(day_gross - day_cost)
        gross.append(day_gross)
        tover.append(day_turn)
        holdings[d] = tuple(sorted(final))
        held = final
    idx = pd.DatetimeIndex(dates)
    return SimResult(net=pd.Series(net, index=idx), gross=pd.Series(gross, index=idx),
                     turnover=pd.Series(tover, index=idx), holdings=holdings)


# ---------------- L/S mechanical transform (track3_ls_shadow_preregistration) ----------------

LS_SLOT_WEIGHT = 0.10          # ±10% per name: gross <= 1.0, target net 0


def ls_targets(score_row: pd.Series, member_row: pd.Series, shortable: set,
               held_long: set, held_short: set,
               neutral: float = 0.5) -> dict[str, float]:
    """One day's L/S target weights from the frozen rule. Pure function:
    hysteresis state (currently held names) is passed in, targets come out.
    `neutral`: confidence midpoint — 0.5 for GRU probabilities, 0.0 for the
    R1-LS momentum reference (long needs score >= neutral, short <= neutral)."""
    sc = score_row.where(member_row)
    valid = sc.dropna()
    n = len(valid)
    if n == 0:
        return {}
    ranks = valid.rank(ascending=False, method="first")
    target: dict[str, float] = {}
    # long leg: keep while rank <= 10 and score >= 0.5; enter only rank <= 5
    for s in held_long:
        if s in ranks and ranks[s] <= EXIT_RANK and valid[s] >= neutral:
            target[s] = LS_SLOT_WEIGHT
    for s in ranks[ranks <= N_SLOTS].sort_values().index:
        if sum(1 for v in target.values() if v > 0) >= N_SLOTS:
            break
        if s not in target and valid[s] >= neutral:
            target[s] = LS_SLOT_WEIGHT
    # short leg: mirror at the bottom, perp-shortable only
    for s in held_short:
        if s in ranks and ranks[s] >= n - EXIT_RANK + 1 and valid[s] <= neutral \
                and s in shortable:
            target[s] = -LS_SLOT_WEIGHT
    bottom = ranks[ranks >= n - N_SLOTS + 1].sort_values(ascending=False).index
    for s in bottom:
        if sum(1 for v in target.values() if v < 0) >= N_SLOTS:
            break
        if s not in target and valid[s] <= neutral and s in shortable:
            target[s] = -LS_SLOT_WEIGHT
    return target


def simulate_ls(dates: pd.DatetimeIndex, score: pd.DataFrame, member: pd.DataFrame,
                ret_next: pd.DataFrame, cost_rate: pd.DataFrame, shortable: set,
                funding_next: pd.DataFrame | None = None,
                neutral: float = 0.5) -> SimResult:
    """L/S daily loop with the same threshold/cost/delisting semantics as
    `simulate`. funding_next: next-day funding rate paid by longs / received
    by shorts (perp convention: position * -funding)."""
    held: dict[str, float] = {}
    net, gross, tover = [], [], []
    holdings = {}
    for d in dates:
        held_long = {s for s, w in held.items() if w > 0}
        held_short = {s for s, w in held.items() if w < 0}
        target = ls_targets(score.loc[d], member.loc[d], shortable,
                            held_long, held_short, neutral=neutral)
        final = {}
        for s in set(held) | set(target):
            cur, tgt = held.get(s, 0.0), target.get(s, 0.0)
            w = cur if abs(tgt - cur) <= REBALANCE_THRESHOLD else tgt
            if w != 0.0:
                final[s] = w
        day_turn, day_cost = 0.0, 0.0
        for s in set(held) | set(final):
            dw = abs(final.get(s, 0.0) - held.get(s, 0.0))
            if dw > 0:
                day_turn += dw
                cr = cost_rate.at[d, s] if s in cost_rate.columns else np.nan
                day_cost += dw * (cr if np.isfinite(cr) else 0.0011)
        day_gross = 0.0
        forced = []
        for s, w in final.items():
            r = ret_next.at[d, s] if s in ret_next.columns else np.nan
            if np.isfinite(r):
                day_gross += w * r
                if funding_next is not None and s in funding_next.columns:
                    f = funding_next.at[d, s]
                    if np.isfinite(f):
                        day_gross -= w * f          # longs pay positive funding
            else:
                forced.append(s)
        for s in forced:
            cr = cost_rate.at[d, s]
            day_cost += abs(final[s]) * (cr if np.isfinite(cr) else 0.0011)
            del final[s]
        net.append(day_gross - day_cost)
        gross.append(day_gross)
        tover.append(day_turn)
        holdings[d] = tuple(sorted(final))
        held = final
    idx = pd.DatetimeIndex(dates)
    return SimResult(net=pd.Series(net, index=idx), gross=pd.Series(gross, index=idx),
                     turnover=pd.Series(tover, index=idx), holdings=holdings)
