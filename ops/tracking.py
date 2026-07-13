"""Tracking review v3: multi-book — each book compared independently (paper vs model replay vs frozen expected band).

Basis rules (established in the second review, consistent across all books):
  1) Each book's expected-band baseline is frozen independently (paper/<book>/baseline.*); the review does not re-estimate it as data updates;
  2) Band = block-bootstrap empirical quantiles of the frozen OOS daily-return sample (fixed seed);
  3) Daily statistics use only equity where note=='settled';
  4) Model replay = replay the strategy on the same-period actual data (same-period comparison, not baseline drift).
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data import storeio
from execution.paper.ledger import Ledger
from research.costs import SPOT_TAKER
from research.metrics import ANN, sharpe
from research.walkforward import build_variant_returns
from strategies.base import load_spot_daily

log = logging.getLogger("qvt.track")

TRAIN_BARS = 730   # consistent with the Phase 1 protocol: OOS starts from the 730th bar
BOOT_N = 4000
BOOT_BLOCK = 10
BOOT_SEED = 42

# book → (research family, fixed params): the single mapping for model replay and baseline construction
BOOK_MODEL = {
    "donchian_ensemble": ("donchian", None),
    "tsmom_ensemble": ("tsmom", {"long_short": False, "max_lev": 1.0}),
}


def _family_ens(df: pd.DataFrame, family: str, fixed: dict | None) -> pd.Series:
    vr = build_variant_returns(df, family, SPOT_TAKER, None, fixed)
    return vr.mean(axis=1)


def portfolio_model_returns(settings: dict, book: str) -> pd.Series:
    """This book's strategy model daily returns (full history, net of costs) — for same-period comparison."""
    family, fixed = BOOK_MODEL[book]
    store = storeio.store_dir(settings)
    legs = {sym: _family_ens(load_spot_daily(store, sym), family, fixed)
            for sym in settings["symbols"]}
    port = pd.concat(legs, axis=1).dropna().mean(axis=1)
    port.index = port.index.normalize()
    return port


def build_phase1_portfolio(settings: dict, book: str) -> pd.Series:
    """This book's strategy Phase 1 portfolio-basis OOS daily returns (raw material for the frozen baseline)."""
    family, fixed = BOOK_MODEL[book]
    store = storeio.store_dir(settings)
    legs = [_family_ens(load_spot_daily(store, sym), family, fixed).iloc[TRAIN_BARS:]
            for sym in settings["symbols"]]
    return pd.concat(legs, axis=1).dropna().mean(axis=1)


# ---------------- baseline freeze (per book) ----------------

def baseline_paths(store: Path, book: str) -> tuple[Path, Path]:
    d = store / "paper" / book
    return d / "baseline.json", d / "baseline_returns.parquet"


def freeze_baseline(settings: dict, book: str, force: bool = False) -> dict:
    store = storeio.store_dir(settings)
    jp, rp = baseline_paths(store, book)
    if jp.exists() and not force:
        raise FileExistsError(f"baseline already frozen at {jp} (use --force to refreeze)")
    jp.parent.mkdir(parents=True, exist_ok=True)
    port = build_phase1_portfolio(settings, book)
    meta = {"book": book, "frozen_at": str(pd.Timestamp.now(tz="UTC")),
            "n_days": int(len(port)),
            "window": [str(port.index[0].date()), str(port.index[-1].date())],
            "mu_d": float(port.mean()), "sd_d": float(port.std(ddof=1)),
            "boot": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED}}
    df = port.to_frame("ret")
    df.index.name = "day"
    tmp = rp.with_suffix(".parquet.tmp")
    df.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, rp)
    jtmp = jp.with_suffix(".json.tmp")
    jtmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(jtmp, jp)
    log.info("baseline[%s] frozen: %d days %s mu_d=%.5f sd_d=%.5f",
             book, meta["n_days"], meta["window"], meta["mu_d"], meta["sd_d"])
    return meta


def load_baseline(settings: dict, book: str) -> tuple[dict, np.ndarray] | None:
    store = storeio.store_dir(settings)
    jp, rp = baseline_paths(store, book)
    if not (jp.exists() and rp.exists()):
        return None
    meta = json.loads(jp.read_text(encoding="utf-8"))
    return meta, pd.read_parquet(rp)["ret"].to_numpy()


def bootstrap_band(rets: np.ndarray, horizon: int,
                   n_boot: int = BOOT_N, block: int = BOOT_BLOCK,
                   seed: int = BOOT_SEED) -> dict:
    """Block bootstrap of the frozen sample: empirical quantiles of n-day cumulative return (fixed seed)."""
    rng = np.random.default_rng(seed)
    t = len(rets)
    n_blocks = max(1, math.ceil(horizon / block))
    starts = rng.integers(0, max(1, t - block), size=(n_boot, n_blocks))
    sims = np.empty(n_boot)
    for i in range(n_boot):
        path = np.concatenate([rets[s:s + block] for s in starts[i]])[:horizon]
        sims[i] = float(np.prod(1 + path) - 1)
    q = np.percentile(sims, [2.5, 10, 50, 90, 97.5])
    return {"p2_5": q[0], "p10": q[1], "p50": q[2], "p90": q[3], "p97_5": q[4]}


# ---------------- review (all books) ----------------

def _book_section(settings: dict, book: str, bcfg: dict, store: Path) -> tuple[list[str], dict]:
    led = Ledger.load_or_init(store, float(bcfg["initial_capital_usdt"]),
                              str(bcfg["start_date"]), book=book)
    settled = led.equity_series(settled_only=True)
    lines = [f"## Book: {book} (started {bcfg['start_date']}, ${bcfg['initial_capital_usdt']:,})", ""]
    stats: dict = {"settled_days": int(len(settled))}
    if len(settled) < 1:
        lines.append("No settled equity records yet, so no statistics available.")
        return lines + [""], stats

    start_anchor = pd.Timestamp(bcfg["start_date"], tz="UTC") - pd.Timedelta(days=1)
    eq = pd.concat([pd.Series([led.initial_capital], index=[start_anchor]), settled])
    paper_ret = eq.pct_change().dropna()
    n = len(paper_ret)
    model = portfolio_model_returns(settings, book).reindex(paper_ret.index).fillna(0.0)
    diff = paper_ret - model
    te_ann = float(diff.std(ddof=1)) * math.sqrt(ANN["1d"]) if n > 2 else float("nan")
    cum_paper = float((1 + paper_ret).prod() - 1)
    cum_model = float((1 + model).prod() - 1)

    frozen = load_baseline(settings, book)
    if frozen:
        meta, rets = frozen
        band = bootstrap_band(rets, max(n, 1))
        band_src = f"frozen baseline ({meta['frozen_at'][:10]}, {meta['n_days']} days) bootstrap"
    else:
        band = bootstrap_band(build_phase1_portfolio(settings, book).to_numpy(), max(n, 1))
        band_src = "**informal** (not frozen — run python -m scripts.freeze_baseline)"
    in80 = band["p10"] <= cum_paper <= band["p90"]
    in95 = band["p2_5"] <= cum_paper <= band["p97_5"]

    trades = led.trades_df()
    n_live = int((trades["mode"] == "live").sum()) if len(trades) else 0
    n_catch = int((trades["mode"] == "catchup").sum()) if len(trades) else 0
    fees = float(trades["fee"].sum()) if len(trades) else 0.0
    weeks = n / 7.0
    stats.update(cum_paper_pct=round(100 * cum_paper, 2), in_band95=bool(in95),
                 te_ann_pct=round(100 * te_ann, 2) if te_ann == te_ann else None,
                 weeks=round(weeks, 1), baseline_frozen=bool(frozen))

    lines += [
        f"Window: {n} settled day(s) (≈{weeks:.1f}/6 weeks); band baseline: {band_src}",
        "",
        "| Metric | Paper | Model replay | Note |",
        "|---|---|---|---|",
        f"| Cumulative return | {100*cum_paper:.2f}% | {100*cum_model:.2f}% | diff {1e4*(cum_paper-cum_model):.0f} bps |",
        f"| Annualized Sharpe | {sharpe(paper_ret):.2f} | {sharpe(model):.2f} | |",
        (f"| TE (annualized) | {100*te_ann:.2f}% | — | target <2% |" if te_ann == te_ann
         else "| TE | insufficient sample | — | |"),
        f"| Expected band 80%/95% | {100*band['p10']:.2f}%~{100*band['p90']:.2f}% | ±({100*band['p2_5']:.2f}%~{100*band['p97_5']:.2f}%) | {'in band ✅' if in95 else '**outside 95% band ❌**'} |",
        f"| Fills/fees | live {n_live} / catchup {n_catch} | ${fees:.2f} | |",
        f"| Positions | {led.positions or 'flat'} | cash ${led.cash:,.2f} | |",
        "",
    ]
    for r in trades.tail(5).itertuples():
        lines.append(f"- {r.day} {r.side} {r.symbol} {r.qty:.6g} @ {r.price:.2f} [{r.mode}]")
    lines.append("")
    return lines, stats


def build_review(settings: dict) -> tuple[str, dict]:
    store = storeio.store_dir(settings)
    books: dict = settings["paper"]["books"]
    from ops import incident_log
    earliest = min(str(b["start_date"]) for b in books.values())
    inc = incident_log.counts_since(store, earliest)

    lines = [f"# Paper Review (multi-book) — {pd.Timestamp.now(tz='UTC').date()}", "",
             f"Operational incidents (since {earliest}): P0={inc['P0']} P1={inc['P1']} P2={inc['P2']} P3={inc['P3']}"
             f" (details in data/store/ops/incidents.parquet); P0 gate: {'✅ 0' if inc['P0'] == 0 else '❌'}",
             "",
             "> Both books are **uncertified research candidates** (Phase 1 revised verdict), so this is "
             "exploratory validation; Phase 3 selection discipline (ex-ante): if both books pass → deploy "
             "each at half size, do not pick a winner.",
             ""]
    all_stats: dict = {"incidents": inc}
    for book, bcfg in books.items():
        sec, stats = _book_section(settings, book, bcfg, store)
        lines += sec
        all_stats[book] = stats
    return "\n".join(lines) + "\n", all_stats


def write_review(settings: dict) -> Path:
    report, _ = build_review(settings)
    out_dir = Path(__file__).resolve().parents[1] / "research" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"paper_review_{pd.Timestamp.now(tz='UTC').date()}.md"
    p.write_text(report, encoding="utf-8")
    log.info("paper review -> %s", p)
    return p
