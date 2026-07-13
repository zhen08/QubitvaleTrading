"""跟踪复盘 v3：多账本——每本账独立对照（paper vs 模型回放 vs 冻结期望带）。

口径规则（第二轮 review 确立，全账本一致）：
  1) 每本账的期望带基准独立冻结（paper/<book>/baseline.*），复盘不随数据更新重估；
  2) 带 = 冻结 OOS 日收益样本的 block bootstrap 经验分位（固定种子）；
  3) 日度统计只用 note=='settled' 的权益；
  4) 模型回放 = 用同期实际数据重放该策略（同期对照，非基准漂移）。
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

TRAIN_BARS = 730   # 与 Phase 1 协议一致：OOS 从第 730 根起
BOOT_N = 4000
BOOT_BLOCK = 10
BOOT_SEED = 42

# 账本 → (research 家族, 固定参数)：模型回放与基准构造的单一映射
BOOK_MODEL = {
    "donchian_ensemble": ("donchian", None),
    "tsmom_ensemble": ("tsmom", {"long_short": False, "max_lev": 1.0}),
}


def _family_ens(df: pd.DataFrame, family: str, fixed: dict | None) -> pd.Series:
    vr = build_variant_returns(df, family, SPOT_TAKER, None, fixed)
    return vr.mean(axis=1)


def portfolio_model_returns(settings: dict, book: str) -> pd.Series:
    """该账本策略的模型日收益（全历史、净成本）——同期对照用。"""
    family, fixed = BOOK_MODEL[book]
    store = storeio.store_dir(settings)
    legs = {sym: _family_ens(load_spot_daily(store, sym), family, fixed)
            for sym in settings["symbols"]}
    port = pd.concat(legs, axis=1).dropna().mean(axis=1)
    port.index = port.index.normalize()
    return port


def build_phase1_portfolio(settings: dict, book: str) -> pd.Series:
    """该账本策略的 Phase 1 组合口径 OOS 日收益（冻结基准的原料）。"""
    family, fixed = BOOK_MODEL[book]
    store = storeio.store_dir(settings)
    legs = [_family_ens(load_spot_daily(store, sym), family, fixed).iloc[TRAIN_BARS:]
            for sym in settings["symbols"]]
    return pd.concat(legs, axis=1).dropna().mean(axis=1)


# ---------------- 基准冻结（按账本） ----------------

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
    """冻结样本的 block bootstrap：n 日累计收益经验分位数（固定种子）。"""
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


# ---------------- 复盘（全账本） ----------------

def _book_section(settings: dict, book: str, bcfg: dict, store: Path) -> tuple[list[str], dict]:
    led = Ledger.load_or_init(store, float(bcfg["initial_capital_usdt"]),
                              str(bcfg["start_date"]), book=book)
    settled = led.equity_series(settled_only=True)
    lines = [f"## 账本：{book}（起始 {bcfg['start_date']}，${bcfg['initial_capital_usdt']:,}）", ""]
    stats: dict = {"settled_days": int(len(settled))}
    if len(settled) < 1:
        lines.append("尚无已结算权益记录，暂无统计。")
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
        band_src = f"冻结基准（{meta['frozen_at'][:10]}，{meta['n_days']} 日）bootstrap"
    else:
        band = bootstrap_band(build_phase1_portfolio(settings, book).to_numpy(), max(n, 1))
        band_src = "**非正式**（未冻结——运行 python -m scripts.freeze_baseline）"
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
        f"窗口：{n} 个已结算日（≈{weeks:.1f}/6 周）；带基准：{band_src}",
        "",
        "| 指标 | Paper | 模型回放 | 备注 |",
        "|---|---|---|---|",
        f"| 累计收益 | {100*cum_paper:.2f}% | {100*cum_model:.2f}% | 差 {1e4*(cum_paper-cum_model):.0f} bps |",
        f"| 年化 Sharpe | {sharpe(paper_ret):.2f} | {sharpe(model):.2f} | |",
        (f"| TE(年化) | {100*te_ann:.2f}% | — | 目标 <2% |" if te_ann == te_ann
         else "| TE | 样本不足 | — | |"),
        f"| 期望带 80%/95% | {100*band['p10']:.2f}%~{100*band['p90']:.2f}% | ±({100*band['p2_5']:.2f}%~{100*band['p97_5']:.2f}%) | {'带内 ✅' if in95 else '**出 95% 带 ❌**'} |",
        f"| 成交/费用 | live {n_live} / catchup {n_catch} | ${fees:.2f} | |",
        f"| 持仓 | {led.positions or '空仓'} | 现金 ${led.cash:,.2f} | |",
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

    lines = [f"# Paper 复盘（多账本）— {pd.Timestamp.now(tz='UTC').date()}", "",
             f"运维事故（自 {earliest}）：P0={inc['P0']} P1={inc['P1']} P2={inc['P2']} P3={inc['P3']}"
             f"（明细 data/store/ops/incidents.parquet）；P0 门槛：{'✅ 0' if inc['P0'] == 0 else '❌'}",
             "",
             "> 两本账均为**未经统计认证的研究候选**（Phase 1 修订判定），属探索性验证；"
             "Phase 3 选择纪律（ex-ante）：两本都达标 → 各半仓部署，不选赢家。",
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
