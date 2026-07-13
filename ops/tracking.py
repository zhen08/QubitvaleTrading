"""跟踪复盘 v2：paper vs 模型回放 vs **冻结**的期望带（第二轮 review 修正）。

口径规则：
  1) 期望带基准在自动化正式启动时 **冻结一次**（scripts.freeze_baseline →
     data/store/paper/baseline.json + baseline_returns.parquet），复盘不再随
     数据更新重估；未冻结时退回滚动估计并明确标注"非正式"。
  2) 带由冻结 OOS 日收益样本的 **block bootstrap 经验分布** 给出（固定种子、
     块长 10、4000 次重采样的 n 日累计收益分位数），不再用正态近似。
  3) 日度统计只用 note=='settled' 的权益记录，intraday 标记不入样本。
  4) 模型回放（TE 的对比对象）按定义必须用同期实际数据重放——这是"同期对照"，
     不属于基准漂移。
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
from strategies.donchian_ensemble import load_spot_daily

log = logging.getLogger("qvt.track")

TRAIN_BARS = 730   # 与 Phase 1 协议一致：OOS 从第 730 根起
BOOT_N = 4000
BOOT_BLOCK = 10
BOOT_SEED = 42


def _family_ens(df: pd.DataFrame) -> pd.Series:
    vr = build_variant_returns(df, "donchian", SPOT_TAKER)
    return vr.mean(axis=1)


def portfolio_model_returns(settings: dict) -> pd.Series:
    """donchian 集成 × 3 币等权的模型日收益（全历史，净成本）——同期对照用。"""
    store = storeio.store_dir(settings)
    legs = {sym: _family_ens(load_spot_daily(store, sym)) for sym in settings["symbols"]}
    port = pd.concat(legs, axis=1).dropna().mean(axis=1)
    port.index = port.index.normalize()
    return port


def build_phase1_portfolio(settings: dict) -> pd.Series:
    """Phase 1 组合口径的 OOS 日收益序列（冻结基准的原料）。"""
    store = storeio.store_dir(settings)
    legs = [_family_ens(load_spot_daily(store, sym)).iloc[TRAIN_BARS:]
            for sym in settings["symbols"]]
    return pd.concat(legs, axis=1).dropna().mean(axis=1)


# ---------------- 基准冻结 ----------------

def baseline_paths(store: Path) -> tuple[Path, Path]:
    d = store / "paper"
    return d / "baseline.json", d / "baseline_returns.parquet"


def freeze_baseline(settings: dict, force: bool = False) -> dict:
    store = storeio.store_dir(settings)
    jp, rp = baseline_paths(store)
    if jp.exists() and not force:
        raise FileExistsError(f"baseline already frozen at {jp} (use --force to refreeze)")
    port = build_phase1_portfolio(settings)
    meta = {"frozen_at": str(pd.Timestamp.now(tz="UTC")),
            "n_days": int(len(port)),
            "window": [str(port.index[0].date()), str(port.index[-1].date())],
            "mu_d": float(port.mean()), "sd_d": float(port.std(ddof=1)),
            "boot": {"n": BOOT_N, "block": BOOT_BLOCK, "seed": BOOT_SEED}}
    tmp = rp.with_suffix(".parquet.tmp")
    df = port.to_frame("ret")
    df.index.name = "day"
    df.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, rp)
    jtmp = jp.with_suffix(".json.tmp")
    jtmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(jtmp, jp)
    log.info("baseline frozen: %d days %s mu_d=%.5f sd_d=%.5f",
             meta["n_days"], meta["window"], meta["mu_d"], meta["sd_d"])
    return meta


def load_baseline(settings: dict) -> tuple[dict, np.ndarray] | None:
    store = storeio.store_dir(settings)
    jp, rp = baseline_paths(store)
    if not (jp.exists() and rp.exists()):
        return None
    meta = json.loads(jp.read_text(encoding="utf-8"))
    rets = pd.read_parquet(rp)["ret"].to_numpy()
    return meta, rets


def bootstrap_band(rets: np.ndarray, horizon: int,
                   n_boot: int = BOOT_N, block: int = BOOT_BLOCK,
                   seed: int = BOOT_SEED) -> dict:
    """冻结样本的 block bootstrap：n 日累计收益经验分位数（固定种子，可复现）。"""
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


# ---------------- 复盘 ----------------

def build_review(settings: dict) -> tuple[str, dict]:
    store = storeio.store_dir(settings)
    pcfg = settings["paper"]
    led = Ledger.load_or_init(store, float(pcfg["initial_capital_usdt"]),
                              str(pcfg["start_date"]))
    settled = led.equity_series(settled_only=True)          # 只用已结算权益
    stats: dict = {"settled_days": int(len(settled))}

    lines = [f"# Paper 复盘 — {pd.Timestamp.now(tz='UTC').date()}", ""]
    if len(settled) < 1:
        lines.append(f"起始 {pcfg['start_date']}，尚无已结算(settled)权益记录，暂无统计。")
        return "\n".join(lines) + "\n", stats

    # 从起始资金起算，首个结算日的收益也计入
    start_anchor = pd.Timestamp(pcfg["start_date"], tz="UTC") - pd.Timedelta(days=1)
    eq = pd.concat([pd.Series([led.initial_capital], index=[start_anchor]), settled])
    paper_ret = eq.pct_change().dropna()
    n = len(paper_ret)

    model = portfolio_model_returns(settings).reindex(paper_ret.index).fillna(0.0)
    diff = paper_ret - model
    te_ann = float(diff.std(ddof=1)) * math.sqrt(ANN["1d"]) if n > 2 else float("nan")
    cum_paper = float((1 + paper_ret).prod() - 1)
    cum_model = float((1 + model).prod() - 1)

    # 期望带：优先冻结基准 + bootstrap；未冻结 → 滚动估计（标注非正式）
    frozen = load_baseline(settings)
    if frozen:
        meta, rets = frozen
        band = bootstrap_band(rets, max(n, 1))
        band_src = f"冻结基准（{meta['frozen_at'][:10]}，{meta['n_days']} 日样本）block bootstrap"
        in80 = band["p10"] <= cum_paper <= band["p90"]
        in95 = band["p2_5"] <= cum_paper <= band["p97_5"]
    else:
        port = build_phase1_portfolio(settings)
        band = bootstrap_band(port.to_numpy(), max(n, 1))
        band_src = "**非正式**（基准未冻结——先运行 python -m scripts.freeze_baseline）"
        in80 = band["p10"] <= cum_paper <= band["p90"]
        in95 = band["p2_5"] <= cum_paper <= band["p97_5"]

    trades = led.trades_df()
    n_live = int((trades["mode"] == "live").sum()) if len(trades) else 0
    n_catch = int((trades["mode"] == "catchup").sum()) if len(trades) else 0
    fees = float(trades["fee"].sum()) if len(trades) else 0.0

    from ops import incident_log
    inc = incident_log.counts_since(store, str(pcfg["start_date"]))

    weeks = n / 7.0
    stats.update(cum_paper_pct=round(100 * cum_paper, 2),
                 cum_model_pct=round(100 * cum_model, 2),
                 te_ann_pct=round(100 * te_ann, 2) if te_ann == te_ann else None,
                 in_band95=bool(in95), weeks=round(weeks, 1),
                 baseline_frozen=bool(frozen))

    lines += [
        f"窗口：{n} 个已结算日收益（≈{weeks:.1f} 周 / 门槛 6 周）；带基准：{band_src}",
        "",
        "| 指标 | Paper | 模型回放 | 备注 |",
        "|---|---|---|---|",
        f"| 累计收益 | {100*cum_paper:.2f}% | {100*cum_model:.2f}% | 差 {1e4*(cum_paper-cum_model):.0f} bps |",
        f"| 年化 Sharpe | {sharpe(paper_ret):.2f} | {sharpe(model):.2f} | |",
        (f"| 跟踪误差 TE(年化) | {100*te_ann:.2f}% | — | 目标 < 2% |" if te_ann == te_ann
         else "| 跟踪误差 | 样本不足 | — | |"),
        f"| 期望带(80%) | {100*band['p10']:.2f}% ~ {100*band['p90']:.2f}% | 中位 {100*band['p50']:.2f}% | paper {'带内 ✅' if in80 else '出带'} |",
        f"| 期望带(95%) | {100*band['p2_5']:.2f}% ~ {100*band['p97_5']:.2f}% | | {'带内 ✅' if in95 else '**出带 ❌**'} |",
        f"| 成交 | live {n_live} 笔 / catchup {n_catch} 笔 | 费用 ${fees:.2f} | live 占比 {n_live/max(1,n_live+n_catch):.0%} |",
        f"| 运维事故 | P0={inc['P0']} P1={inc['P1']} | P2={inc['P2']} P3={inc['P3']} | 明细 data/store/ops/incidents.parquet |",
        "",
        f"**Gate 进度**：{weeks:.1f}/6 周（自动化稳定运行后正式起算）；95% 带内：{'✅' if in95 else '❌'}；"
        f"TE：{'✅' if te_ann == te_ann and te_ann < 0.02 else '观察中'}；"
        f"P0 事故：{'✅ 0' if inc['P0'] == 0 else '❌ ' + str(inc['P0'])}。",
        "",
        "> 注：本策略为**未经统计认证的研究候选**（Phase 1 修订判定），本模拟盘属探索性验证；"
        "6 周达标也只支持『极小额、可全损』级别的 Phase 3 试点，不构成任何部署背书。",
        "",
        "最近 10 笔成交：",
        "",
        "| day | symbol | side | qty | price | mode |",
        "|---|---|---|---|---|---|",
    ]
    for r in trades.tail(10).itertuples():
        lines.append(f"| {r.day} | {r.symbol} | {r.side} | {r.qty:.6g} | {r.price:.2f} | {r.mode} |")

    return "\n".join(lines) + "\n", stats


def write_review(settings: dict) -> Path:
    report, _ = build_review(settings)
    out_dir = Path(__file__).resolve().parents[1] / "research" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"paper_review_{pd.Timestamp.now(tz='UTC').date()}.md"
    p.write_text(report, encoding="utf-8")
    log.info("paper review -> %s", p)
    return p
