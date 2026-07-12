"""跟踪复盘：paper 权益 vs 模型回放 vs Phase 1 期望分布带。

三条线：
  paper   —— 账本 settled 权益（真实模拟盘轨迹）
  model   —— 用当前数据把同一策略在 paper 窗口内重放（研究引擎语义、taker 成本）
  band    —— Phase 1 组合口径（donchian 集成 × 3 币 OOS 交集）的 μ/σ 推出的
             期望累计收益带（80% / 95%），gate 要求 paper 落在带内
跟踪误差 TE = std(paper 日收益 − model 日收益)（年化）。
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from data import storeio
from execution.paper.ledger import Ledger
from research.costs import SPOT_TAKER
from research.metrics import ANN, sharpe
from research.walkforward import build_variant_returns
from strategies.donchian_ensemble import load_spot_daily

log = logging.getLogger("qvt.track")

TRAIN_BARS = 730   # 与 Phase 1 协议一致：OOS 从第 730 根起


def _family_ens(df: pd.DataFrame) -> pd.Series:
    vr = build_variant_returns(df, "donchian", SPOT_TAKER)
    return vr.mean(axis=1)


def portfolio_model_returns(settings: dict) -> pd.Series:
    """donchian 集成 × 3 币等权的模型日收益（全历史，净成本）。"""
    store = storeio.store_dir(settings)
    legs = {}
    for sym in settings["symbols"]:
        legs[sym] = _family_ens(load_spot_daily(store, sym))
    port = pd.concat(legs, axis=1).dropna().mean(axis=1)
    port.index = port.index.normalize()
    return port


def phase1_band_params(settings: dict) -> tuple[float, float]:
    """Phase 1 组合口径的日频 μ/σ（OOS 交集窗口，随数据更新重算，口径不变）。"""
    store = storeio.store_dir(settings)
    legs = []
    for sym in settings["symbols"]:
        df = load_spot_daily(store, sym)
        ens = _family_ens(df).iloc[TRAIN_BARS:]
        legs.append(ens)
    port = pd.concat(legs, axis=1).dropna().mean(axis=1)
    return float(port.mean()), float(port.std(ddof=1))


def build_review(settings: dict) -> tuple[str, dict]:
    store = storeio.store_dir(settings)
    pcfg = settings["paper"]
    led = Ledger.load_or_init(store, float(pcfg["initial_capital_usdt"]),
                              str(pcfg["start_date"]))
    eq = led.equity_series()
    settled = eq  # settled + 最新 intraday；日收益只用相邻差
    stats: dict = {"days": int(len(settled))}

    lines = [f"# Paper 复盘 — {pd.Timestamp.now(tz='UTC').date()}", ""]
    if len(settled) < 2:
        lines.append(f"已运行 {len(settled)} 天（起始 {pcfg['start_date']}），"
                     "样本不足两日，暂无对比统计。")
        return "\n".join(lines) + "\n", stats

    paper_ret = settled.pct_change().dropna()
    model = portfolio_model_returns(settings).reindex(paper_ret.index).fillna(0.0)
    diff = paper_ret - model
    te_ann = float(diff.std(ddof=1)) * math.sqrt(ANN["1d"]) if len(diff) > 2 else float("nan")

    mu_d, sd_d = phase1_band_params(settings)
    n = len(paper_ret)
    cum_paper = float((1 + paper_ret).prod() - 1)
    cum_model = float((1 + model).prod() - 1)
    exp_mu = mu_d * n
    band80 = 1.2816 * sd_d * math.sqrt(n)
    band95 = 1.9600 * sd_d * math.sqrt(n)
    in80 = abs(cum_paper - exp_mu) <= band80
    in95 = abs(cum_paper - exp_mu) <= band95

    trades = led.trades_df()
    n_live = int((trades["mode"] == "live").sum()) if len(trades) else 0
    n_catch = int((trades["mode"] == "catchup").sum()) if len(trades) else 0
    fees = float(trades["fee"].sum()) if len(trades) else 0.0

    weeks = n / 7.0
    stats.update(cum_paper_pct=round(100 * cum_paper, 2),
                 cum_model_pct=round(100 * cum_model, 2),
                 te_ann_pct=round(100 * te_ann, 2) if te_ann == te_ann else None,
                 in_band95=bool(in95), weeks=round(weeks, 1))

    lines += [
        f"窗口：{paper_ret.index[0].date()} → {paper_ret.index[-1].date()}（{n} 个日收益，≈{weeks:.1f} 周 / 门槛 6 周）",
        "",
        "| 指标 | Paper | 模型回放 | 备注 |",
        "|---|---|---|---|",
        f"| 累计收益 | {100*cum_paper:.2f}% | {100*cum_model:.2f}% | 差 {1e4*(cum_paper-cum_model):.0f} bps |",
        f"| 年化 Sharpe | {sharpe(paper_ret):.2f} | {sharpe(model):.2f} | |",
        f"| 跟踪误差 TE(年化) | {100*te_ann:.2f}% | — | 目标 < 2% |" if te_ann == te_ann else "| 跟踪误差 | 样本不足 | — | |",
        f"| Phase1 期望带 | {100*(exp_mu-band80):.2f}% ~ {100*(exp_mu+band80):.2f}% (80%) | ±{100*band95:.2f}% (95%) | paper {'在' if in80 else ('在95%带内' if in95 else '**出带**')} |",
        f"| 成交 | live {n_live} 笔 / catchup {n_catch} 笔 | 费用 ${fees:.2f} | live 占比 {n_live/max(1,n_live+n_catch):.0%} |",
        "",
        f"**Gate 进度**：{weeks:.1f}/6 周；95% 带内：{'✅' if in95 else '❌'}；"
        f"TE：{'✅' if te_ann == te_ann and te_ann < 0.02 else '观察中'}。",
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
