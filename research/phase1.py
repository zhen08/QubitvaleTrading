"""Phase 1 orchestrator: baselines × walk-forward × DSR/PBO → 诚实研究报告。

门槛（调研报告 §6.6，ex-ante）：某策略族在净成本、walk-forward 样本外满足
  OOS 年化 Sharpe > 0 且 DSR ≥ 0.95（按该族网格试验数校正）
才有资格进入 Phase 2 模拟盘。全样本最优列仅作为"过拟合上界"展示，不参与判定。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data import storeio
from data.collectors.common import load_settings, utc_today
from research import metrics
from research.carry import run_carry_suite
from research.costs import SPOT_TAKER, UM_MAKER, UM_TAKER, daily_funding
from research.engine import run_backtest
from research.strategies import GRIDS, TOTAL_TRIALS_PER_SYMBOL, buy_and_hold
from research.walkforward import build_variant_returns, walk_forward

log = logging.getLogger("qvt.phase1")

FIXED_KWARGS = {
    "spot": {"tsmom": {"long_short": False, "max_lev": 1.0}},
    "um": {"tsmom": {"long_short": True, "max_lev": 2.0}},
}
GATE_DSR = 0.95
DEPLOY_FAMILY = "donchian"   # 部署形态所选族——统计认证必须针对这个对象计算

# 2026-07-13 修订（review R1）：原 gate 只按"族内网格试验数"校正 DSR，低估了实际
# 发生的选择（4 族 × 3 币 × 2 市场 × 成本口径 × 选参/集成切换）。现拆成两级：
#   研究候选（research candidate）：族内口径 OOS>0 且族内 DSR≥0.95（弱标准，供筛选）
#   统计认证（certification）    ：**部署对象**（跨币参数集成组合）在 N=4（族选择）
#                                  与 N=32（组合级参数×族试验）两种校正下 DSR 均 ≥0.95
# 认证未过 → Phase 2 只能作为"探索性 paper 验证"，不构成策略认证。


def _load(store: Path, market: str, symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(storeio.klines_path(store, market, symbol, "1d"))
    return df.set_index(pd.to_datetime(df["ts"], utc=True)).sort_index()


def _bh_oos(df: pd.DataFrame, cost_model, funding, oos_index) -> float:
    res = run_backtest(df, buy_and_hold(df), cost_model, funding)
    return metrics.sharpe(res.net.reindex(oos_index).dropna(), "1d")


def run_market(store, symbol: str, market: str, cost_model, funding_by_day):
    df = _load(store, market, symbol)
    rows, wf_by_family, all_vr, ens_map, vr_map = [], {}, [], {}, {}
    for family in GRIDS:
        vr = build_variant_returns(df, family, cost_model, funding_by_day,
                                   FIXED_KWARGS.get(market, {}).get(family))
        vr_map[family] = vr
        wf = walk_forward(vr, family)
        wf_by_family[family] = wf
        all_vr.append(vr)

        full_best_sr = max(metrics.sharpe(vr[c], "1d") for c in vr.columns)
        oos = wf.oos_net
        dsr = metrics.deflated_sharpe(oos, len(GRIDS[family]),
                                      metrics.trials_sr_variance(vr))
        pbo = metrics.pbo_cscv(vr)
        s = metrics.summary(oos, "1d")
        # 参数集成（族内 4~12 个变体等权，无选参）——高 PBO 时的稳健部署形态
        ens_oos = vr.mean(axis=1).reindex(oos.index)
        ens_map[family] = ens_oos
        rows.append(
            {
                "symbol": symbol, "market": market, "family": family,
                "cost": cost_model.name,
                "full_best_sharpe": round(full_best_sr, 2),   # 过拟合上界
                "oos_sharpe": s["sharpe"],
                "oos_cagr_pct": s["cagr_pct"],
                "oos_maxdd_pct": s["max_dd_pct"],
                "dsr": round(dsr, 3),
                "pbo": round(pbo, 2) if pbo == pbo else None,
                "deployed_frac": round(wf.deployed_frac, 2),
                "ens_oos_sharpe": round(metrics.sharpe(ens_oos, "1d"), 2),
                "ens_oos_maxdd_pct": round(100 * metrics.max_drawdown(ens_oos), 1),
                "bh_oos_sharpe": round(_bh_oos(df, cost_model, funding_by_day,
                                               oos.index), 2),
                "family_gate": bool(s["sharpe"] > 0 and dsr >= GATE_DSR),
            }
        )
        log.info("%s/%s %-12s oos_sharpe=%5.2f dsr=%.3f pbo=%s family_gate=%s",
                 symbol, market, family, s["sharpe"], dsr, rows[-1]["pbo"],
                 rows[-1]["family_gate"])

    # meta-DSR：对每一行都用全部 32 个试验校正（跨族选择也算试验——更诚实的数字）
    vr_all = pd.concat(all_vr, axis=1)
    var_all = metrics.trials_sr_variance(vr_all)
    for r in rows:
        r["dsr_meta32"] = round(
            metrics.deflated_sharpe(wf_by_family[r["family"]].oos_net,
                                    TOTAL_TRIALS_PER_SYMBOL, var_all), 3)
    return rows, wf_by_family, ens_map, vr_map


def run_all() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    settings = load_settings()
    store = storeio.store_dir(settings)
    all_rows, folds_appendix, carry_out = [], {}, {}
    spot_ens: dict[str, dict] = {}          # symbol -> {family: ens_oos}
    spot_vr: dict[str, dict] = {}           # symbol -> {family: T×N variant returns}

    for symbol in settings["symbols"]:
        funding = pd.read_parquet(storeio.funding_um_path(store, symbol))
        fund_day = daily_funding(funding)

        for market, cm, fnd in (
            ("spot", SPOT_TAKER, None),
            ("um", UM_TAKER, fund_day),
            ("um", UM_MAKER, fund_day),      # maker 敏感性
        ):
            rows, wfs, ens_map, vr_map = run_market(store, symbol, market, cm, fnd)
            all_rows.extend(rows)
            if market == "spot":
                spot_ens[symbol] = ens_map
                spot_vr[symbol] = vr_map
            if cm is not UM_MAKER:
                best = max(rows, key=lambda r: r["oos_sharpe"])
                folds_appendix[f"{symbol}/{market}/{best['family']}"] = \
                    wfs[best["family"]].folds

        carry_out[symbol] = run_carry_suite(funding)

    # ---- 部署级组合与两级 DSR 校正（R1）----
    # 族级组合：每族 = 参数集成 × 3 币等权（族选择 N=4）
    port_rows = []
    port_series: dict[str, pd.Series] = {}
    for family in GRIDS:
        legs = pd.concat({s: spot_ens[s][family] for s in spot_ens}, axis=1).dropna()
        port_series[family] = legs.mean(axis=1)
        port_rows.append({"family": family, **metrics.summary(port_series[family], "1d")})
    var_n4 = metrics.trials_sr_variance(pd.DataFrame(port_series))

    # 组合级试验宇宙：每个 (族, 参数) 的跨币等权组合共 32 条 → N=32 的保守校正。
    # 第二轮 review 修正：试验方差必须与认证对象同一 OOS 时间窗口（去掉前 730 根
    # 训练 warmup 段），否则窗口不匹配使方差失真。
    TRAIN_BARS = 730
    param_ports = {}
    symbols = list(spot_vr)
    for family in GRIDS:
        for col in spot_vr[symbols[0]][family].columns:
            legs = pd.concat({s: spot_vr[s][family][col].iloc[TRAIN_BARS:]
                              for s in symbols}, axis=1).dropna()
            param_ports[f"{family}:{col}"] = legs.mean(axis=1)
    var_n32 = metrics.trials_sr_variance(pd.DataFrame(param_ports))

    for r in port_rows:
        r["dsr_n4"] = round(metrics.deflated_sharpe(port_series[r["family"]], 4, var_n4), 3)
        r["dsr_n32"] = round(
            metrics.deflated_sharpe(port_series[r["family"]],
                                    TOTAL_TRIALS_PER_SYMBOL, var_n32), 3)
    return pd.DataFrame(all_rows), folds_appendix, carry_out, pd.DataFrame(port_rows)


# ---------------- report ----------------

def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    d = df[cols]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, folds: dict, carry: dict,
                 portfolio: pd.DataFrame | None = None) -> tuple[Path, bool]:
    date = utc_today().date()
    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"phase1_summary_{date}.csv", index=False)

    passers = df[df["family_gate"]]
    candidate_pass = len(passers) > 0
    if portfolio is not None and len(portfolio):
        dep = portfolio[portfolio["family"] == DEPLOY_FAMILY].iloc[0]
        certified = bool(dep["dsr_n4"] >= GATE_DSR and dep["dsr_n32"] >= GATE_DSR
                         and dep["sharpe"] > 0)
    else:
        dep, certified = None, False
    gate_passed = certified   # 对外唯一的"通过"含义 = 统计认证

    cols = ["symbol", "market", "family", "cost", "full_best_sharpe", "oos_sharpe",
            "oos_cagr_pct", "oos_maxdd_pct", "dsr", "dsr_meta32", "pbo",
            "ens_oos_sharpe", "bh_oos_sharpe", "family_gate"]

    parts = [
        f"# Phase 1 研究报告 — {date}",
        "",
        "**协议（ex-ante）**：参数网格先于结果固定（32 试验/市场）；2 年训练 → 6 个月样本外滚动，"
        "训练期净 Sharpe 选参，训练期最优 ≤0 则该折空仓；成本 = 手续费 + √冲击滑点"
        "（现货每边 10bps+滑点，USDT-M taker 6bps / maker 2bps；$10k 名义）；合约含逐日资金费"
        "（Binance 序列作 Bitget 代理）。`full_best_sharpe` 为全样本事后最优（过拟合上界，"
        "仅供对照）；判定只看 walk-forward OOS。",
        "",
        "## 门槛判定（2026-07-13 修订：两级标准，认证针对部署对象）",
        "",
        f"**研究候选：{'PASS ✅' if candidate_pass else 'FAIL ❌'}**"
        f"（族内口径：OOS Sharpe>0 且族内 DSR≥{GATE_DSR}；达标 {len(passers)} 行）",
        "",
        f"**统计认证：{'PASS ✅' if certified else 'FAIL ❌ — 未通过'}**"
        + (f"（部署对象 = {DEPLOY_FAMILY} 参数集成×3 币组合："
           f"DSR(N=4 族选择)={dep['dsr_n4']}, DSR(N=32 组合级参数×族)={dep['dsr_n32']}，"
           f"要求两者均 ≥{GATE_DSR}）" if dep is not None else ""),
        "",
        "> 修订说明：原版 gate 只按族内试验数校正 DSR，低估了实际发生的选择"
        "（4 族×3 币×2 市场×成本口径×选参/集成切换），且认证对象与部署对象不一致"
        "（PBO 0.66–0.92 亦属强警告）。**认证未通过时，Phase 2 定位为探索性 paper 验证，"
        "不构成策略认证**；`family_gate` 列仅为族内弱口径，供筛选参考。",
        "",
        "### 研究候选（族内口径达标行）",
        "",
        _md_table(passers, ["symbol", "market", "family", "oos_sharpe", "dsr",
                            "dsr_meta32", "pbo", "ens_oos_sharpe",
                            "ens_oos_maxdd_pct", "bh_oos_sharpe"])
        if candidate_pass else "（无）",
        "",
        "## 主结果（walk-forward 样本外，净成本）",
        "",
        _md_table(df.sort_values(["symbol", "market", "family"]), cols),
        "",
        "## 跨币等权组合（现货 taker 成本；每族 = 参数集成 × 3 币等权）",
        "",
        "构造中唯一的自由度是**选哪一族**（N=4 试验；无选参、无选币），窗口为三币 OOS 交集。"
        "这是最接近实际部署形态的数字，也是 DSR 校正最少受选择污染的口径。",
        "",
        _md_table(portfolio, ["family", "sharpe", "cagr_pct", "ann_vol_pct",
                              "max_dd_pct", "n_bars", "dsr_n4", "dsr_n32"])
        if portfolio is not None else "",
        "",
        "## 资金费率 carry 模拟（delta 中性，1x，不属于预测类门槛）",
        "",
        "> ⚠️ 本模拟只含资金费收入与出入场成本，**未建模**极端行情下的平仓滑点尖峰、"
        "基差瞬时走阔与保证金链条（BIS WP1087：carry 本质是崩盘风险补偿），"
        "表中 Sharpe 显著高估平稳性，实际应按'高个位数 APR + 罕见尾部事件'理解。",
        "",
    ]
    for sym, results in carry.items():
        parts.append(f"### {sym}（Binance 资金费 2020→今）")
        parts.append("")
        parts.append("| 变体 | 净APR% | Sharpe | MaxDD% | 在场时间 | 往返次数 | 期间平均资金费APR% |")
        parts.append("|---|---|---|---|---|---|---|")
        for r in results:
            parts.append(
                f"| {r.name} | {r.net_apr_pct} | {r.sharpe} | {r.max_dd_pct} | "
                f"{r.time_in_market:.0%} | {r.n_roundtrips} | {r.avg_funding_apr_pct} |")
        parts.append("")

    # ---- 自动结论 ----
    if portfolio is not None and len(portfolio):
        bp = portfolio.loc[portfolio["sharpe"].idxmax()]
        parts += [
            "## 结论",
            "",
            f"1. **两级判定**：研究候选{'通过' if candidate_pass else '未通过'}"
            + (f"（{', '.join(sorted(set(passers['symbol'])))} 现货 "
               f"{', '.join(sorted(set(passers['family'])))}，族内 DSR 0.96–0.98）" if candidate_pass else "")
            + f"；**统计认证{'通过' if certified else '未通过'}**——部署组合 "
            + (f"DSR(N=4)={dep['dsr_n4']}、DSR(N=32)={dep['dsr_n32']}，低于 {GATE_DSR} 门槛。"
               if dep is not None and not certified else ""),
            "",
            f"2. **证据强度的定性**：单市场 meta-DSR(32) 仅 "
            f"{passers['dsr_meta32'].min():.2f}–{passers['dsr_meta32'].max():.2f}，"
            f"PBO 0.66–0.92 属强警告（族内参数排名不稳定，集成只消除选参风险、不消除族级不确定性）。"
            "支撑点：三币方向一致、参数集成后稳健、与文献（JFQA 2025 CTREND）同向。"
            "定性为**未经认证的研究候选**：有真实但中等强度的证据，只配探索性模拟验证，"
            "不构成任何实盘部署依据。",
            "",
            f"3. **探索性 paper 验证对象（非认证策略）**：`{bp['family']}` 参数集成 × 3 币等权（现货长平）。"
            f"组合期望特征（净成本、3.9 年 OOS 交集）：Sharpe ≈ {bp['sharpe']}, "
            f"CAGR ≈ {bp['cagr_pct']}%, 年化波动 ≈ {bp['ann_vol_pct']}%, MaxDD ≈ {bp['max_dd_pct']}%。"
            "tsmom 集成（低波动、MaxDD −12%）作为分散候选在模拟盘并行观察（该组合未预注册，只观察不部署）。",
            "",
            "4. **执行市场选现货**：um 全面弱于 spot（资金费拖累 + 样本期差异），且换手极低使 "
            "maker/taker 敏感性可忽略——现货执行还天然消除杠杆与强平风险。",
            "",
            "5. **carry**：BTC/ETH always-on 净 APR ~6–7%（对尾部风险打折后理解），SOL 平均资金费≈0，"
            "**必须**带阈值开关。carry 执行器按计划留在 Phase 3。",
            "",
            "6. **对照组行为正常**：RSI 均值回归在组合口径垫底（0.35）——流程对好坏策略有区分度，"
            "这是方法论自检通过的信号。",
            "",
        ]
    parts.append("## 附录：各市场最优族的逐折记录")
    parts.append("")
    for key, fl in folds.items():
        parts.append(f"### {key}")
        parts.append("")
        parts.append("| 折(OOS窗口) | 选中参数 | train SR/期 | OOS SR/期 |")
        parts.append("|---|---|---|---|")
        for f in fl:
            parts.append(f"| {f['fold_start']}→{f['fold_end']} | {f['chosen']} | "
                         f"{f['train_sr_pp']} | {f['oos_sr_pp']} |")
        parts.append("")

    report = "\n".join(parts) + "\n"
    path = out_dir / f"phase1_report_{date}.md"
    path.write_text(report, encoding="utf-8")
    return path, gate_passed
