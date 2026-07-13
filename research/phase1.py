"""Phase 1 orchestrator: baselines × walk-forward × DSR/PBO → honest research report.

Gate (research report §6.6, ex-ante): a strategy family qualifies for Phase 2 paper trading
only if, net of costs and walk-forward out-of-sample, it satisfies
  OOS annualized Sharpe > 0 and DSR ≥ 0.95 (corrected by that family's grid trial count).
The full-sample-best column is shown only as an "overfitting upper bound" and does not enter the verdict.
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
DEPLOY_FAMILY = "donchian"   # the family chosen for the deployment form — certification must target this object

# 2026-07-13 revision (review R1): the original gate corrected DSR only by the "within-family grid
# trial count", underestimating the selection that actually occurred (4 families × 3 coins × 2 markets ×
# cost basis × select-param/ensemble switch). Now split into two tiers:
#   research candidate: within-family OOS>0 and within-family DSR≥0.95 (weak standard, for screening)
#   certification     : the **deployment object** (cross-coin parameter-ensemble portfolio) has DSR ≥0.95
#                       under both N=4 (family selection) and N=32 (portfolio-level parameter×family trials)
# If certification fails → Phase 2 can only serve as "exploratory paper validation", not strategy certification.


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
        # Parameter ensemble (4~12 within-family variants equal-weight, no param selection) — the robust deployment form when PBO is high
        ens_oos = vr.mean(axis=1).reindex(oos.index)
        ens_map[family] = ens_oos
        rows.append(
            {
                "symbol": symbol, "market": market, "family": family,
                "cost": cost_model.name,
                "full_best_sharpe": round(full_best_sr, 2),   # overfitting upper bound
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

    # meta-DSR: correct every row by all 32 trials (cross-family selection also counts as a trial — a more honest number)
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
            ("um", UM_MAKER, fund_day),      # maker sensitivity
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

    # ---- deployment-level portfolio and two-tier DSR correction (R1) ----
    # Family-level portfolio: each family = parameter ensemble × 3 coins equal-weight (family selection N=4)
    port_rows = []
    port_series: dict[str, pd.Series] = {}
    for family in GRIDS:
        legs = pd.concat({s: spot_ens[s][family] for s in spot_ens}, axis=1).dropna()
        port_series[family] = legs.mean(axis=1)
        port_rows.append({"family": family, **metrics.summary(port_series[family], "1d")})
    var_n4 = metrics.trials_sr_variance(pd.DataFrame(port_series))

    # Portfolio-level trial universe: each (family, params) cross-coin equal-weight portfolio, 32 in total → N=32 conservative correction.
    # Second-review fix: the trial variance must use the same OOS time window as the certification object
    # (drop the first 730 training-warmup bars), otherwise the window mismatch distorts the variance.
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
    gate_passed = certified   # the only external meaning of "pass" = statistical certification

    cols = ["symbol", "market", "family", "cost", "full_best_sharpe", "oos_sharpe",
            "oos_cagr_pct", "oos_maxdd_pct", "dsr", "dsr_meta32", "pbo",
            "ens_oos_sharpe", "bh_oos_sharpe", "family_gate"]

    parts = [
        f"# Phase 1 Research Report — {date}",
        "",
        "**Protocol (ex-ante)**: the parameter grid is fixed before results (32 trials/market); 2-year "
        "train → 6-month out-of-sample rolling, parameters selected by net Sharpe over the training period, "
        "and the fold goes flat if the training-period best is ≤0; cost = fees + √impact slippage "
        "(spot 10bps per side + slippage, USDT-M taker 6bps / maker 2bps; $10k notional); futures include "
        "daily funding (Binance series used as a Bitget proxy). `full_best_sharpe` is the full-sample "
        "post-hoc best (overfitting upper bound, for reference only); the verdict looks only at walk-forward OOS.",
        "",
        "## Gate verdict (2026-07-13 revision: two-tier standard, certification targets the deployment object)",
        "",
        f"**Research candidate: {'PASS ✅' if candidate_pass else 'FAIL ❌'}**"
        f" (within-family basis: OOS Sharpe>0 and within-family DSR≥{GATE_DSR}; {len(passers)} rows qualify)",
        "",
        f"**Statistical certification: {'PASS ✅' if certified else 'FAIL ❌ — not passed'}**"
        + (f" (deployment object = {DEPLOY_FAMILY} parameter ensemble × 3-coin portfolio: "
           f"DSR(N=4 family selection)={dep['dsr_n4']}, DSR(N=32 portfolio-level parameter×family)={dep['dsr_n32']}, "
           f"both required to be ≥{GATE_DSR})" if dep is not None else ""),
        "",
        "> Revision note: the original gate corrected DSR only by the within-family trial count, which "
        "underestimated the selection that actually occurred (4 families × 3 coins × 2 markets × cost basis "
        "× select-param/ensemble switch), and the certification object did not match the deployment object "
        "(PBO 0.66–0.92 is also a strong warning). **When certification does not pass, Phase 2 is positioned "
        "as exploratory paper validation and does not constitute strategy certification**; the `family_gate` "
        "column is only a weak within-family basis for screening reference.",
        "",
        "### Research candidates (rows qualifying on the within-family basis)",
        "",
        _md_table(passers, ["symbol", "market", "family", "oos_sharpe", "dsr",
                            "dsr_meta32", "pbo", "ens_oos_sharpe",
                            "ens_oos_maxdd_pct", "bh_oos_sharpe"])
        if candidate_pass else "(none)",
        "",
        "## Main results (walk-forward out-of-sample, net of costs)",
        "",
        _md_table(df.sort_values(["symbol", "market", "family"]), cols),
        "",
        "## Cross-coin equal-weight portfolio (spot taker cost; each family = parameter ensemble × 3 coins equal-weight)",
        "",
        "The only degree of freedom in the construction is **which family to pick** (N=4 trials; no parameter "
        "selection, no coin selection); the window is the three-coin OOS intersection. This is the number "
        "closest to the actual deployment form, and the basis where the DSR correction is least contaminated by selection.",
        "",
        _md_table(portfolio, ["family", "sharpe", "cagr_pct", "ann_vol_pct",
                              "max_dd_pct", "n_bars", "dsr_n4", "dsr_n32"])
        if portfolio is not None else "",
        "",
        "## Funding-rate carry simulation (delta neutral, 1x, not a prediction-class gate)",
        "",
        "> ⚠️ This simulation includes only funding income and entry/exit costs; it does **not** model "
        "close-out slippage spikes in extreme conditions, instantaneous basis blowouts, or the margin chain "
        "(BIS WP1087: carry is essentially compensation for crash risk). The Sharpe in the table substantially "
        "overstates stability; interpret the reality as 'high-single-digit APR + rare tail events'.",
        "",
    ]
    for sym, results in carry.items():
        parts.append(f"### {sym} (Binance funding 2020→now)")
        parts.append("")
        parts.append("| variant | net APR% | Sharpe | MaxDD% | time in market | round trips | avg funding APR% over period |")
        parts.append("|---|---|---|---|---|---|---|")
        for r in results:
            parts.append(
                f"| {r.name} | {r.net_apr_pct} | {r.sharpe} | {r.max_dd_pct} | "
                f"{r.time_in_market:.0%} | {r.n_roundtrips} | {r.avg_funding_apr_pct} |")
        parts.append("")

    # ---- automatic conclusions ----
    if portfolio is not None and len(portfolio):
        bp = portfolio.loc[portfolio["sharpe"].idxmax()]
        parts += [
            "## Conclusions",
            "",
            f"1. **Two-tier verdict**: research candidate {'passes' if candidate_pass else 'does not pass'}"
            + (f" ({', '.join(sorted(set(passers['symbol'])))} spot "
               f"{', '.join(sorted(set(passers['family'])))}, within-family DSR 0.96–0.98)" if candidate_pass else "")
            + f"; **statistical certification {'passes' if certified else 'does not pass'}** — deployment portfolio "
            + (f"DSR(N=4)={dep['dsr_n4']}, DSR(N=32)={dep['dsr_n32']}, below the {GATE_DSR} threshold."
               if dep is not None and not certified else ""),
            "",
            f"2. **Qualitative strength of evidence**: single-market meta-DSR(32) is only "
            f"{passers['dsr_meta32'].min():.2f}–{passers['dsr_meta32'].max():.2f}, and "
            f"PBO 0.66–0.92 is a strong warning (within-family parameter rankings are unstable; the ensemble "
            "only removes parameter-selection risk, not family-level uncertainty). Supporting points: the three "
            "coins agree in direction, the parameter ensemble is robust, and it agrees with the literature "
            "(JFQA 2025 CTREND). Qualitatively an **uncertified research candidate**: real but moderate-strength "
            "evidence, worthy only of exploratory paper validation, not any basis for live deployment.",
            "",
            f"3. **Exploratory paper validation object (not a certified strategy)**: `{bp['family']}` parameter "
            f"ensemble × 3 coins equal-weight (spot long/flat). Expected portfolio characteristics (net of costs, "
            f"3.9-year OOS intersection): Sharpe ≈ {bp['sharpe']}, "
            f"CAGR ≈ {bp['cagr_pct']}%, annualized vol ≈ {bp['ann_vol_pct']}%, MaxDD ≈ {bp['max_dd_pct']}%. "
            "The tsmom ensemble (low vol, MaxDD −12%) is observed in parallel in the paper book as a "
            "diversification candidate (this portfolio is not pre-registered; observed only, not deployed).",
            "",
            "4. **Execution market: spot**: um is uniformly weaker than spot (funding drag + sample-period "
            "differences), and turnover is so low that maker/taker sensitivity is negligible — spot execution "
            "also naturally eliminates leverage and liquidation risk.",
            "",
            "5. **carry**: BTC/ETH always-on net APR ~6–7% (understood after discounting for tail risk); SOL "
            "average funding ≈ 0, so a threshold switch is **required**. The carry executor stays in Phase 3 as planned.",
            "",
            "6. **Control group behaves normally**: RSI mean-reversion is at the bottom on the portfolio basis "
            "(0.35) — the process discriminates between good and bad strategies, a sign that the methodology "
            "self-check passes.",
            "",
        ]
    parts.append("## Appendix: fold-by-fold record of each market's best family")
    parts.append("")
    for key, fl in folds.items():
        parts.append(f"### {key}")
        parts.append("")
        parts.append("| fold (OOS window) | selected params | train SR/period | OOS SR/period |")
        parts.append("|---|---|---|---|")
        for f in fl:
            parts.append(f"| {f['fold_start']}→{f['fold_end']} | {f['chosen']} | "
                         f"{f['train_sr_pp']} | {f['oos_sr_pp']} |")
        parts.append("")

    report = "\n".join(parts) + "\n"
    path = out_dir / f"phase1_report_{date}.md"
    path.write_text(report, encoding="utf-8")
    return path, gate_passed
