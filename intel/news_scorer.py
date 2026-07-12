"""新闻打分器：LLM（DeepSeek，OpenAI 兼容）优先，关键词规则兜底 → risk_flags.json。

角色定位（调研报告 §3.2 的直接推论）：LLM 只做**信息结构化传感器**——
把标题分类成 {category, direction, severity, assets}，供风控规则消费；
不产生任何交易信号。无 DEEPSEEK_API_KEY 或调用失败时自动退化为关键词规则，
保证管线在任何环境都能跑（降级会记录在输出里）。
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import requests

from data import storeio
from data.collectors.common import load_env

log = logging.getLogger("qvt.scorer")

ASSETS = ("BTC", "ETH", "SOL")

# 关键词兜底规则：(正则, category, direction, severity)
_RULES = [
    (r"hack|exploit|stolen|drained|breach", "hack", "neg", 4),
    (r"\bban(s|ned|ning)?\b|crackdown|criminal", "reg", "neg", 4),
    (r"sec sues|lawsuit|charges|enforcement|subpoena|indict", "reg", "neg", 3),
    (r"delist", "exchange", "neg", 3),
    (r"insolven|bankrupt|halt(s|ed)? withdraw", "exchange", "neg", 5),
    (r"etf.*(approv|inflow)|approval of.*etf", "etf", "pos", 3),
    (r"etf.*(reject|outflow|denied)", "etf", "neg", 2),
    (r"rate hike|hawkish|inflation (hot|surge|jump)", "macro", "neg", 3),
    (r"rate cut|dovish|inflation (cool|fall|ease)", "macro", "pos", 3),
    (r"liquidat(ed|ion)s?", "market", "neg", 2),
    (r"all[- ]time high|ath\b", "market", "pos", 2),
    (r"crash|plunge|collapse", "market", "neg", 3),
]


def _detect_assets(text: str) -> list[str]:
    t = text.lower()
    found = [a for a, pat in (("BTC", r"bitcoin|\bbtc\b"), ("ETH", r"ethereum|\beth\b"),
                              ("SOL", r"solana|\bsol\b")) if re.search(pat, t)]
    return found or ["ALL"]


def _score_keywords(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        text = it["title"].lower()
        best = None
        for pat, cat, direction, sev in _RULES:
            if re.search(pat, text):
                if best is None or sev > best["severity"]:
                    best = {"category": cat, "direction": direction, "severity": sev}
        if best is None:
            best = {"category": "other", "direction": "neutral", "severity": 1}
        out.append({**it, **best, "assets": _detect_assets(it["title"]), "scorer": "keywords"})
    return out


def _score_llm(items: list[dict], api_key: str, model: str) -> list[dict] | None:
    numbered = "\n".join(f"{i}. [{it['source']}] {it['title']}" for i, it in enumerate(items))
    system = (
        "You classify crypto news headlines for a risk-control system. For EACH numbered "
        "headline return: category ∈ [reg, macro, hack, etf, exchange, market, other]; "
        "direction ∈ [pos, neg, neutral]; severity ∈ 1-5 (5 = existential/systemic, "
        "4 = major risk event, 3 = significant, 2 = notable, 1 = noise); assets ⊆ "
        "[BTC, ETH, SOL, ALL] (use ALL for market-wide). Judge only the headline text. "
        'Output JSON: {"items": [{"i": <index>, "category": "...", "direction": "...", '
        '"severity": <int>, "assets": ["..."]}]}'
    )
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": numbered}]},
            timeout=90,
        )
        r.raise_for_status()
        parsed = json.loads(r.json()["choices"][0]["message"]["content"])
        by_i = {int(x["i"]): x for x in parsed.get("items", [])}
        out = []
        for i, it in enumerate(items):
            x = by_i.get(i)
            if x is None:
                out.append({**it, "category": "other", "direction": "neutral",
                            "severity": 1, "assets": ["ALL"], "scorer": "llm_miss"})
            else:
                out.append({**it,
                            "category": str(x.get("category", "other")),
                            "direction": str(x.get("direction", "neutral")),
                            "severity": int(x.get("severity", 1)),
                            "assets": [a for a in x.get("assets", ["ALL"])
                                       if a in (*ASSETS, "ALL")] or ["ALL"],
                            "scorer": "llm"})
        return out
    except Exception as exc:  # noqa: BLE001 — 任意失败都降级
        log.warning("LLM scorer failed, falling back to keywords: %s", exc)
        return None


def _recent_headlines(store: Path, window_hours: int, max_items: int) -> list[dict]:
    frames = []
    rss_p = storeio.news_path(store, "rss")
    if rss_p.exists():
        df = pd.read_parquet(rss_p)
        frames.append(pd.DataFrame({
            "title": df["title"], "source": df["source"],
            "t": pd.to_datetime(df["published_utc"], utc=True)}))
    g_p = storeio.news_path(store, "gdelt")
    if g_p.exists():
        df = pd.read_parquet(g_p)
        frames.append(pd.DataFrame({
            "title": df["title"], "source": df["domain"],
            "t": pd.to_datetime(df["seen_utc"], utc=True)}))
    if not frames:
        return []
    allf = pd.concat(frames, ignore_index=True).dropna(subset=["title"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=window_hours)
    allf = allf[allf["t"] >= cutoff].sort_values("t", ascending=False)
    allf = allf.drop_duplicates(subset=["title"]).head(max_items)
    return [{"title": str(r.title)[:200], "source": str(r.source),
             "t": str(r.t)} for r in allf.itertuples()]


def refresh_risk_flags(settings: dict) -> dict:
    """打分近 window_hours 新闻 → data/store/intel/risk_flags.json。"""
    load_env()
    cfg = settings.get("intel", {})
    store = storeio.store_dir(settings)
    items = _recent_headlines(store, int(cfg.get("scorer_window_hours", 48)),
                              int(cfg.get("scorer_max_items", 80)))
    scored: list[dict] = []
    if items:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            scored = _score_llm(items, api_key,
                                os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")) or []
        if not scored:
            scored = _score_keywords(items)

    # 资产特定旗与市场级(ALL)旗分开：ALL 旗多为泛匹配噪音（无关协议被黑、
    # 措辞含 ban 的中性新闻），只有 sev≥5 的系统性事件才应影响交易（引擎侧判定）。
    neg_sev: dict[str, int] = {a: 0 for a in ASSETS}
    market_neg = 0
    for s in scored:
        if s["direction"] != "neg":
            continue
        if "ALL" in s["assets"]:
            market_neg = max(market_neg, int(s["severity"]))
        for a in s["assets"]:
            if a in ASSETS:
                neg_sev[a] = max(neg_sev[a], int(s["severity"]))

    payload = {
        "generated_at": str(pd.Timestamp.now(tz="UTC")),
        "n_scored": len(scored),
        "scorer": scored[0]["scorer"] if scored else "none",
        "asset_neg_severity": neg_sev,
        "market_neg_severity": market_neg,
        "top_negative": sorted(
            [s for s in scored if s["direction"] == "neg"],
            key=lambda s: -s["severity"])[:10],
    }
    out_dir = store / "intel"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "risk_flags.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log.info("risk_flags: %d scored via %s, neg_sev=%s",
             payload["n_scored"], payload["scorer"], neg_sev)
    return payload


def load_risk_flags(settings: dict) -> dict:
    p = storeio.store_dir(settings) / "intel" / "risk_flags.json"
    if not p.exists():
        return {"asset_neg_severity": {}}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
