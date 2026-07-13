"""新闻打分器：LLM 优先（**OpenRouter 接口**调 DeepSeek，2026-07-13 按用户要求切换；
无 OPENROUTER_API_KEY 时退回 DeepSeek 直连），关键词规则兜底 → risk_flags.json。

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


def _parse_llm_json(content: str) -> dict:
    """容错解析：剥掉可能的 ```json 代码围栏后再 json.loads。"""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def _llm_backend() -> tuple[str, str, str, dict] | None:
    """选择 LLM 后端：优先 OpenRouter（用户指定接口），其次 DeepSeek 直连。
    返回 (endpoint, api_key, model, extra_headers) 或 None。"""
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        return ("https://openrouter.ai/api/v1/chat/completions", or_key,
                os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"),
                {"X-Title": "QubitvaleTrading"})
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key:
        return ("https://api.deepseek.com/chat/completions", ds_key,
                os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), {})
    return None


def _score_llm(items: list[dict], endpoint: str, api_key: str, model: str,
               extra_headers: dict | None = None) -> list[dict] | None:
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
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", **(extra_headers or {})},
            json={"model": model, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": numbered}]},
            timeout=90,
        )
        r.raise_for_status()
        parsed = _parse_llm_json(r.json()["choices"][0]["message"]["content"])
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


def _recent_headlines(store: Path, window_hours: int, max_items: int,
                      rss_reserve: int = 0) -> list[dict]:
    """Newest headlines in the window, capped at max_items.

    RSS is the curated, higher-signal source but low-volume; GDELT is a
    high-frequency multilingual firehose whose fresher timestamps otherwise
    evict RSS from the top `max_items`. `rss_reserve` guarantees RSS at least
    min(rss_reserve, RSS-available) slots; GDELT takes the remainder, and either
    source expands into the other's unused slots. Cross-source title duplicates
    keep the RSS copy.
    """
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=window_hours)

    def _load(kind: str, tcol: str, srccol: str) -> pd.DataFrame:
        p = storeio.news_path(store, kind)
        if not p.exists():
            return pd.DataFrame(columns=["title", "source", "t"])
        df = pd.read_parquet(p)
        out = pd.DataFrame({"title": df["title"], "source": df[srccol],
                            "t": pd.to_datetime(df[tcol], utc=True)}).dropna(subset=["title"])
        out = out[out["t"] >= cutoff].sort_values("t", ascending=False)
        return out.drop_duplicates(subset=["title"])

    rss = _load("rss", "published_utc", "source")
    gdelt = _load("gdelt", "seen_utc", "domain")
    if rss.empty and gdelt.empty:
        return []

    reserve = min(max(0, int(rss_reserve)), len(rss))
    gdelt_sel = gdelt.head(max(0, max_items - reserve))
    rss_sel = rss.head(max_items - len(gdelt_sel))          # RSS fills reserve + any GDELT shortfall
    combined = pd.concat([rss_sel, gdelt_sel], ignore_index=True)  # RSS first -> wins title dedup
    combined = combined.drop_duplicates(subset=["title"])
    combined = combined.sort_values("t", ascending=False).head(max_items)
    return [{"title": str(r.title)[:200], "source": str(r.source),
             "t": str(r.t)} for r in combined.itertuples()]


def refresh_risk_flags(settings: dict) -> dict:
    """打分近 window_hours 新闻 → data/store/intel/risk_flags.json。"""
    load_env()
    cfg = settings.get("intel", {})
    store = storeio.store_dir(settings)
    items = _recent_headlines(store, int(cfg.get("scorer_window_hours", 48)),
                              int(cfg.get("scorer_max_items", 80)),
                              int(cfg.get("scorer_rss_reserve", 40)))
    scored: list[dict] = []
    if items:
        backend = _llm_backend()
        if backend:
            endpoint, api_key, model, extra = backend
            scored = _score_llm(items, endpoint, api_key, model, extra) or []
            for s in scored:                      # 审计：记录实际使用的模型
                if s.get("scorer") == "llm":
                    s["scorer"] = f"llm:{model}"
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
    tmp = out_dir / "risk_flags.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_dir / "risk_flags.json")

    # R6: 历史归档（append-only）——审计"某日为何放行/拦截"，catchup 也能按时点取旗
    hist_row = pd.DataFrame([{
        "generated_at": payload["generated_at"], "scorer": payload["scorer"],
        "n_scored": payload["n_scored"],
        "market_neg_severity": payload["market_neg_severity"],
        **{f"neg_{a}": neg_sev[a] for a in ASSETS},
    }])
    hp = out_dir / "risk_flags_history.parquet"
    hist = pd.read_parquet(hp) if hp.exists() else None
    hist = pd.concat([hist, hist_row], ignore_index=True) if hist is not None else hist_row
    htmp = out_dir / "risk_flags_history.parquet.tmp"
    hist.to_parquet(htmp, index=False)
    os.replace(htmp, hp)

    log.info("risk_flags: %d scored via %s, neg_sev=%s mkt=%s",
             payload["n_scored"], payload["scorer"], neg_sev, market_neg)
    return payload


def load_risk_flags(settings: dict) -> dict:
    """当前旗 + 时效标注（R6：TTL 过期由调用方按'状态未知'保守处理）。"""
    p = storeio.store_dir(settings) / "intel" / "risk_flags.json"
    if not p.exists():
        return {"asset_neg_severity": {}, "market_neg_severity": 0, "stale": True,
                "age_hours": None}
    with open(p, "r", encoding="utf-8") as f:
        flags = json.load(f)
    ttl = float(settings.get("intel", {}).get("risk_flags_ttl_hours", 24))
    try:
        age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(flags["generated_at"])) \
            / pd.Timedelta(hours=1)
    except (KeyError, ValueError):
        age_h = None
    flags["age_hours"] = round(age_h, 1) if age_h is not None else None
    flags["stale"] = bool(age_h is None or age_h > ttl)
    return flags


def load_flags_asof(settings: dict, ts: pd.Timestamp, max_age_hours: float = 24) -> dict | None:
    """历史归档中 ≤ts 且未过期的最近一条（catchup 回放风控用）；无则 None。"""
    hp = storeio.store_dir(settings) / "intel" / "risk_flags_history.parquet"
    if not hp.exists():
        return None
    h = pd.read_parquet(hp)
    gen = pd.to_datetime(h["generated_at"], utc=True, format="mixed")
    m = (gen <= ts) & (gen >= ts - pd.Timedelta(hours=max_age_hours))
    if not m.any():
        return None
    r = h[m].iloc[-1]
    return {"asset_neg_severity": {a: int(r[f"neg_{a}"]) for a in ASSETS},
            "market_neg_severity": int(r["market_neg_severity"]), "stale": False}
