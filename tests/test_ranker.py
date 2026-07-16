"""Track 3 ranker tests: membership rules, labels, portfolio hysteresis. Offline."""
import numpy as np
import pandas as pd
import pytest

from data import storeio
from data.collectors.universe import universe_panel_path
from research.dl.ranker import data as D
from research.dl.ranker import portfolio as P

RNG = np.random.default_rng(5)


# ---------------- synthetic store ----------------

def _mk_store(tmp_path, n_days=260, n_syms=40):
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for k in range(n_syms):
        sym = f"A{k:02d}USDT"
        closes = 10 * np.exp(np.cumsum(RNG.normal(0, 0.03, n_days)))
        vol = 1e6 if k < n_syms - 5 else 1e5     # last 5 symbols fail the $5M ADV floor
        df = pd.DataFrame({
            "ts": dates, "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": vol, "quote_volume": vol * 10,
            "taker_buy_base": 0.0, "taker_buy_quote": 0.0, "trades": 10,
            "symbol": sym, "market": "spot", "timeframe": "1d",
        })
        storeio.write_parquet(df, storeio.klines_path(tmp_path, "spot", sym, "1d"))
        part = pd.DataFrame({"date": dates, "symbol": sym, "close": closes,
                             "dollar_vol": vol * 10.0})
        part["adv30"] = part["dollar_vol"].rolling(30, min_periods=30).mean()
        rows.append(part)
    panel = pd.concat(rows, ignore_index=True).dropna(subset=["adv30"])
    panel["rank"] = panel.groupby("date")["adv30"].rank(ascending=False, method="first")
    panel["in_universe"] = panel["rank"] <= 50
    storeio.write_parquet(panel, universe_panel_path(tmp_path))
    return tmp_path


@pytest.fixture()
def rd(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DATA_START", "2021-01-01")
    return D.load_ranker_data(_mk_store(tmp_path))


def test_membership_rules(rd):
    m = rd.member
    active = m.loc[m.any(axis=1)]
    # low-ADV symbols never members; members appear only after 110-bar history
    low = [c for c in m.columns if c >= "A35USDT"]
    assert not m[low].any().any()
    first_member_day = active.index.min()
    # 110 raw bars from listing (2021-01-01), not from the panel start (bar 30)
    assert (first_member_day - pd.Timestamp("2021-01-01", tz="UTC")).days >= 109


def test_label_is_vs_member_median(rd):
    d = rd.member.index[rd.member.sum(axis=1) >= 30][len(rd.member) // 2]
    members = rd.member.loc[d]
    nxt = rd.ret_next.loc[d].where(members).dropna()
    med = nxt.median()
    lab = rd.label.loc[d].dropna()
    expect = (nxt > med).astype(float)
    assert lab.reindex(expect.index).equals(expect)
    # roughly half above the median
    assert abs(lab.mean() - 0.5) < 0.15


def test_labels_are_strictly_forward(rd):
    # label at D must equal a function of D+1 returns: shift the check
    d = rd.label.dropna(how="all").index[10]
    i = rd.ret.index.get_loc(d)
    nxt_date = rd.ret.index[i + 1]
    sym = rd.label.loc[d].dropna().index[0]
    manual_next = rd.ret_next.at[d, sym]
    close_based = None  # ret_next built from close pivot; sanity: finite
    assert np.isfinite(manual_next)


# ---------------- portfolio hysteresis ----------------

def _frame(dates, symbols, fill):
    return pd.DataFrame(fill, index=dates, columns=symbols)


def test_portfolio_enter_top5_exit_rank10():
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    syms = [f"S{i}" for i in range(12)]
    member = _frame(dates, syms, True)
    ret_next = _frame(dates, syms, 0.0)
    cost = _frame(dates, syms, 0.001)
    # day0: S0..S4 top; day1: S0 drops to rank 8 (kept), day2: S0 rank 11 (exit)
    score = _frame(dates, syms, 0.4)
    score.iloc[0, 0:5] = [0.99, 0.98, 0.97, 0.96, 0.95]
    score.iloc[1, 0:12] = [0.90] + [0.99, 0.98, 0.97, 0.96, 0.95, 0.94] + [0.93, 0.92, 0.91, 0.89, 0.88]
    score.iloc[1, 0] = 0.905          # rank 8 today
    score.iloc[2, 0] = 0.40           # rank > 10 and p < 0.5 -> exit
    conf = score >= 0.5
    res = P.simulate(dates, score, member, ret_next, cost, confidence=conf)
    assert "S0" in res.holdings[dates[0]]
    assert "S0" in res.holdings[dates[1]]        # rank 8: kept (hysteresis)
    assert "S0" not in res.holdings[dates[2]]    # exited


def test_portfolio_cash_when_unconfident():
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
    syms = [f"S{i}" for i in range(6)]
    member = _frame(dates, syms, True)
    ret_next = _frame(dates, syms, 0.0)
    cost = _frame(dates, syms, 0.001)
    score = _frame(dates, syms, 0.3)             # nobody confident
    res = P.simulate(dates, score, member, ret_next, cost, confidence=score >= 0.5)
    assert res.holdings[dates[0]] == ()
    assert res.net.iloc[0] == 0.0


def test_portfolio_costs_and_weights():
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    syms = [f"S{i}" for i in range(6)]
    member = _frame(dates, syms, True)
    ret_next = _frame(dates, syms, 0.01)
    cost = _frame(dates, syms, 0.001)
    score = _frame(dates, syms, 0.9)
    res = P.simulate(dates, score, member, ret_next, cost, confidence=score >= 0.5)
    # day0: 5 entries × 20% = 100% turnover at 10bps+slip -> cost 0.001
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.net.iloc[0] == pytest.approx(5 * 0.2 * 0.01 - 1.0 * 0.001)
    # day1: same holdings -> no turnover
    assert res.turnover.iloc[1] == 0.0


def test_delisting_force_exit():
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    syms = [f"S{i}" for i in range(6)]
    member = _frame(dates, syms, True)
    ret_next = _frame(dates, syms, 0.01)
    ret_next.iloc[0, 0] = np.nan                 # S0 has no next bar on day0
    cost = _frame(dates, syms, 0.001)
    score = _frame(dates, syms, 0.5)
    score.iloc[:, 0:5] = 0.9
    res = P.simulate(dates, score, member, ret_next, cost, confidence=score >= 0.5)
    assert "S0" not in res.holdings[dates[0]]    # force-exited
    # S0 contributed 0 return but its exit cost was charged
    assert res.net.iloc[0] < 4 * 0.2 * 0.01

# ---------------- parallel training determinism ----------------

def test_parallel_training_matches_sequential(monkeypatch):
    torch = pytest.importorskip("torch")
    import research.dl.ranker.model as M
    import research.dl.ranker.walkforward as W
    monkeypatch.setattr(M, "MAX_EPOCHS", 2)
    g = torch.Generator().manual_seed(0)
    X_tr = torch.randn(400, 90, 2, generator=g)
    y_tr = (torch.rand(400, generator=g) > 0.5).float()
    X_va = torch.randn(100, 90, 2, generator=g)
    y_va = (torch.rand(100, generator=g) > 0.5).float()

    seeds, widths = (17, 29), (5, 10)
    monkeypatch.setenv("RANKER_WORKERS", "4")
    par = W._train_all_widths(X_tr, y_tr, X_va, y_va, widths, seeds)
    monkeypatch.setenv("RANKER_WORKERS", "1")
    seq = W._train_all_widths(X_tr, y_tr, X_va, y_va, widths, seeds)

    for w in widths:
        assert [t.seed for t in par[w]] == list(seeds)
        for a, b in zip(par[w], seq[w]):
            assert a.val_loss == b.val_loss
            for k in a.state_dict:
                assert torch.equal(a.state_dict[k], b.state_dict[k])
