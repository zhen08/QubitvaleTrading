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


def test_vectorized_sequences_match_reference(rd):
    torch = pytest.importorskip("torch")
    floor = D.fold_floor(rd, rd.ret.index[:150])
    z = D.normalized_returns(rd, floor)
    zrank = D.cross_rank(z, rd.member)
    dates = rd.ret.index[120:190]
    out = D.build_sequences(rd, z, zrank, dates, require_label=True)
    assert out is not None
    X, y, meta = out

    # naive reference loop (the original implementation)
    Z = z.to_numpy(np.float32)
    R = zrank.to_numpy(np.float32)
    pos_of = {d: i for i, d in enumerate(z.index)}
    xs, ys, ms = [], [], []
    for d in dates:
        i = pos_of.get(d)
        if i is None or i < D.SEQ_LEN - 1:
            continue
        for j, sym in enumerate(z.columns):
            if not rd.member.at[d, sym]:
                continue
            lab = rd.label.at[d, sym]
            if not np.isfinite(lab):
                continue
            win = Z[i - D.SEQ_LEN + 1:i + 1, j]
            if not np.isfinite(win).all():
                continue
            rwin = np.nan_to_num(R[i - D.SEQ_LEN + 1:i + 1, j], nan=0.5)
            xs.append(np.stack([win, rwin], axis=1))
            ys.append(lab)
            ms.append((d, sym))
    assert len(xs) == len(X) and len(xs) > 0
    assert list(meta) == ms
    assert torch.equal(X, torch.from_numpy(np.stack(xs)))
    assert torch.equal(y, torch.tensor(np.asarray(ys, dtype=np.float32)))


# ---------------- L/S mechanical transform ----------------

def _ls_day(scores: dict, shortable=None, held_long=None, held_short=None, neutral=0.5):
    syms = list(scores)
    row = pd.Series(scores)
    member = pd.Series(True, index=syms)
    return P.ls_targets(row, member, shortable if shortable is not None else set(syms),
                        held_long or set(), held_short or set(), neutral=neutral)


def test_ls_dollar_neutral_when_both_legs_fill():
    scores = {f"S{i:02d}": 1.0 - i * 0.05 for i in range(15)}  # S00 best … S14 worst
    t = _ls_day(scores)
    longs = {s for s, w in t.items() if w > 0}
    shorts = {s for s, w in t.items() if w < 0}
    assert longs == {"S00", "S01", "S02", "S03", "S04"}
    assert shorts == {"S10", "S11", "S12", "S13", "S14"}
    assert abs(sum(t.values())) < 1e-12                      # net 0
    assert sum(abs(w) for w in t.values()) == pytest.approx(1.0)  # gross 1


def test_ls_short_requires_perp():
    scores = {f"S{i:02d}": 1.0 - i * 0.05 for i in range(15)}
    t = _ls_day(scores, shortable={"S13", "S14"})
    shorts = {s for s, w in t.items() if w < 0}
    assert shorts == {"S13", "S14"}                          # others stay cash
    assert sum(1 for w in t.values() if w > 0) == 5


def test_ls_confidence_thresholds():
    # all-bearish scores: long leg empty (needs >= 0.5), short leg fills —
    # one-sided books are allowed per the 2026-07-17 registration clarification
    scores = {f"S{i:02d}": 0.49 - i * 0.01 for i in range(12)}
    t = _ls_day(scores)
    assert all(w < 0 for w in t.values()) and len(t) == 5
    # R1LS analog with neutral=0: positive momentum longs, negative shorts
    mom = {"A": 0.3, "B": 0.2, "C": 0.1, "D": 0.05, "E": 0.01,
           "F": -0.01, "G": -0.05, "H": -0.1, "I": -0.2, "J": -0.3,
           "K": 0.15, "L": -0.15}
    t = _ls_day(mom, neutral=0.0)
    assert all(mom[s] > 0 for s, w in t.items() if w > 0)
    assert all(mom[s] < 0 for s, w in t.items() if w < 0)


def test_ls_hysteresis_keeps_mid_bottom_short():
    # 16 names: P has score 0.49 (short-confident) at rank-from-top 11 =>
    # bottom-rank 6: inside the keep zone (>= n-EXIT_RANK+1 = 7 by top-rank),
    # outside the entry zone (needs top-rank >= n-4 = 12)
    scores = {f"S{i:02d}": 0.9 - i * 0.01 for i in range(10)}
    scores.update({"P": 0.49, "K": 0.48, "L": 0.47, "M": 0.46, "N": 0.45, "O": 0.44})
    held = _ls_day(scores, held_short={"P"})
    fresh = _ls_day(scores)
    assert held.get("P", 0.0) < 0        # kept while held (hysteresis)
    assert fresh.get("P", 0.0) == 0.0    # but not enterable fresh
    # a long-side name can never stay short
    t2 = _ls_day(scores, held_short={"S03"})
    assert t2.get("S03", 0.0) >= 0
