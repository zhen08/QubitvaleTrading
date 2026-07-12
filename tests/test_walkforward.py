"""Walk-forward: 折边界无重叠、flat 规则。"""
import numpy as np
import pandas as pd

from research.walkforward import walk_forward


def _vr(n=1000, ncols=4, mu=0.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rng.normal(mu, 0.01, (n, ncols)),
                        index=idx, columns=[f"v{i}" for i in range(ncols)])


def test_fold_boundaries_contiguous_no_overlap():
    vr = _vr(1000)
    wf = walk_forward(vr, "fam", train_bars=200, test_bars=100)
    assert len(wf.folds) == 8
    # OOS 拼接后 = 原始索引第 200 根之后的全部，且无重复
    assert wf.oos_net.index.equals(vr.index[200:])
    assert not wf.oos_net.index.duplicated().any()


def test_flat_rule_when_all_losers():
    vr = _vr(600, mu=-0.002, seed=2)   # 所有变体稳定亏损
    wf = walk_forward(vr, "fam", train_bars=200, test_bars=100)
    assert all(not f["deployed"] for f in wf.folds)
    assert float(wf.oos_net.abs().sum()) == 0.0
