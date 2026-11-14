import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import kalshi_bot.backtest.evaluate as EV  # noqa: E402
from kalshi_bot.backtest.stats import clopper_pearson, cluster_bootstrap, cluster_groups  # noqa: E402
from synth_world import gen  # noqa: E402


def test_bootstrap_ci_covers_known_mean():
    rng = np.random.default_rng(1)
    cover, lows = 0, 0
    for _ in range(60):
        g = np.repeat(np.arange(40), 5)
        v = rng.normal(0.02, 0.5, 200) + np.repeat(rng.normal(0, 0.1, 40), 5)
        r = cluster_bootstrap(v, g, reps=500, alpha=0.05, seed=3)
        cover += r["lo"] <= 0.02 <= r["hi"]            # 90% two-sided band built from the two one-sided bounds
        lows += r["lo"] <= 0.02                          # one-sided 95% lower bound must sit below the truth ~95%
        assert r["mde"] == pytest.approx(2.8 * r["se"])
    assert cover >= 48 and lows >= 52


def test_clopper_pearson_exact():
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-9)   # rule of three-ish, not a point
    assert clopper_pearson(20, 20)[1] == 1.0
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-3) and hi == pytest.approx(0.8129, abs=1e-3)
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_cluster_groups_falls_back_to_blocks():
    days = np.repeat(["a", "b", "c"], 100)
    g = cluster_groups(days, min_groups=20)
    assert len(np.unique(g)) >= 15
    assert len(np.unique(cluster_groups(np.arange(50), min_groups=20))) == 50


def test_burn_in_no_dev_trades_before_min_train():
    d = gen(400, seed=2)
    sp = EV.split_windows(d)
    parts = EV.out_of_sample(d, 0.0, sp, min_train=150)            # dev = 240 windows, burn = min(150, 96)
    burn = min(150, int(0.4 * len(sp["dev"])))
    first_test = set(parts["dev"].ticker)
    assert not first_test & set(sp["dev"][:burn])
    assert len(first_test) == len(sp["dev"]) - burn
    parts = EV.out_of_sample(d, 0.0, sp, min_train=50)
    assert len(set(parts["dev"].ticker)) == len(sp["dev"]) - 50


def test_next_quote_is_adjacent_minute_even_after_filter():
    d = pd.DataFrame({"ticker": ["A"] * 4, "open_ts": 0, "end_ts": [60, 120, 240, 300],   # minute 180 is missing
                      "ask": [.50, .52, .60, .62], "bid": [.49, .51, .59, .61],
                      "x": 1.0, "m3": 1.0, "m5": 1.0, "vol30": 1.0})
    nq = EV.with_next_quote(d)
    assert nq.ask_n.iloc[0] == pytest.approx(.52)
    assert np.isnan(nq.ask_n.iloc[1])                 # next row is 2 minutes away: must not be used
    assert nq.ask_n.iloc[2] == pytest.approx(.62)
    assert np.isnan(EV.stale(d).x.iloc[0:1]).sum() == 0
    assert list(EV.stale(d).end_ts) == [120, 300]    # only rows whose previous minute exists


def test_fee_and_latency_friction_lower_pnl():
    s = pd.DataFrame({"side": ["yes"], "ask": [.5], "bid": [.49], "yes": [1]})
    assert EV.pnl(s, slip=0.0014).iloc[0] < EV.pnl(s).iloc[0]


def _quiet(d, **kw):
    return EV.evaluate("X", data=d, verbose=False, min_train=300, boot_reps=1000, **kw)


def test_null_world_not_supported():
    for seed in (11, 12):
        r = _quiet(gen(1500, lam=0.0, seed=seed), perms=40)
        assert not r["verdict"].startswith("SUPPORTED")


def test_planted_feature_edge_supported():
    r = _quiet(gen(3000, lam=0.8, seed=5), perms=100)
    assert r["primary_trade"]["lo"] > 0.02
    assert r["placebo"]["p_features"] < 0.02 and r["placebo"]["p_outcomes"] < 0.02
    assert r["verdict"].startswith("SUPPORTED")


def test_underpowered_run_labelled_inconclusive():
    r = _quiet(gen(700, lam=0.0, seed=4), perms=0)
    assert "SUPPORTED" not in r["verdict"].split(":")[0] or r["verdict"].startswith("NOT")
    assert r["primary_trade"]["mde"] > 0.01
