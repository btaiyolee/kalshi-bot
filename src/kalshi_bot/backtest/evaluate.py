"""One honest evaluation of a series: is there an edge, and does it survive realistic frictions?

    kalshi-bot evaluate --series KXBTC15M

Windows are split IN TIME ORDER into development (first 60%), holdout (next 20%) and forward
(last 20%). The model is only ever fitted on data BEFORE the block it is tested on:

  dev walk-forward   burn-in (>= ``min_train`` windows, never traded) then expanding folds
  holdout            fit on development, test on the holdout block
  forward            fit on development + holdout, test on the forward block

Every statistic is: first signal per window, taker at the ask, hold to settlement, fees included,
per contract (one trade per window, so trades are independent windows).

PRIMARY DECISION STATISTIC: the pooled out-of-sample mean net P&L per contract with a one-sided 95%
lower confidence bound from a day-block bootstrap, at BOTH the trading threshold (1c) and the
high threshold (4c). The old six gates (per-block bounds, days-positive, ...) are printed as
diagnostics only: they had almost no power (80%-power minimum detectable edge ~ +4c gold / +2.5c BTC)
and the days-positive gate passes 10-30% of the time under the null.

Verdict (never "proven"; live fills, regime change and repeated use of this harness are untested):
  SUPPORTED     lower bound > 0 at the trade threshold AND +1c slippage AND 2 s latency fill keep the mean > 0
                AND both placebo p-values < 0.02
  DEAD          upper bound of the pooled mean < +1c
  INCONCLUSIVE  the interval still includes +2c: the data cannot rule a useful edge in or out
  NOT SUPPORTED otherwise (the bound or a friction/placebo check failed, but a +2c edge is excluded)

Frictions: +1c/+2c slippage and a LATENCY fill (``latency_slip`` per contract, default 0.14c = the
measured mean ask drift over ~2 s). The old "fill at the next minute's quote" (60 s) is informational only.

Placebos apply the exact same fold design (burn-in, dev walk-forward, holdout, forward, pooled at the trade
threshold) to shuffled data: (A) reference features shuffled across windows within minutes-left (kills feature
edges); (B) outcomes shuffled within (quote-price bucket x minutes-left) cells (keeps the quote->outcome
marginal; asks whether model + selection beat what cell-level base rates give by luck).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kalshi_bot.backtest.dataset import build
from kalshi_bot.backtest.stats import clopper_pearson, cluster_bootstrap, cluster_groups
from kalshi_bot.model import fit, predict_df
from kalshi_bot.report import ci

LOT = 5          # typical order size: Kalshi rounds the fee UP to the cent per order, which matters for small lots
LATENCY_SLIP = 0.0014   # per contract, ~2 s of ask drift (journals: +0.14c over 2 s vs +0.98c over 60 s)
MIN_TRAIN = 500         # windows of burn-in before the dev walk-forward may produce test trades


def FEE(x, lot: int = LOT):
    """Taker fee per contract for an order of ``lot`` contracts at price x (rounded up to the cent per order)."""
    x = np.asarray(x, dtype=float)
    return np.ceil(np.round(0.07 * lot * x * (1 - x) * 100, 9)) / 100 / lot
REF_COLS = ["x", "m3", "m5", "vol30"]


def split_windows(d: pd.DataFrame, fractions=(0.6, 0.2, 0.2)) -> dict[str, list[str]]:
    order = d.drop_duplicates("ticker").sort_values("open_ts").ticker.tolist()
    n = len(order)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return {"dev": order[:a], "holdout": order[a:b], "forward": order[b:]}


def first_signals(d: pd.DataFrame, p: np.ndarray, thr: float) -> pd.DataFrame:
    """The first minute in each window where a taker's expected profit after fees >= thr."""
    ey = p - d.ask - FEE(d.ask)
    en = (1 - p) - (1 - d.bid) - FEE(1 - d.bid)
    yes = (ey > thr) & (ey >= en)
    no = (en > thr) & ~yes
    s = d[yes | no].copy()
    s["side"] = np.where(yes[yes | no], "yes", "no")
    return s.sort_values(["open_ts", "end_ts"]).drop_duplicates("ticker")


def pnl(s: pd.DataFrame, price_yes: str = "ask", price_no_bid: str = "bid", slip: float = 0.0) -> pd.Series:
    """Per-contract profit of buying ``side`` at the chosen quote columns (plus slippage)."""
    buy = np.where(s.side == "yes", s[price_yes], 1 - s[price_no_bid]) + slip
    win = np.where(s.side == "yes", s.yes, 1 - s.yes)
    return pd.Series(win - buy - FEE(buy), index=s.index)


def stat(per: pd.Series) -> dict:
    if len(per) < 3:
        return {"n": len(per), "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "win": float("nan")}
    m, lo, hi = ci(per.tolist())
    k = int((per > 0).sum())
    wl, wh = clopper_pearson(k, len(per))          # exact: Wald is invalid at the edges
    return {"n": len(per), "mean": m, "lo": lo, "hi": hi, "win": k / len(per), "win_lo": wl, "win_hi": wh}


def _fmt(s: dict) -> str:
    if not s["n"] or np.isnan(s["mean"]):
        return f"n={s['n']:<5} (too few)"
    return f"n={s['n']:<5} {s['mean']*100:+7.2f}c  [{s['lo']*100:+.1f}, {s['hi']*100:+.1f}]  win {s['win']:.0%}"


def _adjacent(d: pd.DataFrame, col: str, step: int) -> pd.Series:
    """``col`` of the row ``step`` minutes away within the same window, NaN unless that row is EXACTLY
    60*|step| s away (so a gap in the data never yields a non-adjacent quote). Call BEFORE any row filter."""
    g = d.groupby("ticker")
    v = g[col].shift(-step)
    dt = g["end_ts"].shift(-step) - d["end_ts"]
    return v.where(dt == 60 * step)


def with_next_quote(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["open_ts", "end_ts"]).copy()
    for c in ("ask", "bid"):
        d[c + "_n"] = _adjacent(d, c, 1)
    return d


def stale(d: pd.DataFrame) -> pd.DataFrame:
    """The same rows, but the reference-price features are one (adjacent) minute old."""
    d = d.sort_values(["open_ts", "end_ts"]).copy()
    for c in REF_COLS:
        d[c] = _adjacent(d, c, -1)
    return d.dropna(subset=REF_COLS)


def out_of_sample(d: pd.DataFrame, thr: float = 0.0, splits: dict | None = None, folds: int = 4,
                  min_train: int = MIN_TRAIN):
    """Predictions for dev (walk-forward), holdout and forward, each from a model that never saw it.
    Dev walk-forward: the first ``burn`` windows are only ever training data, the rest is cut into
    ``folds`` expanding folds. ``burn`` = min_train, capped at 40% of the dev windows for small datasets."""
    splits = splits or split_windows(d)
    dev_order = splits["dev"]
    burn = max(1, min(min_train, int(0.4 * len(dev_order))))
    rest = np.array_split(np.arange(burn, len(dev_order)), folds)
    dev_rows, seen = [], burn
    for blk in rest:
        if len(blk) == 0:
            continue
        train = d[d.ticker.isin(dev_order[:seen])]
        test = d[d.ticker.isin([dev_order[i] for i in blk])]
        dev_rows.append(test.assign(p=predict_df(fit(train), test)))
        seen += len(blk)
    parts = {"dev": pd.concat(dev_rows) if dev_rows else d.iloc[:0].assign(p=np.zeros(0))}
    dev = d[d.ticker.isin(splits["dev"])]
    ho = d[d.ticker.isin(splits["holdout"])]
    fw = d[d.ticker.isin(splits["forward"])]
    parts["holdout"] = ho.assign(p=predict_df(fit(dev), ho))
    parts["forward"] = fw.assign(p=predict_df(fit(pd.concat([dev, ho])), fw))
    return parts


def pooled_trades(parts: dict, thr: float) -> tuple[dict, pd.DataFrame]:
    firsts = {k: first_signals(v, v.p.to_numpy(), thr) for k, v in parts.items()}
    return firsts, pd.concat(firsts.values())


def trade_days(trades: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(trades.end_ts, unit="s", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y%m%d").to_numpy()


def primary(trades: pd.DataFrame, slip: float = 0.0, reps: int = 4000) -> dict:
    """Pooled mean net P&L per contract, day-block bootstrap, one-sided 95% bounds, MDE = 2.8*SE."""
    if not len(trades):
        return cluster_bootstrap([], [])
    return cluster_bootstrap(pnl(trades, slip=slip).to_numpy(), cluster_groups(trade_days(trades)), reps=reps)


def _shuffle_features(sh: pd.DataFrame, rng) -> None:
    tau_bucket = sh.tau.round().to_numpy()
    vals = sh[REF_COLS].to_numpy().copy()
    for b in np.unique(tau_bucket):
        idx = np.where(tau_bucket == b)[0]
        vals[idx] = vals[rng.permutation(idx)]
    sh[REF_COLS] = vals


def _shuffle_outcomes(sh: pd.DataFrame, rng) -> None:
    mid = ((sh.bid + sh.ask) / 2).to_numpy()
    cell = pd.Series(list(zip(np.floor(mid * 50).astype(int), sh.tau.round().astype(int))), index=sh.index)
    y = sh.yes.to_numpy().copy()
    for _, idx in cell.groupby(cell).indices.items():
        y[idx] = y[rng.permutation(idx)]
    sh["yes"] = y


def placebo(d: pd.DataFrame, thr: float, observed_mean: float, perms: int, kind: str = "features",
            seed: int = 0, min_train: int = MIN_TRAIN, observed_n: int = 0) -> dict:
    """Run the SAME fold design as the real test on shuffled data and compare the pooled mean at ``thr``
    (the threshold is lowered, down to -3c, until the null trades >= 80% as often as the real run).
    kind='features': shuffle reference features across windows within minutes-left.
    kind='outcomes': shuffle outcomes within (quote-price bucket x minutes-left) cells."""
    rng = np.random.default_rng(seed)
    splits = split_windows(d)
    means = []
    for _ in range(perms):
        sh = d.copy()
        (_shuffle_features if kind == "features" else _shuffle_outcomes)(sh, rng)
        parts = out_of_sample(sh, thr, splits, min_train=min_train)
        _, tr = pooled_trades(parts, thr)
        for t in np.linspace(thr, -0.03, 12)[1:] if len(tr) < 0.8 * observed_n else []:   # trade as often as the real run
            _, tr = pooled_trades(parts, t)
            if len(tr) >= 0.8 * observed_n:
                break
        if len(tr):
            means.append(float(pnl(tr).mean()))
    means = np.array(means)
    if not len(means):
        return {"null_mean": float("nan"), "null_sd": float("nan"), "p_value": 1.0, "perms": 0}
    return {"null_mean": float(means.mean()), "null_sd": float(means.std()),
            "p_value": float((1 + (means >= observed_mean).sum()) / (1 + len(means))), "perms": len(means)}


def _cls(p: dict, trade_thr: float = 0.01) -> str:
    """Statistical band of the interval alone (not the verdict): DEAD / INCONCLUSIVE / NOT SUPPORTED."""
    if np.isnan(p["mean"]):
        return "INCONCLUSIVE"
    if p["hi"] < 0.01:
        return "DEAD"
    if p["hi"] >= 0.02:
        return "INCONCLUSIVE"
    return "NOT SUPPORTED"


_BAND = {"DEAD": "upper bound < +1c", "INCONCLUSIVE": "interval includes +2c", "NOT SUPPORTED": "upper bound in [+1c, +2c)"}


def evaluate(series: str, thr: float = 0.04, perms: int = 200, source: str = "auto",
             live_rules: bool = True, thr_trade: float = 0.01, latency_slip: float = LATENCY_SLIP,
             min_train: int = MIN_TRAIN, data: pd.DataFrame | None = None, verbose: bool = True,
             boot_reps: int = 4000) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    d = build(series, source=source) if data is None else data
    # adjacent-minute columns are computed on the FULL table, before the live-rules filter
    d = with_next_quote(d)
    st_full = stale(d)
    if live_rules:                                   # the bot's own entry filters: spread <= 3c, 0.75 <= minutes left <= 14
        keep = lambda x: x[((x.ask - x.bid) <= 0.03 + 1e-9) & (x.tau >= 0.75) & (x.tau <= 14)]
        d, st_full = keep(d), keep(st_full)
    splits = split_windows(d)
    parts = out_of_sample(d, thr, splits, min_train=min_train)
    say(f"\n=== {series} [{source} features]: {d.ticker.nunique()} windows, {len(d)} market-minutes ===")
    say("windows: dev %d (burn-in %d) / holdout %d / forward %d (in time order)" % (
        len(splits["dev"]), max(1, min(min_train, int(0.4 * len(splits["dev"])))), len(splits["holdout"]), len(splits["forward"])))
    res: dict = {}

    def block(t: float):
        firsts, pooled = pooled_trades(parts, t)
        return firsts, pooled, primary(pooled, reps=boot_reps)

    say("\nPRIMARY: pooled out-of-sample net P&L per contract, day-block bootstrap, one-sided 95% bounds")
    for name, t in (("trade", thr_trade), ("high", thr)):
        firsts, pooled, pr = block(t)
        res[f"primary_{name}"], res[f"pooled_{name}"], res[f"firsts_{name}"] = pr, pooled, firsts
        say(f"  threshold {t*100:>4.0f}c ({name:<5}) n={pr['n']:<5} {pr['mean']*100:+6.2f}c  one-sided [{pr['lo']*100:+.2f}, {pr['hi']*100:+.2f}]"
            f"   MDE(2.8*SE)={pr['mde']*100:.2f}c   [{_BAND[_cls(pr, t)]}]")
    firsts, pooled, pr = res["firsts_trade"], res["pooled_trade"], res["primary_trade"]

    say(f"\nper-block diagnostics at {thr_trade*100:.0f}c (per-window Wald; low power, not decisive):")
    for name in ("dev", "holdout", "forward"):
        res[name] = stat(pnl(firsts[name]))
        say(f"  {name:<10} {_fmt(res[name])}")
    res["pooled"] = stat(pnl(pooled))
    say(f"  {'pooled':<10} {_fmt(res['pooled'])}   win-rate exact 95% [{res['pooled'].get('win_lo', float('nan')):.0%}, {res['pooled'].get('win_hi', float('nan')):.0%}]")

    say(f"\nfrictions on the pooled out-of-sample trades at {thr_trade*100:.0f}c (day-block bootstrap):")
    fr = {}
    for label, slip in ((f"latency fill (+{latency_slip*100:.2f}c, ~2s)", latency_slip), ("+1c slippage", 0.01), ("+2c slippage", 0.02)):
        fr[label] = primary(pooled, slip=slip, reps=boot_reps)
        say(f"  {label:<26} {fr[label]['mean']*100:+6.2f}c  [{fr[label]['lo']*100:+.2f}, {fr[label]['hi']*100:+.2f}]")
    res["latency"], res["slip1"] = fr[f"latency fill (+{latency_slip*100:.2f}c, ~2s)"], fr["+1c slippage"]
    pn = pooled.dropna(subset=["ask_n", "bid_n"])
    res["t1"] = stat(pnl(pn, "ask_n", "bid_n")) if len(pn) else stat(pd.Series(dtype=float))
    say(f"  [info only] fill at next minute's quote (60 s, harsher than live ~2 s): {_fmt(res['t1'])}")
    ho_fw = st_full[st_full.ticker.isin(splits["holdout"] + splits["forward"])]
    if len(ho_fw):
        fs = first_signals(ho_fw, predict_df(fit(d[d.ticker.isin(splits["dev"])]), ho_fw), thr_trade)
        res["stale"] = stat(pnl(fs))
        say(f"  [info only] reference price 1 min stale: {_fmt(res['stale'])}   (holdout+forward only)")

    days = trade_days(pooled)
    per_day = pnl(pooled).groupby(days).mean()
    res["days_pos"] = float((per_day > 0).mean()) if len(per_day) else float("nan")
    say(f"  [info only] by day: {len(per_day)} days, {res['days_pos']:.0%} positive (noisy: passes 10-30% under the null)")

    res["placebo_features"] = res["placebo_outcomes"] = None
    if perms > 0:
        say(f"\nplacebos ({perms} shuffles each; same fold design, pooled mean at {thr_trade*100:.0f}c, observed {pr['mean']*100:+.2f}c):")
        for kind, desc in (("features", "reference features shuffled across windows"), ("outcomes", "outcomes shuffled within quote-price x minutes-left cells")):
            plc = placebo(d, thr_trade, pr["mean"], perms, kind, min_train=min_train, observed_n=pr["n"])
            res[f"placebo_{kind}"] = plc
            say(f"  {desc}: null mean {plc['null_mean']*100:+.2f}c (sd {plc['null_sd']*100:.2f}c); p = {plc['p_value']:.3f}")
    pf, po = (res[k]["p_value"] if res[k] else 1.0 for k in ("placebo_features", "placebo_outcomes"))
    res["placebo"] = {"p_value": max(pf, po), "p_features": pf, "p_outcomes": po}

    checks = {
        f"primary lower bound > 0 at {thr_trade*100:.0f}c": pr["lo"] > 0,
        "+1c slippage mean > 0": res["slip1"]["mean"] > 0,
        f"latency-fill mean > 0": res["latency"]["mean"] > 0,
        "placebo p < 0.02 (features)": pf < 0.02,
        "placebo p < 0.02 (outcome cells)": po < 0.02,
    }
    res["checks"] = checks
    say("\nVERDICT CHECKS (all must pass for SUPPORTED):")
    for k, v in checks.items():
        say(f"  [{'PASS' if v else 'fail'}] {k}")
    res["primary_high_lo_gt0"] = res["primary_high"]["lo"] > 0
    say(f"  (info) lower bound > 0 at {thr*100:.0f}c: {res['primary_high_lo_gt0']}")
    say("diagnostics (old gates, not decisive): dev lo>0 %s | holdout lo>0 %s | forward mean>0 %s | next-min fill>=+1c %s | days>=60%% %s" % (
        res["dev"]["lo"] > 0, res["holdout"]["lo"] > 0, res["forward"]["mean"] > 0, res["t1"]["mean"] >= 0.01, res["days_pos"] >= 0.6))
    cls = _cls(pr, thr_trade)
    if all(checks.values()):
        verdict = "SUPPORTED by this data (not proven live: fills, regime, selection untested)"
    elif cls == "DEAD":
        verdict = f"DEAD: upper bound {pr['hi']*100:+.2f}c < +1c at the trade threshold"
    elif cls == "INCONCLUSIVE":
        verdict = (f"INCONCLUSIVE: {pr['mean']*100:+.2f}c, interval includes +2c (upper {pr['hi']*100:+.2f}c); "
                   f"minimum detectable edge {pr['mde']*100:.1f}c. This is NOT evidence of no edge")
    else:
        verdict = "NOT SUPPORTED: at least one verdict check failed, though an edge of +2c is excluded"
    res["verdict"] = verdict
    say("VERDICT:", verdict)
    return res


def main(series: str = "KXGOLD15M", thr: float = 0.04, perms: int = 200) -> None:
    evaluate(series, thr, perms)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "KXGOLD15M")
