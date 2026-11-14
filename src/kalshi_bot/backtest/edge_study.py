"""Does a free reference price (PAXG) beat Kalshi's own quotes, after fees?

Walk-forward: fit on the first 60% of markets (by time), evaluate on the last 40%.
Reports (1) how well-calibrated the market mid is, (2) whether adding the reference
improves out-of-sample log-loss, and (3) taker P&L per contract after Kalshi fees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from kalshi_bot.backtest.dataset import build
from kalshi_bot.fees import TAKER_RATE
from kalshi_bot.model import fit, predict_df


def taker_pnl(d: pd.DataFrame, p: np.ndarray, min_edge: float) -> pd.DataFrame:
    """Buy whichever side the model likes if edge net of fee beats ``min_edge``."""
    fee_yes = TAKER_RATE * d.ask * (1 - d.ask)
    fee_no = TAKER_RATE * d.bid * (1 - d.bid)          # NO costs 1-bid, same p(1-p)
    edge_yes = p - d.ask - fee_yes
    edge_no = (1 - p) - (1 - d.bid) - fee_no
    buy_yes, buy_no = edge_yes > min_edge, edge_no > min_edge
    take_yes = buy_yes & (edge_yes >= edge_no)
    take_no = buy_no & ~take_yes
    pnl = np.where(take_yes, d.yes - d.ask - fee_yes, np.where(take_no, (1 - d.yes) - (1 - d.bid) - fee_no, np.nan))
    out = d.assign(pnl=pnl, side=np.where(take_yes, "yes", np.where(take_no, "no", "")))
    return out[out.pnl.notna()]


def summarize(t: pd.DataFrame, label: str) -> None:
    if t.empty:
        print(f"{label:<28} no trades")
        return
    per_mkt = t.groupby("ticker").pnl.sum()          # cluster by market for honest error bars
    n_mkt = t.ticker.nunique()
    se = per_mkt.std(ddof=1) / np.sqrt(n_mkt) * n_mkt / len(t) if n_mkt > 1 else float("nan")
    print(f"{label:<28} trades={len(t):6d} markets={n_mkt:5d} "
          f"avg_pnl/contract={t.pnl.mean():+.4f} (±{1.96 * se:.4f})  hit={np.mean(t.pnl > 0):.3f}")


def main(series: str = "KXGOLD15M") -> None:
    d = build(series)
    cut = d.open_ts.quantile(0.6)
    train, test = d[d.open_ts <= cut], d[d.open_ts > cut]
    print(f"rows: train={len(train)} test={len(test)}  markets: "
          f"{train.ticker.nunique()} / {test.ticker.nunique()}\n")

    print("== 1. Is the market mid calibrated? (test period) ==")
    bins = pd.cut(test.mid, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    print(test.groupby(bins, observed=True).agg(n=("yes", "size"), mid=("mid", "mean"),
                                                realized=("yes", "mean")).round(3), "\n")

    print("== 2. Out-of-sample log-loss ==")
    base = test.mid.clip(1e-3, 1 - 1e-3)
    print(f"market mid only          {log_loss(test.yes, base):.4f}")
    model = fit(train)
    p_test = predict_df(model, test)
    print(f"market mid + PAXG move   {log_loss(test.yes, p_test):.4f}   {model}\n")

    print("== 3. Taker P&L per contract after fees (test period) ==")
    for edge in (0.0, 0.01, 0.02, 0.03, 0.05):
        summarize(taker_pnl(test, p_test, edge), f"min net edge {edge:.2f}")
    print("\n   by minutes remaining (min net edge 0.02):")
    t = taker_pnl(test, p_test, 0.02)
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 15)):
        summarize(t[(t.tau > lo) & (t.tau <= hi)], f"tau in ({lo},{hi}]")


if __name__ == "__main__":
    main()
