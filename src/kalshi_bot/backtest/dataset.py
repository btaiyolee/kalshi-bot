"""Join Kalshi market-minutes with the PAXG reference price into one table.

One row per (market, minute-boundary t). Everything in a row is knowable at time t:
    x          PAXG move since window open, in dollars
    vol30, m3, m5   trailing 1-minute volatility ($) and 3/5-minute momentum ($)
    tau        minutes left until the window closes
    bid, ask   Kalshi YES best bid / ask at t (close of the minute candle ending at t)
    yes        1 if the market resolved YES (the label)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kalshi_bot.series import SeriesSpec, spec_for


def load_reference_ticks(spec: SeriesSpec) -> pd.Series:
    """Minute-mark prices from RAW trade ticks, exactly as the live bot sees them: the price at
    time t is the last trade at or before t. Unlike the 1-minute candle file, no minute with
    trades is dropped and nothing is forward-filled over a minute that had a trade. Requires
    ``paxg_ticks.csv`` (gold only; see scripts in the research notes)."""
    ticks = pd.read_csv(spec.ticks_csv).sort_values(["ts", "trade_id"])
    pt, pp = ticks.ts.to_numpy(), ticks.price.to_numpy()
    start, end = int(np.ceil(pt[0] / 60) * 60), int(pt[-1] // 60 * 60)
    grid = np.arange(start, end + 1, 60)
    idx = np.searchsorted(pt, grid, side="right") - 1
    return pd.Series(pp[idx], index=pd.RangeIndex(start, end + 1, 60))


def load_reference(spec: SeriesSpec | None = None) -> pd.Series:
    """PAXG price by second-aligned minute, forward-filled where nothing traded.

    Index t holds the price known AT time t (= close of the candle that started at t-60).
    """
    spec = spec or spec_for("KXGOLD15M")
    r = pd.read_csv(spec.ref_csv).set_index("ts")["close"]
    r.index = r.index + 60
    return r.reindex(pd.RangeIndex(r.index.min(), r.index.max() + 60, 60)).ffill()


def build(series: str = "KXGOLD15M", min_quote_width: float = 1.0, source: str = "auto") -> pd.DataFrame:
    """source: "ticks" (raw public trades: what the live bot sees), "candles" (Coinbase 1-minute candle
    file) or "auto" = ticks when a tick file exists. CAUTION: candles are REVISED after a minute closes (only ~50% are final
    2.5 minutes later, 99% after 5+), and for thin products like PAXG they contain trades that are NOT
    on the public tape. Edges measured on them read 1-5 minutes of the future and do not reproduce live. Use candles only for liquid products such as BTC-USD."""
    spec = spec_for(series)
    if source == "auto":
        # Gold must use raw ticks (its candle file contains non-public trades). BTC candles match the
        # public tape; a partial BTC tick download must NOT silently replace the full candle history.
        source = spec.default_source if (spec.default_source == "candles" or spec.ticks_csv.exists()) else "candles"
    markets = pd.read_csv(spec.markets_csv)
    candles = pd.read_csv(spec.candles_csv)
    ref = load_reference_ticks(spec) if source == "ticks" else load_reference(spec)

    df = candles.merge(markets[["ticker", "open_ts", "close_ts", "strike", "settle_value",
                                "result"]], on="ticker")
    df = df[(df.end_ts > df.open_ts) & (df.end_ts < df.close_ts)].copy()  # decision minutes
    df = df.rename(columns={"bid_close": "bid", "ask_close": "ask"})
    df = df[(df.ask - df.bid) < min_quote_width].dropna(subset=["bid", "ask"])
    df = df[(df.bid > 0) & (df.ask < 1)]  # need a two-sided book

    df["ref_now"] = ref.reindex(df.end_ts).to_numpy()
    df["ref_open"] = ref.reindex(df.open_ts).to_numpy()
    df["x"] = df.ref_now - df.ref_open
    # volatility (std of the last 30 one-minute moves) and 3/5-minute momentum, all known at t
    df["vol30"] = ref.diff().rolling(30).std().reindex(df.end_ts).to_numpy()
    df["m3"] = df.ref_now.to_numpy() - ref.reindex(df.end_ts - 180).to_numpy()
    df["m5"] = df.ref_now.to_numpy() - ref.reindex(df.end_ts - 300).to_numpy()
    df["tau"] = (df.close_ts - df.end_ts) / 60.0
    df["yes"] = (df.result == "yes").astype(int)
    df["mid"] = (df.bid + df.ask) / 2
    return df.dropna(subset=["x", "vol30", "m3", "m5"]).sort_values(["open_ts", "end_ts"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    d = build(sys.argv[1] if len(sys.argv) > 1 else "KXGOLD15M")
    print(d.shape, d.open_ts.nunique(), "markets")
    print(d.head())
