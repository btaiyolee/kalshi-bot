"""Download settled KXGOLD15M markets and their 1-minute bid/ask candlesticks.

Output (resumable; re-running only fetches what is missing):
    data/markets.csv   one row per market: strike (Pyth price at window start),
                       settlement value (Pyth price at window end), result, times
    data/candles.csv   one row per market-minute: yes bid/ask OHLC, trade price, volume
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from kalshi_bot.kalshi_client import KalshiClient
from kalshi_bot.series import spec_for

log = logging.getLogger("collect_kalshi")
SERIES = "KXGOLD15M"          # default; override with --series
DATA = Path("data")

MARKET_COLS = ["ticker", "open_ts", "close_ts", "strike", "settle_value", "result", "volume"]
CANDLE_COLS = ["ticker", "end_ts", "bid_open", "bid_high", "bid_low", "bid_close",
               "ask_open", "ask_high", "ask_low", "ask_close", "trade_close", "volume"]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _f(x) -> str:
    return "" if x in (None, "") else str(float(x))


def fetch_candles(client: KalshiClient, m: dict, series: str = SERIES) -> list[list]:
    start, end = m["open_ts"], m["close_ts"]
    rows = []
    for c in client.get_candlesticks(series, m["ticker"], start, end, 1):
        bid, ask, px = c.get("yes_bid", {}), c.get("yes_ask", {}), c.get("price", {})
        rows.append([m["ticker"], c["end_period_ts"],
                     _f(bid.get("open_dollars")), _f(bid.get("high_dollars")),
                     _f(bid.get("low_dollars")), _f(bid.get("close_dollars")),
                     _f(ask.get("open_dollars")), _f(ask.get("high_dollars")),
                     _f(ask.get("low_dollars")), _f(ask.get("close_dollars")),
                     _f(px.get("close_dollars")), _f(c.get("volume_fp"))])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--series", default=SERIES)
    args = ap.parse_args()
    series, DATA = args.series, spec_for(args.series).data_dir
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    DATA.mkdir(parents=True, exist_ok=True)
    client = KalshiClient()

    markets, skipped = [], 0
    for m in client.iter_markets(series_ticker=series, status="settled"):
        if m["result"] not in ("yes", "no") or not m.get("expiration_value") \
                or m.get("floor_strike") is None:
            skipped += 1
            continue
        markets.append({"ticker": m["ticker"], "open_ts": _ts(m["open_time"]),
                        "close_ts": _ts(m["close_time"]), "strike": m["floor_strike"],
                        "settle_value": m["expiration_value"], "result": m["result"],
                        "volume": m["volume_fp"]})
    markets.sort(key=lambda m: m["open_ts"])
    with open(DATA / "markets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, MARKET_COLS)
        w.writeheader()
        w.writerows(markets)
    log.info("saved %d markets (%d skipped: unresolved or no strike)", len(markets), skipped)

    candle_path = DATA / "candles.csv"
    done = set()
    if candle_path.exists():
        with open(candle_path) as fh:
            done = {row["ticker"] for row in csv.DictReader(fh)}
    todo = [m for m in markets if m["ticker"] not in done]
    log.info("%d markets already have candles, fetching %d", len(done), len(todo))

    new_file = not candle_path.exists()
    with open(candle_path, "a", newline="") as fh, ThreadPoolExecutor(args.workers) as pool:
        w = csv.writer(fh)
        if new_file:
            w.writerow(CANDLE_COLS)
        start = time.time()
        for i, rows in enumerate(pool.map(lambda m: fetch_candles(client, m, series), todo), 1):
            w.writerows(rows)
            if i % 200 == 0:
                fh.flush()
                log.info("%d/%d markets (%.0fs)", i, len(todo), time.time() - start)
    log.info("done")


if __name__ == "__main__":
    main()
