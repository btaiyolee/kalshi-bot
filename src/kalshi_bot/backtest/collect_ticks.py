"""Download (and keep extending) the raw public trades of a series' reference product.

    kalshi-bot ticks --series KXGOLD15M             # everything newer than what is on disk
    kalshi-bot ticks --series KXBTC15M --hours 6    # BTC trades constantly: bound the history

Why this matters: Coinbase's candle endpoint contains trades that are NOT on the public trade tape the
live bot reads (for PAXG, 60% of minutes carry extra volume), so features built from candles
can look predictive in a backtest and not exist live. Raw trades are the honest source.
Output is appended; loaders sort by (ts, trade_id)."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from kalshi_bot.series import spec_for

URL = "https://api.exchange.coinbase.com/products/{product}/trades"


def main(series: str = "KXGOLD15M", hours: float | None = None) -> int:
    spec = spec_for(series)
    path: Path = spec.ticks_csv
    known_max = 0
    if path.exists():
        known_max = int(pd.read_csv(path, usecols=["trade_id"]).trade_id.max())
    stop_ts = time.time() - hours * 3600 if hours else 0
    after, pages, new = None, 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh_file = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if fresh_file:
            w.writerow(["trade_id", "ts", "price", "size", "side"])
        while True:
            for attempt in range(6):
                r = requests.get(URL.format(product=spec.product),
                                 params={"limit": 1000, **({"after": after} if after else {})}, timeout=20)
                if r.status_code == 200:
                    break
                time.sleep(1 + attempt)
            else:
                raise SystemExit(f"Coinbase kept failing: {r.status_code}")
            page = r.json()
            if not page:
                break
            rows = [t for t in page if t["trade_id"] > known_max]
            for t in rows:
                ts = datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp()
                w.writerow([t["trade_id"], ts, t["price"], t["size"], t["side"]])
            new += len(rows)
            pages += 1
            oldest = datetime.fromisoformat(page[-1]["time"].replace("Z", "+00:00")).timestamp()
            if len(rows) < len(page) or oldest < stop_ts:      # reached trades we already have (or the time bound)
                break
            after = r.headers.get("cb-after")
            if not after:
                break
            time.sleep(0.12)
    print(f"{series}: added {new} trades ({pages} pages) to {path}")
    return new


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="KXGOLD15M")
    ap.add_argument("--hours", type=float, default=None)
    a = ap.parse_args()
    main(a.series, a.hours)
