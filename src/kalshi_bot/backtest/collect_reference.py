"""Download 1-minute PAXG-USD candles from Coinbase (public, no key).

PAXG is a gold-backed token that trades 24/7 and tracks spot gold, so it serves as a
free stand-in for the Pyth gold feed that Kalshi settles on (Pyth now needs an API key).
Output: data/ref_coinbase_1m.csv  (ts, open, high, low, close, volume)
Minutes with no trades are absent from Coinbase's response; consumers forward-fill.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

URL_FMT = "https://api.exchange.coinbase.com/products/{product}/candles"
log = logging.getLogger("collect_reference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-08-27", help="UTC date, YYYY-MM-DD")
    ap.add_argument("--series", default="KXGOLD15M")
    ap.add_argument("--out", default=None, help="default: the series' data dir")
    args = ap.parse_args()
    from kalshi_bot.series import spec_for
    spec = spec_for(args.series)
    out_path = args.out or str(spec.ref_csv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    start = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp())
    end = int(time.time())
    step = 300 * 60  # Coinbase returns at most 300 candles per request
    rows: dict[int, list] = {}
    for lo in range(start, end, step):
        hi = min(lo + step, end)
        for attempt in range(5):
            r = requests.get(URL_FMT.format(product=spec.product), params={"granularity": 60,
                                          "start": datetime.fromtimestamp(lo, timezone.utc).isoformat(),
                                          "end": datetime.fromtimestamp(hi, timezone.utc).isoformat()},
                             timeout=15)
            if r.status_code == 200:
                break
            time.sleep(1 + attempt)
        else:
            raise SystemExit(f"Coinbase kept failing: {r.status_code} {r.text[:100]}")
        for ts, low, high, op, close, vol in r.json():
            rows[ts] = [ts, op, high, low, close, vol]
        time.sleep(0.15)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        w.writerows(rows[k] for k in sorted(rows))
    log.info("saved %d candles to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
