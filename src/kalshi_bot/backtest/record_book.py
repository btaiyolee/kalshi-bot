"""Record Kalshi's live order book and trades for one or more 15-minute series, about once a second.

    kalshi-bot record-book --series KXGOLD15M --series KXBTC15M

Why: strategies that live in market STRUCTURE (book imbalance, queue dynamics, spread and depth
changes, trade flow, volatility of the quote) cannot be tested on candles. The historical API offers
only 1-minute candles, so this recorder builds the missing dataset going forward.

Output: data/book/<series>-<YYYYMMDD>-<HHMMSS of run start>.jsonl.gz (gzip, flushed every poll; read with read_book), one JSON line per poll:
    {"t": poll time, "ticker": ..., "close_ts": ..., "yes": [[price, size], ...], "no": [[price, size], ...],
     "trades": [[created_ts, yes_price, count, taker_side], ...]}
``yes``/``no`` are the best 6 BID levels of each side (a NO bid at p is a YES ask at 1-p), best last.
Only public, unauthenticated endpoints are used (about 2 requests per second per series).
Safe to stop and restart any time; it never places orders."""

from __future__ import annotations

import argparse
import gzip
import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

from kalshi_bot.engine import _ts, pick_active_market
from kalshi_bot.kalshi_client import KalshiClient, KalshiError

OUT = Path("data/book")


def _iso_ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def read_book(path) -> list[dict]:
    """Read a recorder file, tolerating a truncated final gzip block (a recorder that was killed)."""
    rows = []
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except (EOFError, zlib.error, OSError):      # truncated or corrupted tail: keep what was readable
        pass
    return rows


def record(series: list[str], poll_s: float = 1.0, depth: int = 6) -> None:
    client = KalshiClient()
    OUT.mkdir(parents=True, exist_ok=True)
    last_trade_ts = {s: 0.0 for s in series}
    seen_ids: dict[str, set] = {s: set() for s in series}
    current: dict[str, str] = {}
    handles: dict[str, tuple] = {}              # series -> (day, open gzip file)
    run_tag = datetime.now(timezone.utc).strftime("%H%M%S")   # one file per recorder run: never append to a
                                                              # stream a killed earlier run left truncated
    print(f"recording {', '.join(series)} to {OUT}/ every {poll_s}s (Ctrl-C to stop)")
    while True:
        started = time.time()
        for s in series:
            try:
                if current.get(s) is None or started > _ts(current[s + "_close"]) - 1:
                    m = pick_active_market(client.get_markets(series_ticker=s, status="open", limit=200)["markets"], started)
                    if m is None:
                        current[s] = None
                        continue
                    current[s], current[s + "_close"] = m["ticker"], m["close_time"]
                    last_trade_ts[s], seen_ids[s] = _ts(m["close_time"]) - 900, set()
                ticker = current[s]
                raw = client.get_orderbook(ticker, depth=depth).get("orderbook_fp", {})
                new = []
                for tr in client.get_trades(ticker, int(last_trade_ts[s])):
                    if tr["trade_id"] in seen_ids[s]:
                        continue
                    seen_ids[s].add(tr["trade_id"])
                    ts = _iso_ts(tr["created_time"])
                    last_trade_ts[s] = max(last_trade_ts[s], ts)
                    new.append([round(ts, 3), float(tr["yes_price_dollars"]), float(tr["count_fp"]), tr["taker_side"]])
                row = {"t": round(started, 3), "ticker": ticker, "close_ts": _ts(current[s + "_close"]),
                       "yes": [[float(p), float(z)] for p, z in raw.get("yes_dollars", [])],
                       "no": [[float(p), float(z)] for p, z in raw.get("no_dollars", [])],
                       "trades": sorted(new)}
                day = datetime.fromtimestamp(started, timezone.utc).strftime("%Y%m%d")
                if s not in handles or handles[s][0] != day:
                    if s in handles:
                        handles[s][1].close()
                    handles[s] = (day, gzip.open(OUT / f"{s}-{day}-{run_tag}.jsonl.gz", "at"))
                handles[s][1].write(json.dumps(row, separators=(",", ":")) + "\n")
                handles[s][1].flush()               # sync-flush: readable even if the process is killed
            except KalshiError as exc:
                print(f"{s}: {exc}")
                time.sleep(2)
        time.sleep(max(0.0, poll_s - (time.time() - started)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", default=None, help="repeatable; default KXGOLD15M and KXBTC15M")
    ap.add_argument("--poll-s", type=float, default=1.0)
    a = ap.parse_args()
    try:
        record(a.series or ["KXGOLD15M", "KXBTC15M"], a.poll_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
