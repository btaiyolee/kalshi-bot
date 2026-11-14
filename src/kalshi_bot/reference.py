"""Live reference price (a Coinbase product, PAXG-USD for gold) kept fresh by a background thread.

Why PAXG: Kalshi settles on Pyth's gold price, which now needs a paid API key. PAXG is a
gold-backed token that trades 24/7 and followed Pyth's 15-minute moves with correlation
~0.91 in our data. If you get a Pyth key, add a feed with the same interface here.
"""

from __future__ import annotations

import bisect
import logging
from datetime import datetime
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import requests

log = logging.getLogger(__name__)
API = "https://api.exchange.coinbase.com/products"


class CoinbaseFeed:
    def __init__(self, product: str = "PAXG-USD", poll_s: float = 0.5, keep_s: float = 3600):
        self.product = product
        self._ticker_url = f"{API}/{product}/ticker"
        self._candles_url = f"{API}/{product}/candles"
        self._poll_s, self._keep_s = poll_s, keep_s
        self._ticks: deque[tuple[float, float]] = deque()   # (local receive time, last trade price)
        self._last_trade_ts: float = 0.0                     # exchange timestamp of the newest trade seen
        self._history: dict[int, float] = {}                # minute boundary -> price known then
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="coinbase-feed")
        self._session = requests.Session()

    def start(self) -> None:
        self._backfill()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _backfill(self) -> None:
        """Seed ~5h of minute prices from candles so volatility is available immediately.
        Convention (same as the backtest): price at time t = close of the candle starting t-60."""
        try:
            end = int(time.time())
            r = self._session.get(self._candles_url, params={"granularity": 60, "start": end - 300 * 60,
                                                       "end": end}, timeout=10)
            r.raise_for_status()
            closes = {c[0] + 60: float(c[4]) for c in r.json()}
            last = None
            for t in range(min(closes), end, 60):          # forward-fill minutes with no trades
                last = closes.get(t, last)
                self._history[t] = last
        except Exception as exc:
            log.warning("candle backfill failed: %s", exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r = self._session.get(self._ticker_url, timeout=3)
                r.raise_for_status()
                now = time.time()
                body = r.json()
                with self._lock:
                    self._ticks.append((now, float(body["price"])))
                    try:
                        self._last_trade_ts = datetime.fromisoformat(body["time"].replace("Z", "+00:00")).timestamp()
                    except (KeyError, ValueError):
                        pass
                    while self._ticks and self._ticks[0][0] < now - self._keep_s:
                        self._ticks.popleft()
            except Exception as exc:  # network blips must never kill the feed
                log.warning("reference feed error: %s", exc)
            self._stop.wait(self._poll_s)

    # ------------------------------------------------------------------ reads
    def latest(self) -> Optional[tuple[float, float]]:
        """(age_seconds, price) of the newest tick, or None if we have nothing yet."""
        with self._lock:
            if not self._ticks:
                return None
            ts, px = self._ticks[-1]
        return time.time() - ts, px

    def trade_age(self) -> float:
        """Seconds since the last ACTUAL trade (``latest()`` only says when we last polled).
        On a thin market like PAXG this can be minutes; on BTC it is about a second."""
        return time.time() - self._last_trade_ts if self._last_trade_ts else float("inf")

    def _snapshot(self):
        with self._lock:
            ticks = list(self._ticks)
        return [t for t, _ in ticks], [p for _, p in ticks]

    def _price_at(self, times, prices, ts: float) -> Optional[float]:
        if times and times[0] <= ts:                       # our own tick history covers it
            return prices[bisect.bisect_right(times, ts) - 1]
        return self._history.get(int(ts // 60) * 60)       # else the candle backfill

    def price_at(self, ts: float) -> Optional[float]:
        times, prices = self._snapshot()
        return self._price_at(times, prices, ts)

    def features(self, now: float) -> Optional[tuple[float, float, float]]:
        """(vol30, m3, m5): std of the last 30 one-minute moves, and 3/5-minute momentum ($).
        None until enough history exists."""
        times, prices = self._snapshot()
        if not prices:
            return None
        boundary = int(now // 60) * 60
        minute = [self._price_at(times, prices, boundary - 60 * k) for k in range(30, -1, -1)]
        cur = prices[-1]
        p3, p5 = self._price_at(times, prices, now - 180), self._price_at(times, prices, now - 300)
        if None in minute or p3 is None or p5 is None:
            return None
        return float(np.std(np.diff(minute), ddof=1)), cur - p3, cur - p5
