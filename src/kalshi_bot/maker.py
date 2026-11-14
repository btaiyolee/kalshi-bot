"""Maker (resting-order) execution.

A maker order sits in the book until someone trades into it, so it pays no fee on this
series and earns a better price, but it may not fill, and it tends to fill exactly when
the price is about to move against us ("adverse selection"). Both brokers expose the
same two calls: ``place`` and ``poll``.

PaperMaker  decides fills from the REAL public trade tape. A trade must go strictly
            THROUGH our price after our (delayed) order would have reached the book,
            which is conservative: it ignores partial queue priority at our own price.
LiveMaker   sends a post-only GTC order that expires on its own, polls its status, and
            cancels at the deadline.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from kalshi_bot.brokers import Fill
from kalshi_bot.fees import FeeSchedule
from kalshi_bot.kalshi_client import KalshiClient, KalshiError

log = logging.getLogger(__name__)


@dataclass
class RestingOrder:
    ticker: str
    side: str                 # "yes" or "no"
    count: int
    price: float              # our limit price, in our own side's terms
    placed_ts: float
    deadline_ts: float
    order_id: str = ""
    active_from: float = 0.0  # when the order is assumed to reach the book (paper)
    context: dict = field(default_factory=dict)   # the decision that created it
    filled: int = 0
    status: str = "resting"   # resting | filled | expired


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def crossing_volume(trades: list[dict], side: str, price: float, since: float) -> float:
    """Contracts traded strictly through our price after ``since``.

    A YES bid at b is hit by YES trades below b. A NO bid at n is a YES offer at 1-n,
    hit by YES trades above 1-n."""
    total = 0.0
    for t in trades:
        if _ts(t["created_time"]) <= since:
            continue
        yes_px = float(t["yes_price_dollars"])
        if (side == "yes" and yes_px < price - 1e-9) or (side == "no" and yes_px > 1 - price + 1e-9):
            total += float(t["count_fp"])
    return total


class PaperMaker:
    def __init__(self, client: KalshiClient, fees: FeeSchedule, latency_s: float):
        self.client, self.fees, self.latency_s = client, fees, latency_s

    def place(self, market: dict, side: str, count: int, price: float, deadline_ts: float,
              context: dict) -> RestingOrder:
        now = time.time()
        return RestingOrder(market["ticker"], side, count, price, now, deadline_ts,
                            order_id=f"paper-maker-{uuid.uuid4().hex[:8]}",
                            active_from=now + self.latency_s, context=context)

    def poll(self, o: RestingOrder, now: float) -> Fill | None:
        """Returns a Fill when the order completes with at least one contract, updates
        ``o.status`` when it expires empty, and returns None while still resting."""
        if now < o.active_from:
            return None
        try:
            trades = self.client.get_trades(o.ticker, int(o.active_from))
        except KalshiError as exc:
            log.warning("trade tape fetch failed: %s", exc)
            return None
        filled = min(o.count, math.floor(crossing_volume(trades, o.side, o.price, o.active_from)))
        if filled >= o.count or (now >= o.deadline_ts and filled > 0):
            o.filled, o.status = filled, "filled"
            return Fill(o.ticker, o.side, filled, o.price, self.fees.maker_fee(o.price, filled),
                        o.order_id, "maker")
        if now >= o.deadline_ts:
            o.status = "expired"
        return None


class LiveMaker:
    def __init__(self, client: KalshiClient, fees: FeeSchedule):
        self.client, self.fees = client, fees

    def place(self, market: dict, side: str, count: int, price: float, deadline_ts: float,
              context: dict) -> RestingOrder:
        order = self.client.create_order(
            ticker=market["ticker"], side=side, action="buy", count=count,
            time_in_force="good_till_canceled", expiration_ts=int(deadline_ts), post_only=True,
            client_order_id=uuid.uuid4().hex, exchange_index=market.get("exchange_index", -1),
            **{f"{side}_price_dollars": f"{price:.4f}"})
        now = time.time()
        return RestingOrder(market["ticker"], side, count, price, now, deadline_ts,
                            order_id=order["order_id"], active_from=now, context=context)

    def poll(self, o: RestingOrder, now: float) -> Fill | None:
        try:
            order = self.client.get_order(o.order_id)
            if now >= o.deadline_ts and order.get("status") == "resting":
                self.client.cancel_order(o.order_id)
                order = self.client.get_order(o.order_id)
        except KalshiError as exc:
            log.warning("maker order poll failed: %s", exc)
            return None
        filled = int(float(order.get("fill_count_fp", "0")))
        finished = order.get("status") in ("executed", "canceled") or now >= o.deadline_ts + 5
        if not finished:
            return None
        if filled == 0:
            o.status = "expired"
            return None
        cost = float(order.get("maker_fill_cost_dollars", 0)) + float(order.get("taker_fill_cost_dollars", 0))
        fee = float(order.get("maker_fees_dollars", 0)) + float(order.get("taker_fees_dollars", 0))
        o.filled, o.status = filled, "filled"
        return Fill(o.ticker, o.side, filled, cost / filled, fee, o.order_id, "maker")
