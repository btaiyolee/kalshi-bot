"""Order execution. ``PaperBroker`` simulates fills against the real live book;
``LiveBroker`` sends real immediate-or-cancel orders. Same interface, so the engine
does not care which one it has."""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol

from kalshi_bot.fees import FeeSchedule
from kalshi_bot.kalshi_client import KalshiClient

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: str            # "yes" or "no"
    count: int
    price: float         # average price paid per contract ($)
    fee: float           # total fee ($)
    order_id: str = ""
    style: str = "taker"  # "taker" or "maker"

    @property
    def cost(self) -> float:
        return self.price * self.count + self.fee

    def settle_pnl(self, result: str) -> float:
        """Profit once the market resolves ("yes" or "no")."""
        return (self.count if result == self.side else 0.0) - self.cost


class Broker(Protocol):
    def buy(self, market: dict, side: str, count: int, limit: float) -> Optional[Fill]:
        """Try to buy ``count`` contracts of ``side`` at no more than ``limit``."""


def tick_ceil(price: float) -> float:
    """Round up to a valid Kalshi price tick (0.001 below 10c / above 90c, else 0.01)."""
    step = 0.01 if 0.10 <= price < 0.90 else 0.001
    return round(math.ceil(round(price / step, 6)) * step, 3)


def tick_floor(price: float) -> float:
    """Round down to a valid Kalshi price tick."""
    step = 0.01 if 0.10 <= price < 0.90 else 0.001
    return round(math.floor(round(price / step, 6)) * step, 3)


def side_ask(market: dict, side: str) -> tuple[float, float]:
    """(ask price, size) for buying ``side``. Buying NO lifts the YES bid."""
    if side == "yes":
        return float(market["yes_ask_dollars"]), float(market["yes_ask_size_fp"])
    return 1 - float(market["yes_bid_dollars"]), float(market["yes_bid_size_fp"])


class PaperBroker:
    """Fills at the fresh ask if it is still within ``limit`` and has enough size,
    exactly like an IOC limit order would. The caller supplies a market snapshot taken
    AFTER the simulated latency, so stale-quote 'wins' are not credited."""

    def __init__(self, fees: FeeSchedule):
        self.fees = fees

    def buy(self, market: dict, side: str, count: int, limit: float) -> Optional[Fill]:
        price, size = side_ask(market, side)
        if price > limit + 1e-9 or size < count:
            return None
        return Fill(market["ticker"], side, count, price, self.fees.taker_fee(price, count),
                    order_id=f"paper-{uuid.uuid4().hex[:8]}")


class LiveBroker:
    def __init__(self, client: KalshiClient):
        self.client = client

    def buy(self, market: dict, side: str, count: int, limit: float) -> Optional[Fill]:
        price = f"{tick_ceil(limit):.4f}"
        order = self.client.create_order(
            ticker=market["ticker"], side=side, action="buy", count=count,
            time_in_force="immediate_or_cancel",
            client_order_id=uuid.uuid4().hex,
            exchange_index=market.get("exchange_index", -1),
            **{f"{side}_price_dollars": price})
        filled = int(float(order.get("fill_count_fp", "0")))
        if filled == 0:
            return None
        cost = float(order.get("taker_fill_cost_dollars", 0)) + float(order.get("maker_fill_cost_dollars", 0))
        fee = float(order.get("taker_fees_dollars", 0)) + float(order.get("maker_fees_dollars", 0))
        return Fill(market["ticker"], side, filled, cost / filled, fee, order_id=order["order_id"])
