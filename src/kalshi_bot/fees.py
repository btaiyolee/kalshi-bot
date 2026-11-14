"""Kalshi fee model.

Kalshi charges a *quadratic* fee that peaks at 50c and shrinks toward 0c / 100c:

    taker_fee = ceil_to_cent( 0.07   * contracts * P * (1 - P) )
    maker_fee = ceil_to_cent( 0.0175 * contracts * P * (1 - P) )   # only if the series charges makers

where P is the contract price in dollars. Whether makers pay is series-specific:
``GET /series/{ticker}`` returns ``fee_type`` ("quadratic" = takers only,
"quadratic_with_maker_fees" = both) and a ``fee_multiplier`` that scales both rates.

Fees are rounded UP to the next cent per order, so many tiny orders cost more
than one larger order. Kalshi's official fee PDF could not be fetched while
building this (bot protection); the formula above matches several independent
secondary sources. The live bot also records the fees Kalshi actually charges
(``taker_fees_dollars`` on each order) so any drift is visible in the logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TAKER_RATE = 0.07
MAKER_RATE = 0.0175


@dataclass(frozen=True)
class FeeSchedule:
    """Fee parameters for one series."""

    multiplier: float = 1.0
    makers_pay: bool = False

    @classmethod
    def from_series(cls, series: dict) -> "FeeSchedule":
        """Build from the ``series`` object returned by ``GET /series/{ticker}``."""
        return cls(
            multiplier=float(series.get("fee_multiplier", 1.0)),
            makers_pay="maker" in str(series.get("fee_type", "")),
        )

    def taker_fee(self, price: float, contracts: int = 1) -> float:
        """Total taker fee in dollars for an order of ``contracts`` at ``price``."""
        return _quadratic_fee(TAKER_RATE * self.multiplier, price, contracts)

    def maker_fee(self, price: float, contracts: int = 1) -> float:
        """Total maker fee in dollars (zero when this series does not charge makers)."""
        if not self.makers_pay:
            return 0.0
        return _quadratic_fee(MAKER_RATE * self.multiplier, price, contracts)


def _quadratic_fee(rate: float, price: float, contracts: int) -> float:
    raw = rate * contracts * price * (1.0 - price)
    # Round to 1e-9 first so float noise (e.g. 0.0175000000001) cannot bump a cent.
    return math.ceil(round(raw * 100, 9)) / 100
