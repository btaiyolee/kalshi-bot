"""Decision logic. A pure function: same inputs always give the same decision, so it is
easy to unit-test. Every decision (trade or skip) carries the numbers behind it so the
dashboard and the journal can show exactly WHY."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kalshi_bot.brokers import tick_floor
from kalshi_bot.config import Config
from kalshi_bot.fees import FeeSchedule
from kalshi_bot.model import Model


@dataclass(frozen=True)
class Snapshot:
    """Everything known at one instant about the market and the reference price."""
    yes_bid: float
    yes_ask: float
    yes_bid_size: float
    yes_ask_size: float
    tau_min: float                 # minutes until the window closes
    ref_move: float                # reference price now minus at window open ($)
    ref_age_s: float               # how stale the reference price is
    vol30: float                   # trailing 1-minute volatility of the reference ($)
    m3: float                      # 3-minute reference momentum ($)
    m5: float                      # 5-minute reference momentum ($)


@dataclass(frozen=True)
class Decision:
    style: str                     # "taker" or "maker"
    side: Optional[str]            # "yes", "no", or None (no trade)
    price: float = 0.0             # price per contract we would pay / rest at
    p_yes: float = 0.0             # model probability of YES
    edge: float = 0.0              # expected profit per contract after fees
    reason: str = ""               # one line: why we traded / why we skipped
    detail: dict = field(default_factory=dict)   # numbers behind the decision


def decide(snap: Snapshot, model: Model, fees: FeeSchedule, cfg: Config,
           style: str = "taker", contracts: Optional[int] = None) -> Decision:
    n = contracts or cfg.contracts
    mid = (snap.yes_bid + snap.yes_ask) / 2
    args = (mid, snap.ref_move, snap.tau_min, snap.vol30, snap.m3, snap.m5)
    p = model.predict(*args)
    detail = {"mid": mid, "p_yes": p, "features": model.explain(*args), "move": snap.ref_move,
              "vol30": snap.vol30, "m3": snap.m3, "m5": snap.m5, "tau_min": snap.tau_min,
              "bid": snap.yes_bid, "ask": snap.yes_ask}

    def skip(why: str, edge: float = 0.0) -> Decision:
        return Decision(style, None, p_yes=p, edge=edge, reason=why, detail=detail)

    if snap.yes_bid <= 0 or snap.yes_ask >= 1:
        return skip("no two-sided book")
    if snap.ref_age_s > cfg.max_ref_age_s:
        return skip(f"reference stale ({snap.ref_age_s:.0f}s)")
    if not cfg.min_tau_min <= snap.tau_min <= cfg.max_tau_min:
        return skip("outside trading window")
    if snap.yes_ask - snap.yes_bid > cfg.max_spread:
        return skip("spread too wide")

    if style == "taker":                             # lift the ask; pay the taker fee
        yes_price, no_price = snap.yes_ask, 1 - snap.yes_bid
        fee = fees.taker_fee
    else:                                            # rest one step inside our side of the book
        yes_price = tick_floor(min(snap.yes_bid + cfg.maker_improve, snap.yes_ask - 0.001))
        no_price = tick_floor(min((1 - snap.yes_ask) + cfg.maker_improve, (1 - snap.yes_bid) - 0.001))
        fee = fees.maker_fee
    edge_yes = p - yes_price - fee(yes_price, n) / n
    edge_no = (1 - p) - no_price - fee(no_price, n) / n

    side, edge, price = (("yes", edge_yes, yes_price) if edge_yes >= edge_no
                         else ("no", edge_no, no_price))
    detail.update(side=side, price=price, fee_per_contract=fee(price, n) / n,
                  p_side=p if side == "yes" else 1 - p, edge_yes=edge_yes, edge_no=edge_no,
                  yes_price=yes_price, no_price=no_price,
                  fee_yes=fee(yes_price, n) / n, fee_no=fee(no_price, n) / n,
                  min_edge=cfg.min_net_edge)
    if edge < cfg.min_net_edge:
        return skip(f"best edge {edge:+.3f} < {cfg.min_net_edge}", edge)
    if style == "taker":
        size = snap.yes_ask_size if side == "yes" else snap.yes_bid_size
        if size < n:
            return skip("not enough size at the touch", edge)
    return Decision(style, side, price, p, edge,
                    f"{side.upper()} edge {edge:+.3f}/contract after fees", detail)


def explain_trade(d: Decision) -> str:
    """One-line human reason for a trade, e.g. 'ref +4.2 z=+1.8 -> p 0.71 vs mid 0.55'."""
    x = d.detail
    z = x["features"]["z_vol"]["value"]
    return (f"ref {x['move']:+.1f}$ (z {z:+.1f}, mom3 {x['m3']:+.1f}) -> model {x['p_yes']:.2f} "
            f"vs market {x['mid']:.2f}; edge {d.edge:+.3f} after fees")


@dataclass(frozen=True)
class ExitDecision:
    exit: bool
    sell_price: float        # what one contract sells for right now, before the sale fee
    net_sale: float          # sell_price minus the sale fee per contract
    hold_value: float        # the model's probability that our side wins = what holding is worth
    gain: float              # net_sale - hold_value: expected gain from selling instead of holding


def exit_decision(side: str, count: int, yes_bid: float, yes_ask: float, p_yes: float,
                  fees: FeeSchedule, margin: float) -> ExitDecision:
    """Sell now if the cash we'd net beats what the model thinks holding is worth, by ``margin``.

    A contract held to settlement is worth P(our side wins). Selling it now at the bid pays the
    bid minus a second taker fee. So sell exactly when that net cash exceeds the model
    probability (plus a safety margin for model noise). Buying YES is exited by hitting the YES
    bid; buying NO by hitting the NO bid, which is 1 - the YES ask."""
    sell = yes_bid if side == "yes" else 1 - yes_ask
    hold = p_yes if side == "yes" else 1 - p_yes
    if sell <= 0.01 or count < 1:                       # no real bid to sell into
        return ExitDecision(False, max(sell, 0.0), 0.0, hold, 0.0)
    net = sell - fees.taker_fee(sell, count) / count
    return ExitDecision(net > hold + margin, sell, net, hold, net - hold)
