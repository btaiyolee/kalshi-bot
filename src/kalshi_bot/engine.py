"""The trading loop. Once a second: read the book, quotes and reference price, ask the
strategy for a decision per execution style, place orders, service resting maker orders,
settle finished windows, and record WHY for everything.

Paper mode runs a taker and a maker strategy side by side on identical signals so their
real-world (simulated) results can be compared. Live mode runs the one configured style."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from kalshi_bot.brokers import Fill, side_ask
from kalshi_bot.config import Config
from kalshi_bot.dashboard import State, TradeRow
from kalshi_bot.fees import FeeSchedule
from kalshi_bot.journal import Journal
from kalshi_bot.kalshi_client import KalshiClient, KalshiError
from kalshi_bot.maker import RestingOrder
from kalshi_bot.model import Model
from kalshi_bot.reference import CoinbaseFeed
from kalshi_bot.risk import RiskManager
from kalshi_bot.strategy import Decision, Snapshot, decide, exit_decision, explain_trade

log = logging.getLogger(__name__)
POLL_S = 1.0
BOOK_DEPTH = 4
BOOK_LOG_EVERY_S = 5.0


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


WINDOW_S = 15 * 60


def pick_active_market(markets: list[dict], now: float) -> Optional[dict]:
    """The window being traded right now: the open market closing soonest, within 15 minutes.

    Do NOT use ``open_time``: the demo server lists future windows a day early, so many
    markets are "open" with an open_time long before their real 15-minute window starts."""
    live = [m for m in markets if now < _ts(m["close_time"]) - 1 <= now + WINDOW_S]
    return min(live, key=lambda m: _ts(m["close_time"])) if live else None


def _num(m: dict, key: str) -> float:
    """Numeric market field; missing or empty counts as 0 (demo markets are sometimes sparse)."""
    try:
        return float(m.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_book(ob: dict, depth: int = BOOK_DEPTH) -> tuple[list, list]:
    """(asks, bids) as [(YES price, size)], nearest the spread first.
    Kalshi lists YES bids and NO bids; a NO bid at p is a YES ask at 1-p."""
    raw = ob.get("orderbook_fp", ob)
    bids = sorted(((float(p), float(s)) for p, s in raw.get("yes_dollars", [])), reverse=True)[:depth]
    asks = sorted((round(1 - float(p), 4), float(s)) for p, s in raw.get("no_dollars", []))[:depth]
    return asks, bids


@dataclass
class Pending:
    fill: Fill
    close_ts: float
    row: TradeRow
    context: dict
    next_check: float = 0.0


@dataclass
class Inflight:
    """A paper taker order waiting out its simulated network delay (never blocks the loop)."""
    due: float
    d: Decision
    count: int
    limit: float
    row: TradeRow
    close_ts: float


@dataclass
class ExitInflight:
    """A paper early-exit order waiting out its simulated delay (never blocks the loop)."""
    due: float
    pend: Pending
    quote: float          # the sell price that triggered the decision
    gain: float           # expected gain from selling instead of holding, per contract


class Engine:
    def __init__(self, cfg: Config, mode: str, client: KalshiClient, model: Model,
                 feed: CoinbaseFeed, taker, maker, fees: FeeSchedule, journal: Journal,
                 state: State, styles: list[str], bankroll: float):
        self.cfg, self.mode, self.client, self.model = cfg, mode, client, model
        self.feed, self.taker, self.maker, self.fees = feed, taker, maker, fees
        self.journal, self.state, self.styles = journal, state, styles
        self.risk = {s: RiskManager(cfg, bankroll) for s in styles}
        self.pending: list[Pending] = []
        self.resting: dict[str, tuple[RestingOrder, TradeRow, dict]] = {}
        self.inflight: dict[str, Inflight] = {}
        self.exit_inflight: dict[str, ExitInflight] = {}    # order id -> pending early exit
        self._book_ticker: Optional[str] = None      # which market the book thread should watch
        self._stopping = threading.Event()
        self.window_fills: dict[str, int] = {}      # fills in the current window, by style
        for s in styles:
            state.stat(s)

    # ------------------------------------------------------------- helpers
    def _active_market(self) -> Optional[dict]:
        markets = self.client.get_markets(series_ticker=self.cfg.series, status="open",
                                          limit=200)["markets"]
        return pick_active_market(markets, time.time())

    def _refresh_balance(self) -> None:
        if self.mode != "live":
            return
        try:
            bal = self.client.get_balance()["balance"] / 100
        except KalshiError as exc:
            log.warning("balance check failed: %s", exc)
            return
        self.state.balance = bal
        for r in self.risk.values():
            r.bankroll = bal

    def _book_loop(self) -> None:
        """Display-only order book, fetched on its own thread with its own HTTP session so it
        can never delay a trading decision."""
        client = KalshiClient(self.client.base_url, timeout=5)
        last_log = 0.0
        while not self._stopping.is_set():
            ticker = self._book_ticker
            if ticker:
                try:
                    asks, bids = parse_book(client.get_orderbook(ticker, depth=BOOK_DEPTH))
                    self.state.asks, self.state.bids = asks, bids
                    if time.time() - last_log >= BOOK_LOG_EVERY_S:      # full ladder, sampled
                        last_log = time.time()
                        self.journal.write("book", ticker=ticker, asks=asks, bids=bids)
                except KalshiError as exc:
                    log.warning("orderbook fetch failed: %s", exc)
            self._stopping.wait(1.0)

    # ---------------------------------------------------------- settlement
    def _settle_pending(self) -> None:
        now = time.time()
        for p in list(self.pending):
            if now < max(p.close_ts + 3, p.next_check):
                continue
            p.next_check = now + 3
            try:
                result = self.client.get_market(p.fill.ticker).get("result", "")
            except KalshiError:
                continue
            if result not in ("yes", "no"):
                continue
            f, style = p.fill, p.fill.style
            pnl = f.settle_pnl(result)
            st = self.state.stat(style)
            st.settled += 1
            st.wins += pnl > 0
            st.pnl += pnl
            st.open -= 1
            st.at_risk -= f.cost
            self.risk[style].record(pnl)
            p.row.status, p.row.pnl, p.row.result = "settled", pnl, result
            self.state.log(f"settled {f.ticker[-9:]} {result.upper()}: {style} {f.side.upper()} "
                           f"x{f.count} @ {f.price:.3f} -> ${pnl:+.2f}")
            self.journal.write("settled", ticker=f.ticker, style=style, side=f.side, count=f.count,
                               order_id=f.order_id,
                               price=f.price, fee=f.fee, result=result, pnl=pnl,
                               quote=p.row.quote, edge=p.row.edge, why=p.row.why,
                               secs_left=p.row.secs_left, entry_ts=p.row.ts)
            self.pending.remove(p)
            self._sync_open_view()
            self._refresh_balance()

    # ------------------------------------------------------------- trading
    def _on_fill(self, fill: Fill, row: TradeRow, close_ts: float) -> None:
        st = self.state.stat(fill.style)
        st.trades += 1
        st.open += 1
        st.at_risk += fill.cost
        st.fees += fill.fee                       # fees are paid when the order fills
        self.window_fills[fill.style] = self.window_fills.get(fill.style, 0) + 1
        row.status, row.price, row.fee, row.count = "filled", fill.price, fill.fee, fill.count
        self.state.log(f"FILLED {fill.style} {fill.side.upper()} x{fill.count} @ {fill.price:.3f} "
                       f"(quote {row.quote:.3f}, fee ${fill.fee:.2f}) | {row.why}")
        self.journal.write("fill", ticker=fill.ticker, style=fill.style, side=fill.side,
                           count=fill.count, price=fill.price, fee=fill.fee, quote=row.quote,
                           order_id=fill.order_id, why=row.why, edge=row.edge,
                           secs_left=row.secs_left,
                           nth_in_window=self.window_fills[fill.style])
        self.pending.append(Pending(fill, close_ts, row, {}))
        self._sync_open_view()

    def _why_not(self, style: str, d: Decision, block: str, attempts: int) -> str:
        """Why a signal is NOT being acted on right now ('' means: go ahead and place the order)."""
        cfg = self.cfg
        if not d.side:
            return ""
        if block:
            return f"paused: {block}"
        done = self.window_fills.get(style, 0)
        if done >= cfg.max_trades_per_window:
            return f"already traded this window ({done}/{cfg.max_trades_per_window}), watching only"
        if style in self.resting:
            o = self.resting[style][0]
            return f"order resting at {o.price:.3f}, cancels {time.strftime('%H:%M:%S', time.localtime(o.deadline_ts))}"
        if style in self.inflight:
            return "order sent, waiting for the (simulated) fill"
        if attempts >= cfg.max_attempts_per_window:
            return f"attempt limit ({cfg.max_attempts_per_window}) reached this window"
        return ""

    def _execute(self, style: str, market: dict, d: Decision, snap: Snapshot, close_ts: float) -> str:
        """Place the order; returns a one-line status for the dashboard."""
        cfg, s = self.cfg, self.state
        count = self.risk[style].size(d.price)
        if count < 1:
            msg = (f"skipped: bankroll ${self.risk[style].bankroll:,.0f} x {cfg.risk_per_trade_pct:.1%} "
                   f"buys <1 contract at {d.price:.3f}")
            s.log(f"{style}: signal {msg}")
            return msg
        why = explain_trade(d)
        row = TradeRow(time.time(), market["ticker"], style, d.side, count, d.price, d.price,
                       0.0, d.edge, snap.tau_min * 60, why, "sent")
        s.trades.appendleft(row)
        self.journal.write("signal", ticker=market["ticker"], style=style, side=d.side,
                           count=count, price=d.price, edge=d.edge, why=why, detail=d.detail,
                           secs_left=row.secs_left, mode=self.mode)
        try:
            if style == "maker":
                deadline = min(time.time() + cfg.maker_rest_s, close_ts - 15)
                order = self.maker.place(market, d.side, count, d.price, deadline, d.detail)
                row.status = "resting"
                self.resting[style] = (order, row, d.detail)
                s.log(f"maker: resting {d.side.upper()} x{count} @ {d.price:.3f} until "
                      f"{time.strftime('%H:%M:%S', time.localtime(deadline))} | {why}")
                self.journal.write("maker_placed", ticker=market["ticker"], side=d.side, count=count,
                                   price=d.price, deadline=deadline, order_id=order.order_id)
                return f"maker order resting {d.side.upper()} x{count} @ {d.price:.3f}"
            limit = d.price + cfg.slippage_cap
            if self.mode == "paper":                  # fill later, against a fresh book; don't block
                self.inflight[style] = Inflight(time.time() + cfg.paper_latency_s, d, count, limit,
                                                row, close_ts)
                return f"taker order sent: {d.side.upper()} x{count}, waiting for simulated fill"
            fill = self.taker.buy(market, d.side, count, limit)
        except KalshiError as exc:
            row.status = "error"
            s.log(f"{style} order error: {exc}")
            self.journal.write("order_error", style=style, error=str(exc))
            return "order error (see events)"
        self._taker_result(market, d, limit, fill, row, close_ts)
        return f"taker order {row.status}: {d.side.upper()} x{row.count}"

    def _taker_result(self, market: dict, d: Decision, limit: float, fill: Optional[Fill],
                      row: TradeRow, close_ts: float) -> None:
        if fill is None:
            fresh, _ = side_ask(market, d.side)
            row.status = "missed"
            self.state.log(f"taker missed {d.side.upper()}: quote {d.price:.3f} -> now {fresh:.3f} "
                           f"(limit {limit:.3f})")
            self.journal.write("miss", ticker=market["ticker"], side=d.side, quote=d.price, now=fresh)
            return
        self._on_fill(fill, row, close_ts)

    def _service_inflight(self) -> None:
        """Complete paper taker orders whose simulated delay has elapsed."""
        now = time.time()
        for style, f in list(self.inflight.items()):
            if now < f.due:
                continue
            del self.inflight[style]
            try:
                market = self.client.get_market(f.row.ticker)
                fill = self.taker.buy(market, f.d.side, f.count, f.limit)
            except KalshiError as exc:
                f.row.status = "error"
                self.state.log(f"{style} order error: {exc}")
                self.journal.write("order_error", style=style, error=str(exc))
                continue
            self._taker_result(market, f.d, f.limit, fill, f.row, f.close_ts)

    def _service_resting(self, close_ts: float) -> None:
        now = time.time()
        for style, (order, row, _) in list(self.resting.items()):
            fill = self.maker.poll(order, now)
            if fill is not None:
                self._on_fill(fill, row, close_ts)
            elif order.status == "expired":
                row.status = "expired"
                self.state.log(f"maker: {order.side.upper()} @ {order.price:.3f} expired unfilled")
                self.journal.write("maker_expired", ticker=order.ticker, side=order.side,
                                   price=order.price)
            else:
                continue
            del self.resting[style]
        self._sync_resting_view()

    def _sync_open_view(self) -> None:
        self.state.open_positions = [(p.fill.style, p.close_ts, p.fill.cost) for p in self.pending]

    def _sync_resting_view(self) -> None:
        self.state.resting = [(st, o.side, o.price, o.count) for st, (o, _, _) in self.resting.items()]

    def shutdown(self) -> None:
        """Stop background work and cancel resting live orders so none are left open."""
        self._stopping.set()
        for style, (order, _row, _) in list(self.resting.items()):
            if self.mode == "live":
                try:
                    self.client.cancel_order(order.order_id)
                    self.journal.write("maker_cancelled_on_shutdown", order_id=order.order_id)
                except KalshiError as exc:
                    log.warning("could not cancel %s on shutdown: %s", order.order_id, exc)
        self.journal.write("shutdown", open_positions=len(self.pending),
                           resting=len(self.resting))

    # ------------------------------------------------------- early exits (paper)
    def _consider_exits(self, ticker: str, snap: Snapshot, p_yes: float) -> None:
        """For each open position in this window: is selling now worth more than holding?"""
        cfg = self.cfg
        self.state.exit_line = ""
        if cfg.exit_rule != "model" or self.mode != "paper":
            return
        mine = [p for p in self.pending if p.fill.ticker == ticker]
        lines = []
        for p in mine:
            f = p.fill
            ed = exit_decision(f.side, f.count, snap.yes_bid, snap.yes_ask, p_yes, self.fees, cfg.exit_margin)
            lines.append(f"{f.style} {f.side.upper()}: sell {ed.net_sale:.2f} vs hold {ed.hold_value:.2f}"
                         f" -> {'SELL' if ed.exit else 'hold'}")
            if ed.exit and snap.tau_min >= cfg.min_tau_min and f.order_id not in self.exit_inflight:
                self.exit_inflight[f.order_id] = ExitInflight(time.time() + cfg.paper_latency_s, p,
                                                              ed.sell_price, ed.gain)
                p.row.status = "exiting"
                self.state.log(f"exit signal: {f.style} {f.side.upper()} x{f.count} - selling nets "
                               f"{ed.net_sale:.3f} vs {ed.hold_value:.3f} for holding (+{ed.gain:.3f})")
                self.journal.write("exit_signal", ticker=ticker, style=f.style, side=f.side,
                                   order_id=f.order_id, sell_price=ed.sell_price, net_sale=ed.net_sale,
                                   hold_value=ed.hold_value, gain=ed.gain, secs_left=snap.tau_min * 60)
        extra = f" (+{len(lines) - 2} more)" if len(lines) > 2 else ""
        self.state.exit_line = " | ".join(lines[:2]) + extra

    def _service_exits(self) -> None:
        """Complete exits whose simulated delay has elapsed, against a fresh book."""
        now = time.time()
        for oid, x in list(self.exit_inflight.items()):
            if now < x.due:
                continue
            del self.exit_inflight[oid]
            p, f = x.pend, x.pend.fill
            if p not in self.pending:                     # settled while we were waiting
                continue
            try:
                m = self.client.get_market(f.ticker)
            except KalshiError:
                p.row.status = "filled"
                continue
            bid, ask = _num(m, "yes_bid_dollars"), _num(m, "yes_ask_dollars")
            sell = bid if f.side == "yes" else 1 - ask
            size = _num(m, "yes_bid_size_fp") if f.side == "yes" else _num(m, "yes_ask_size_fp")
            if m.get("status") != "active" or sell < x.quote - self.cfg.slippage_cap or size < f.count:
                p.row.status = "filled"                   # missed: still holding, may try again
                self.state.log(f"exit missed: {f.style} {f.side.upper()} quote {x.quote:.3f} -> now {sell:.3f}")
                self.journal.write("exit_missed", ticker=f.ticker, style=f.style, order_id=oid,
                                   quote=x.quote, now=sell, size=size)
                continue
            self._close_early(p, sell)

    def _close_early(self, p: Pending, sell: float) -> None:
        """Book a position sold before settlement: proceeds minus both fees minus what we paid."""
        f, style = p.fill, p.fill.style
        sale_fee = self.fees.taker_fee(sell, f.count)
        pnl = sell * f.count - sale_fee - f.cost
        st = self.state.stat(style)
        st.settled += 1
        st.wins += pnl > 0
        st.pnl += pnl
        st.fees += sale_fee
        st.open -= 1
        st.at_risk -= f.cost
        self.risk[style].record(pnl)
        p.row.status, p.row.pnl, p.row.result, p.row.fee = "settled", pnl, "exit", f.fee + sale_fee
        self.state.log(f"EXITED {style} {f.side.upper()} x{f.count}: bought {f.price:.3f}, sold {sell:.3f} "
                       f"(fees ${f.fee + sale_fee:.2f}) -> ${pnl:+.2f}")
        self.journal.write("settled", ticker=f.ticker, style=style, side=f.side, count=f.count,
                           order_id=f.order_id, price=f.price, fee=f.fee + sale_fee, result="exit",
                           pnl=pnl, quote=p.row.quote, edge=p.row.edge, why=p.row.why,
                           secs_left=p.row.secs_left, entry_ts=p.row.ts, exited=True,
                           exit_price=sell, exit_fee=sale_fee)
        self.pending.remove(p)
        self._sync_open_view()
        self._refresh_balance()

    # ----------------------------------------------------------- main loop
    def run(self) -> None:
        self._refresh_balance()
        threading.Thread(target=self._book_loop, daemon=True, name="orderbook").start()
        while True:
            self._service_inflight()
            self._service_exits()
            self._settle_pending()
            try:
                market = self._active_market()
            except KalshiError as exc:
                self.state.status = f"api error: {exc}"
                time.sleep(2)
                continue
            if market is None:
                self.state.status = "waiting for next window"
                time.sleep(2)
                continue
            self._run_window(market)

    def _run_window(self, market: dict) -> None:
        cfg, s = self.cfg, self.state
        ticker = market["ticker"]
        close_ts = _ts(market["close_time"])
        open_ts = close_ts - WINDOW_S                 # the real start of this 15-minute window
        ref_open = None
        self.window_fills = {st: 0 for st in self.styles}
        self._book_ticker = ticker
        self.state.history.clear()                    # sparklines show THIS window only
        attempts = {st: 0 for st in self.styles}
        s.strike = _num(market, "floor_strike")      # 0 until Kalshi publishes it for the window
        s.log(f"window {ticker[-9:]} opened (strike {'$%.2f' % s.strike if s.strike else 'TBD'})")

        while time.time() < close_ts:
            loop_start = time.time()
            self._service_inflight()
            self._service_exits()
            self._settle_pending()
            try:
                m = self.client.get_market(ticker)
            except KalshiError as exc:
                s.status = f"api error: {exc}"
                time.sleep(self.cfg.poll_s)
                continue
            if ref_open is None:
                ref_open = self.feed.price_at(open_ts)
            latest, feats = self.feed.latest(), self.feed.features(time.time())
            if ref_open is None or latest is None or feats is None:
                s.status = "warming up reference price / volatility"
                time.sleep(self.cfg.poll_s)
                continue

            age, ref_now = latest
            vol30, m3, m5 = feats
            if not s.strike:
                s.strike = _num(m, "floor_strike")
            snap = Snapshot(_num(m, "yes_bid_dollars"), _num(m, "yes_ask_dollars"),
                            _num(m, "yes_bid_size_fp"), _num(m, "yes_ask_size_fp"),
                            (close_ts - time.time()) / 60, ref_now - ref_open, age, vol30, m3, m5)
            s.ticker, s.tau_min, s.yes_bid, s.yes_ask = ticker, snap.tau_min, snap.yes_bid, snap.yes_ask
            s.ref_price, s.ref_open, s.ref_move, s.ref_age_s, s.vol30 = ref_now, ref_open, snap.ref_move, age, vol30
            s.history.append(((snap.yes_bid + snap.yes_ask) / 2, snap.ref_move))
            s.status = "watching"

            for style in sorted(self.styles):            # "maker" before "taker": no sleeping first
                d = decide(snap, self.model, self.fees, cfg, style)
                s.decisions[style] = d
                s.blocked[style] = block = self.risk[style].block_reason()
                action = self._why_not(style, d, block, attempts[style])
                if d.side and not action:                # nothing stops us: place the order
                    attempts[style] += 1
                    action = self._execute(style, m, d, snap, close_ts)
                s.actions[style] = action
                self.journal.write("snapshot", ticker=ticker, style=style, bid=snap.yes_bid,
                                   ask=snap.yes_ask, bid_size=snap.yes_bid_size,
                                   ask_size=snap.yes_ask_size, tau=round(snap.tau_min, 4),
                                   ref=ref_now, ref_open=ref_open, ref_age=round(age, 2),
                                   vol30=round(vol30, 4), m3=round(m3, 3), m5=round(m5, 3),
                                   trade_age=round(min(self.feed.trade_age(), 9999), 1),
                                   p_yes=round(d.p_yes, 4), edge=round(d.edge, 4),
                                   decision=d.side or "", reason=d.reason, blocked=block,
                                   action=action)
            self._consider_exits(ticker, snap, next(iter(s.decisions.values())).p_yes)
            self._service_resting(close_ts)          # after deciding: polling must not delay decisions
            wait = cfg.poll_s - (time.time() - loop_start)
            dues = [f.due for f in self.inflight.values()] + [x.due for x in self.exit_inflight.values()]
            if dues:                                 # wake in time for the earliest simulated fill/exit
                wait = min(wait, min(dues) - time.time())
            time.sleep(max(0.0, wait))

        self._book_ticker = None
        # window over: let any resting order finish (expire / cancel) before moving on
        for _ in range(10):
            if not self.resting:
                break
            self._service_resting(close_ts)
            time.sleep(1)
