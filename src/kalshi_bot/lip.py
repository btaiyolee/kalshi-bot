"""Liquidity Incentive Program (LIP) quoting: paper tester and recorder.

    kalshi-bot lip-paper            select markets, then simulate two resting bids forever
    kalshi-bot lip-paper --report   print results from the event logs in data/lip/
    kalshi-bot lip-paper --select-only   just show what would be selected

PAPER ONLY. Uses unauthenticated public endpoints (``/incentive_programs``, ``/series``,
``/markets``, ``/markets/{t}/orderbook``, ``/markets/trades``). It never signs a request and
never places an order.

What the program pays (https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program)
------------------------------------------------------------------------------------------
* Snapshots of the book are taken once a second at a random moment. Per snapshot, per side
  (YES bids, NO bids), the REFERENCE PRICE is the first level, walking down from the best bid,
  where cumulative resting size reaches Target Size / 5.
* An order at or better than the reference scores size x 1.0; an order n ticks worse scores
  size x DiscountFactor^n. Your side score = your raw score / total raw score on that side.
* Snapshot score = yes share + no share (max 2 across all participants). A snapshot is excluded
  if either side's resting size is below Target Size.
* Period payout = my snapshot scores / all participants' snapshot scores x Reward x
  (included snapshots / all snapshots). Since each included snapshot distributes exactly 2.0,
  that equals  Reward / (2 x period_seconds) x my_snapshot_score  summed over included snapshots,
  which is what ``interval_reward`` integrates over the time between our book polls.
* Resting orders earn even when never filled. Market makers are excluded (we cannot see who
  is one, so every order in the public book is counted as a competitor).

Units (verified by inference, not by a published statement)
-----------------------------------------------------------
``period_reward`` in ``GET /incentive_programs`` is in 1/10,000 of a dollar (0.01 cent):
across ~4,300 active programs the implied daily pool is $3.2 minimum, $14.3 median, $143 maximum,
inside the documented $1-$1,000/day band; a cents unit would give a $1,428 maximum, outside it.
If the unit were different, only the reward scale changes by a constant factor (``REWARD_UNIT``).

Assumptions that are NOT verified
---------------------------------
* "Resting size reaches Target Size" counts ALL resting size on the side at any price.
* Our order gets queue position behind everyone already at its price (conservative variant).
* Maker fee: 0 unless the series is ``quadratic_with_maker_fees``, then ceil-to-cent
  0.0175 x fee_multiplier x C x P x (1-P) per fill (see ``fees.py``).
* Rewards are computed from the book at each poll and held constant until the next poll
  (30-60 s), not from 1-second snapshots.
* Paying out in whole cents / minimum payout thresholds are ignored.

Fills (two variants, both logged; "cons" is the headline)
---------------------------------------------------------
cons  fills only when a public trade prints STRICTLY THROUGH our price (YES bid at b: a trade
      at YES price < b; NO bid at n: a trade at YES price > 1-n), the convention of
      ``maker.PaperMaker``. Volume at our own price never fills us.
opt   additionally fills on volume printing AT our price (as if we were first in the queue).
Fills happen at our limit price. Real fills are somewhere between cons and opt.

Event log (data/lip/events-<session>.jsonl, one JSON object per line):
``config``, ``select``, ``quote``, ``pull``, ``fill``, ``adv`` (mark-out of a fill 1m/1h/1d later),
``snap`` (periodic per-market state with cumulative reward, inventory, mark-to-market),
``settle``, ``stop``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import heapq
import json
import logging
import math
import random
import re
import signal
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kalshi_bot.fees import FeeSchedule
from kalshi_bot.kalshi_client import PROD_URL, KalshiClient, KalshiError

log = logging.getLogger(__name__)

REWARD_UNIT = 1e-4          # dollars per unit of ``period_reward`` (see module docstring)
TICK = 0.01
VARIANTS = ("cons", "opt")
HORIZONS = (("1m", 60), ("1h", 3600), ("1d", 86400))


# ============================================================================ config
@dataclass
class LipConfig:
    n_markets: int = 30
    size: int = 100                 # contracts per resting order
    offset_ticks: int = 0           # 0 = at the reference price, +1 = one tick inside (better)
    cap: int = 200                  # max net |YES - NO| contracts per market
    total_cap: int = 1000           # max sum of per-market |net| contracts
    min_pool: float = 10.0          # $/day, selection
    min_days: float = 7.0           # days to close, selection
    pull_hours: float = 24.0        # stop quoting this close to close_time
    max_spread: float = 0.10        # selection: yes ask - yes bid
    min_mid: float = 0.05
    max_mid: float = 0.95
    max_vol24: float = 5000.0       # selection: contracts traded in 24h ("slow")
    jump: float = 0.05              # pull if mid ranges >= this within jump_window_s
    jump_window_s: float = 900.0
    cooldown_s: float = 3600.0
    requote_ticks: int = 2          # re-place an order only if desired price moved this much
    latency_s: float = 2.0          # order assumed to reach the book this long after placing
    book_poll_s: float = 45.0
    trades_poll_s: float = 90.0
    status_poll_s: float = 600.0
    programs_poll_s: float = 3600.0
    snap_log_s: float = 300.0
    max_rps: float = 1.4
    max_per_series: int = 2
    max_probe: int = 150            # candidate markets examined during selection
    exclude_categories: str = "Economics,Financials,Crypto,Climate and Weather"
    exclude_frequencies: str = "fifteen_min,hourly,daily"
    exclude_early_close: bool = False
    seed: int = 7
    dir: str = "data/lip"


# ============================================================================ pure core
def parse_book(raw: dict) -> tuple[dict[float, float], dict[float, float]]:
    """``/orderbook`` response -> (yes_bids, no_bids), each {price in that side's dollars: size}."""
    ob = raw.get("orderbook_fp") or raw.get("orderbook") or {}

    def side(rows):
        out: dict[float, float] = {}
        for px, sz in rows or []:
            p, s = round(float(px), 4), float(sz)
            if s > 0:
                out[p] = out.get(p, 0.0) + s
        return out
    return side(ob.get("yes_dollars")), side(ob.get("no_dollars"))


def side_total(levels: dict[float, float]) -> float:
    return sum(levels.values())


def reference_price(levels: dict[float, float], target: float) -> float | None:
    """First level, walking down from the best bid, where cumulative size >= target/5."""
    cum = 0.0
    for p in sorted(levels, reverse=True):
        cum += levels[p]
        if cum >= target / 5.0 - 1e-9:
            return p
    return None


def distance_mult(price: float, ref: float, df: float, tick: float = TICK) -> float:
    """1.0 at or better than the reference, df^ticks below it."""
    if price >= ref - 1e-9:
        return 1.0
    return df ** round((ref - price) / tick)


def _with_mine(levels: dict[float, float], mine: tuple[float, float] | None) -> dict[float, float]:
    if not mine or mine[1] <= 0:
        return levels
    out = dict(levels)
    out[round(mine[0], 4)] = out.get(round(mine[0], 4), 0.0) + mine[1]
    return out


def side_share(levels: dict[float, float], mine: tuple[float, float] | None, target: float,
               df: float) -> tuple[float, float | None]:
    """(my share of this side's total discounted size, reference price). Our order is added to the
    book before the reference and the totals are computed."""
    if not mine or mine[1] <= 0:
        return 0.0, reference_price(levels, target)
    book = _with_mine(levels, mine)
    ref = reference_price(book, target)
    if ref is None:
        return 0.0, None
    total = sum(sz * distance_mult(p, ref, df) for p, sz in book.items())
    return (mine[1] * distance_mult(mine[0], ref, df)) / total, ref


@dataclass
class RewardResult:
    score_yes: float = 0.0
    score_no: float = 0.0
    included: bool = False          # both sides >= target (our order counted) at this snapshot
    ref_yes: float | None = None
    ref_no: float | None = None

    @property
    def score(self) -> float:
        return self.score_yes + self.score_no if self.included else 0.0


def reward_share(yes: dict, no: dict, my_yes, my_no, target: float, df: float) -> RewardResult:
    """Snapshot score of our orders (``my_*`` = (price, size) or None) on the given book (which must
    NOT already contain our orders)."""
    by = _with_mine(yes, my_yes)
    bn = _with_mine(no, my_no)
    included = side_total(by) >= target and side_total(bn) >= target
    sy, ry = side_share(yes, my_yes, target, df)
    sn, rn = side_share(no, my_no, target, df)
    return RewardResult(sy, sn, included, ry, rn)


def interval_reward(res: RewardResult, pool_day: float, dt_s: float) -> float:
    """Dollars earned over ``dt_s`` seconds: pool/(2 x period) x snapshot score x time."""
    return pool_day / 86400.0 * dt_s * res.score / 2.0


def desired_price(own: dict[float, float], other: dict[float, float], target: float,
                  offset_ticks: int) -> float | None:
    """Where to rest a bid: the side's reference price (+offset ticks), kept strictly below the
    opposite side's implied ask so it can never cross (post-only behaviour)."""
    ref = reference_price(own, target)
    if ref is None:
        return None
    p = ref + offset_ticks * TICK
    if other:
        p = min(p, 1.0 - max(other) - TICK)
    p = round(min(p, 0.99), 4)
    return p if p >= TICK - 1e-9 else None


def jump_detected(window: list[tuple[float, float]], jump: float) -> bool:
    mids = [m for _, m in window]
    return len(mids) >= 2 and max(mids) - min(mids) >= jump - 1e-9


def pull_reasons(*, status: str, secs_to_close: float, pulled_until: float, now: float,
                 net: int, gross_total: int, size: int, cap: int, total_cap: int,
                 pull_hours: float) -> dict[str, str | None]:
    """{'yes': reason|None, 'no': reason|None}: why each side must not be quoted right now."""
    common = None
    if status != "active":
        common = f"status={status}"
    elif secs_to_close < pull_hours * 3600:
        common = "near_close"
    elif now < pulled_until:
        common = "cooldown_after_jump"
    out = {"yes": common, "no": common}
    for side, sign in (("yes", 1), ("no", -1)):
        if out[side]:
            continue
        new_net = net + sign * size
        if abs(new_net) > cap and abs(new_net) > abs(net):
            out[side] = "market_inventory_cap"
        elif gross_total - abs(net) + abs(new_net) > total_cap and abs(new_net) > abs(net):
            out[side] = "total_inventory_cap"
    return out


@dataclass
class Order:
    side: str                       # 'yes' | 'no'
    price: float                    # in this side's own dollars
    size: int
    placed_ts: float
    active_from: float
    thru: float = 0.0               # contracts traded strictly through our price since active
    at: float = 0.0                 # contracts traded at our price since active
    filled: int = 0


def order_fill_qty(o: Order, mode: str) -> int:
    """Total contracts this order has filled under ``mode`` ('cons' or 'opt'), given the volume
    accumulated by ``feed_trade``."""
    vol = o.thru + (o.at if mode == "opt" else 0.0)
    return min(o.size, int(math.floor(vol + 1e-9)))


def feed_trade(o: Order, ts: float, yes_price: float, count: float) -> None:
    """Accumulate a public trade (YES price, contracts) against a resting order."""
    if ts <= o.active_from:
        return
    if o.side == "yes":
        thru, at = yes_price < o.price - 1e-9, abs(yes_price - o.price) <= 1e-9
    else:
        no_leg = 1.0 - o.price      # a NO bid at n is a YES offer at 1-n
        thru, at = yes_price > no_leg + 1e-9, abs(yes_price - no_leg) <= 1e-9
    if thru:
        o.thru += count
    elif at:
        o.at += count


@dataclass
class Acct:
    """One simulated account (per market, per fill variant)."""
    mode: str
    yes_qty: int = 0
    no_qty: int = 0
    spent: float = 0.0              # fills cost + fees, ever
    payouts: float = 0.0            # settlement receipts
    fees: float = 0.0
    reward: float = 0.0
    fills: int = 0
    orders: dict = field(default_factory=lambda: {"yes": None, "no": None})
    reward_seconds_incl: float = 0.0
    seconds: float = 0.0

    @property
    def net(self) -> int:
        return self.yes_qty - self.no_qty

    def apply_fill(self, side: str, n: int, price: float, fee: float) -> None:
        if side == "yes":
            self.yes_qty += n
        else:
            self.no_qty += n
        self.spent += n * price + fee
        self.fees += fee
        self.fills += 1

    def inventory_value(self, mid_yes: float) -> float:
        return self.yes_qty * mid_yes + self.no_qty * (1.0 - mid_yes)

    def trade_pnl(self, mid_yes: float) -> float:
        return self.payouts - self.spent + self.inventory_value(mid_yes)

    def held_cost(self) -> float:
        """Capital tied up in positions: everything spent, less money already returned."""
        return max(0.0, self.spent - self.payouts) if (self.yes_qty or self.no_qty) else 0.0

    def resting_collateral(self) -> float:
        return sum(o.price * (o.size - o.filled) for o in self.orders.values() if o)

    def capital(self) -> float:
        return self.held_cost() + self.resting_collateral()

    def settle(self, result: str) -> float:
        win = self.yes_qty if result == "yes" else self.no_qty if result == "no" else 0
        self.payouts += float(win)
        pnl = self.payouts - self.spent
        self.yes_qty = self.no_qty = 0
        self.orders = {"yes": None, "no": None}
        return pnl


# ============================================================================ API plumbing
class PacedClient(KalshiClient):
    """Public-data client that spaces requests (shared budget) and counts them."""

    def __init__(self, max_rps: float):
        super().__init__(PROD_URL)
        self.min_gap = 1.0 / max_rps
        self._last = 0.0
        self.requests = 0

    def _request(self, method, endpoint, **kw):
        wait = self._last + self.min_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        self.requests += 1
        return super()._request(method, endpoint, **kw)

    def get_programs(self) -> list[dict]:
        out, cursor = [], ""
        while True:
            page = self._request("GET", "/incentive_programs",
                                 params={"status": "active", "limit": 10000, "cursor": cursor})
            out += page.get("incentive_programs", [])
            cursor = page.get("next_cursor") or ""
            if not cursor or not page.get("incentive_programs"):
                return out


def _t(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


@dataclass
class Prog:
    start: float
    end: float
    target: float
    df: float
    pool_day: float

    def active(self, now: float) -> bool:
        return self.start <= now < self.end


def parse_program(p: dict) -> Prog:
    start, end = _t(p["start_date"]), _t(p["end_date"])
    days = max((end - start) / 86400.0, 1e-9)
    return Prog(start, end, float(p["target_size_fp"]), p["discount_factor_bps"] / 10000.0,
                float(p["period_reward"]) * REWARD_UNIT / days)


def group_programs(raw: list[dict]) -> dict[str, list[Prog]]:
    out: dict[str, list[Prog]] = collections.defaultdict(list)
    for p in raw:
        if p.get("incentive_type", "liquidity") == "liquidity" and not p.get("paid_out"):
            out[p["market_ticker"]].append(parse_program(p))
    return out


# ============================================================================ selection
def balanced_order(series_cat: dict[str, str], rng: random.Random) -> list[str]:
    """Series tickers interleaved round-robin across categories (variety = less correlation)."""
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for s, c in series_cat.items():
        buckets[c].append(s)
    for b in buckets.values():
        rng.shuffle(b)
    out, cats = [], sorted(buckets)
    while any(buckets.values()):
        for c in cats:
            if buckets[c]:
                out.append(buckets[c].pop())
    return out


@dataclass
class Pick:
    ticker: str
    series: str
    event: str
    category: str
    frequency: str
    fee_type: str
    fee_multiplier: float
    close_ts: float
    pool_day: float
    mid: float
    spread: float
    depth_yes: float
    depth_no: float
    vol24: float
    early_close: bool
    progs: list = field(default_factory=list)


def select_markets(client: PacedClient, cfg: LipConfig, emit=lambda **k: None) -> list[Pick]:
    now = time.time()
    rng = random.Random(cfg.seed)
    raw = client.get_programs()
    progs = group_programs(raw)
    excl_cat = {c.strip().lower() for c in cfg.exclude_categories.split(",") if c.strip()}
    excl_freq = {c.strip() for c in cfg.exclude_frequencies.split(",") if c.strip()}
    pool_now = {t: sum(p.pool_day for p in ps if p.active(now)) for t, ps in progs.items()}
    cands = [t for t, v in pool_now.items()
             if v >= cfg.min_pool and "15M" not in t.split("-")[0]]
    by_series: dict[str, list[str]] = collections.defaultdict(list)
    for t in cands:
        by_series[t.split("-")[0]].append(t)
    emit(ev="select_start", programs=len(raw), candidates=len(cands), series=len(by_series))

    # Series metadata (category, frequency, fee) for a random sample of series.
    sample = list(by_series)
    rng.shuffle(sample)
    meta: dict[str, dict] = {}
    for s in sample[:cfg.max_probe]:
        try:
            m = client.get_series(s)
        except KalshiError:
            continue
        if (m.get("category", "") or "").lower() in excl_cat or m.get("frequency") in excl_freq:
            continue
        meta[s] = m
    order = balanced_order({s: meta[s].get("category", "?") for s in meta}, rng)

    picks: list[Pick] = []
    per_series: collections.Counter = collections.Counter()
    events: set[str] = set()
    probed = 0
    # Walk series round-robin by category; inside a series take its biggest pools first.
    queues = {s: sorted(by_series[s], key=lambda t: -pool_now[t]) for s in order}
    while len(picks) < cfg.n_markets and probed < cfg.max_probe and any(queues.values()):
        for s in order:
            if len(picks) >= cfg.n_markets or probed >= cfg.max_probe:
                break
            if not queues[s] or per_series[s] >= cfg.max_per_series:
                queues[s] = []
                continue
            t = queues[s].pop(0)
            probed += 1
            pk = _probe(client, cfg, t, s, meta[s], progs[t], pool_now[t], now, events)
            if pk:
                picks.append(pk)
                per_series[s] += 1
                events.add(pk.event)
    emit(ev="select_done", probed=probed, picked=len(picks))
    return picks


def _probe(client, cfg, t, s, meta, ps, pool, now, events) -> Pick | None:
    try:
        m = client.get_market(t)
    except KalshiError:
        return None
    event = m.get("event_ticker", t)
    if event in events or m.get("status") != "active" or m.get("market_type", "binary") != "binary":
        return None
    if m.get("price_level_structure", "linear_cent") != "linear_cent":
        return None
    if cfg.exclude_early_close and m.get("can_close_early"):
        return None
    close_ts = _t(m["close_time"])
    if close_ts - now < cfg.min_days * 86400:
        return None
    vol24 = float(m.get("volume_24h_fp", 0) or 0)
    if vol24 > cfg.max_vol24:
        return None
    try:
        yes, no = parse_book(client.get_orderbook(t, depth=100))
    except KalshiError:
        return None
    if not yes or not no:
        return None
    bid, ask = max(yes), 1.0 - max(no)
    spread, mid = ask - bid, (ask + bid) / 2
    if spread > cfg.max_spread + 1e-9 or not cfg.min_mid <= mid <= cfg.max_mid:
        return None
    target = ps[0].target
    # Both sides must be (nearly) at the target so that snapshots are included and the pool is live.
    if side_total(yes) + cfg.size < target or side_total(no) + cfg.size < target:
        return None
    if reference_price(yes, target) is None or reference_price(no, target) is None:
        return None
    return Pick(t, s, event, meta.get("category", "?"), meta.get("frequency", "?"),
                meta.get("fee_type", "quadratic"), float(meta.get("fee_multiplier", 1.0)),
                close_ts, pool, mid, spread, side_total(yes), side_total(no), vol24,
                bool(m.get("can_close_early")), ps)


# ============================================================================ runner
@dataclass
class MState:
    pick: Pick
    fees: FeeSchedule
    progs: list
    accts: dict
    status: str = "active"
    close_ts: float = 0.0
    mids: collections.deque = field(default_factory=collections.deque)
    mid: float | None = None
    last_book_ts: float | None = None
    trade_cursor: float = 0.0
    seen_trades: dict = field(default_factory=dict)
    pulled_until: float = 0.0
    last_snap_log: float = 0.0
    first_ts: float = 0.0
    pending_adv: list = field(default_factory=list)
    settled: bool = False
    fill_n: int = 0
    last_pull: dict = field(default_factory=dict)


class LipRunner:
    def __init__(self, cfg: LipConfig, client: PacedClient | None = None):
        self.cfg = cfg
        self.client = client or PacedClient(cfg.max_rps)
        self.stop = False
        Path(cfg.dir).mkdir(parents=True, exist_ok=True)
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(cfg.dir) / f"events-{self.session}.jsonl"
        self._fh = self.path.open("a")
        self.markets: dict[str, MState] = {}
        self.heap: list = []

    # -- logging
    def emit(self, **ev) -> None:
        ev["t"] = round(time.time(), 2)
        self._fh.write(json.dumps(ev, default=str) + "\n")
        self._fh.flush()

    # -- setup
    def setup(self) -> None:
        now = time.time()
        self.emit(ev="config", session=self.session, **asdict(self.cfg), reward_unit=REWARD_UNIT)
        picks = select_markets(self.client, self.cfg, self.emit)
        for i, pk in enumerate(picks):
            ms = MState(pk, FeeSchedule(pk.fee_multiplier, "maker" in pk.fee_type), pk.progs,
                        {v: Acct(v) for v in VARIANTS}, close_ts=pk.close_ts,
                        trade_cursor=now, first_ts=now)
            self.markets[pk.ticker] = ms
            d = asdict(pk)
            d.pop("progs")
            self.emit(ev="select", **d, programs=[asdict(p) for p in pk.progs],
                      makers_pay=ms.fees.makers_pay)
            k = len(picks) or 1
            for kind, every in (("book", self.cfg.book_poll_s), ("trades", self.cfg.trades_poll_s),
                                ("status", self.cfg.status_poll_s)):
                self.schedule(kind, pk.ticker, now + every * (i / k) + 1.0)
        self.schedule("programs", "", now + self.cfg.programs_poll_s)

    def schedule(self, kind: str, ticker: str, due: float) -> None:
        heapq.heappush(self.heap, (due, kind, ticker))

    # -- main loop
    def run(self) -> None:
        self.setup()
        print(f"[lip] session {self.session}: {len(self.markets)} markets, log {self.path}", flush=True)
        while not self.stop and self.heap:
            due, kind, tk = heapq.heappop(self.heap)
            delay = due - time.time()
            while delay > 0 and not self.stop:
                time.sleep(min(delay, 1.0))
                delay = due - time.time()
            if self.stop:
                break
            try:
                getattr(self, f"task_{kind}")(tk)
            except KalshiError as exc:
                log.warning("%s %s failed: %s", kind, tk, exc)
                print(f"[lip] {kind} {tk} failed: {exc}", flush=True)
                time.sleep(5)
            except Exception:                      # never die on a single bad market
                log.exception("%s %s crashed", kind, tk)
            every = {"book": self.cfg.book_poll_s, "trades": self.cfg.trades_poll_s,
                     "status": self.cfg.status_poll_s, "programs": self.cfg.programs_poll_s}[kind]
            ms = self.markets.get(tk)
            if kind == "programs" or (ms and not ms.settled):
                self.schedule(kind, tk, time.time() + every)
        for tk in self.markets:
            self.log_snap(self.markets[tk], force=True)
        self.emit(ev="stop", requests=self.client.requests)

    # -- tasks
    def task_programs(self, _tk: str) -> None:
        grouped = group_programs(self.client.get_programs())
        for tk, ms in self.markets.items():
            if tk in grouped:
                ms.progs = grouped[tk]

    def task_status(self, tk: str) -> None:
        ms = self.markets[tk]
        m = self.client.get_market(tk)
        ms.status = m.get("status", "?")
        ms.close_ts = _t(m["close_time"]) if m.get("close_time") else ms.close_ts
        result = m.get("result") or ""
        if result in ("yes", "no"):
            self.flush_trades(ms)
            for a in ms.accts.values():
                pnl = a.settle(result)
                self.emit(ev="settle", tk=tk, v=a.mode, result=result, trade_pnl=round(pnl, 4))
            ms.settled = True
            self.log_snap(ms, force=True)

    def task_trades(self, tk: str) -> None:
        self.flush_trades(self.markets[tk])

    def task_book(self, tk: str) -> None:
        ms = self.markets[tk]
        now = time.time()
        yes, no = parse_book(self.client.get_orderbook(tk, depth=100))
        if not yes or not no:
            return
        bid, ask = max(yes), 1.0 - max(no)
        mid = round((bid + ask) / 2, 4)
        # 1) reward for the interval since the last poll, orders as they stood during it
        if ms.last_book_ts is not None:
            dt = min(now - ms.last_book_ts, 3 * self.cfg.book_poll_s)
            self.accrue(ms, yes, no, dt)
        ms.last_book_ts, ms.mid = now, mid
        ms.mids.append((now, mid))
        while ms.mids and ms.mids[0][0] < now - self.cfg.jump_window_s:
            ms.mids.popleft()
        self.check_adverse(ms, now, mid)
        # 2) guard + quote decisions
        if jump_detected(list(ms.mids), self.cfg.jump) and now >= ms.pulled_until:
            ms.pulled_until = now + self.cfg.cooldown_s
            self.emit(ev="pull", tk=tk, why="mid_jump", mid=mid,
                      window=[round(m, 3) for _, m in ms.mids][-6:])
        self.requote(ms, yes, no, now)
        self.log_snap(ms)

    # -- reward
    def accrue(self, ms: MState, yes, no, dt: float) -> None:
        now = time.time()
        for a in ms.accts.values():
            my = {s: ((o.price, o.size - o.filled) if o and now >= o.active_from else None)
                  for s, o in a.orders.items()}
            a.seconds += dt
            for p in ms.progs:
                if not p.active(now):
                    continue
                res = reward_share(yes, no, my["yes"], my["no"], p.target, p.df)
                a.reward += interval_reward(res, p.pool_day, dt)
                if res.included and (my["yes"] or my["no"]):
                    a.reward_seconds_incl += dt / max(1, len([q for q in ms.progs if q.active(now)]))

    # -- quoting
    def gross(self, variant: str) -> int:
        return sum(abs(m.accts[variant].net) for m in self.markets.values() if not m.settled)

    def requote(self, ms: MState, yes, no, now: float) -> None:
        cfg, target = self.cfg, None
        active = [p for p in ms.progs if p.active(now)]
        target = active[0].target if active else (ms.progs[0].target if ms.progs else 1000.0)
        flushed = False
        for a in ms.accts.values():
            why = pull_reasons(status=ms.status, secs_to_close=ms.close_ts - now,
                               pulled_until=ms.pulled_until, now=now, net=a.net,
                               gross_total=self.gross(a.mode), size=cfg.size, cap=cfg.cap,
                               total_cap=cfg.total_cap, pull_hours=cfg.pull_hours)
            if not active:
                why = {"yes": "no_active_program", "no": "no_active_program"}
            for side, own, other in (("yes", yes, no), ("no", no, yes)):
                want = None if why[side] else desired_price(own, other, target, cfg.offset_ticks)
                cur = a.orders[side]
                if cur and want is not None and abs(cur.price - want) < cfg.requote_ticks * TICK - 1e-9:
                    continue
                if cur is None and want is None:
                    if why[side] and ms.last_pull.get((a.mode, side)) != why[side]:
                        ms.last_pull[(a.mode, side)] = why[side]
                        self.emit(ev="pull", tk=ms.pick.ticker, v=a.mode, side=side, why=why[side])
                    continue
                if cur and not flushed:                    # settle fills before touching orders
                    self.flush_trades(ms)
                    flushed = True
                    cur = a.orders[side]
                if want is None:
                    a.orders[side] = None
                    self.emit(ev="pull", tk=ms.pick.ticker, v=a.mode, side=side,
                              why=why[side] or "no_reference")
                    continue
                if cur and abs(cur.price - want) < cfg.requote_ticks * TICK - 1e-9:
                    continue
                a.orders[side] = Order(side, want, cfg.size, now, now + cfg.latency_s)
                ms.last_pull.pop((a.mode, side), None)
                self.emit(ev="quote", tk=ms.pick.ticker, v=a.mode, side=side, px=want,
                          size=cfg.size, mid=ms.mid)

    # -- fills
    def flush_trades(self, ms: MState) -> None:
        tk = ms.pick.ticker
        trades = self.client.get_trades(tk, int(ms.trade_cursor) - 5)
        fresh = [t for t in trades if t["trade_id"] not in ms.seen_trades]
        fresh.sort(key=lambda t: t["created_time"])
        for t in fresh:
            ts = _t(t["created_time"])
            ms.seen_trades[t["trade_id"]] = ts
            ms.trade_cursor = max(ms.trade_cursor, ts)
            px, n = float(t["yes_price_dollars"]), float(t["count_fp"])
            for a in ms.accts.values():
                for o in a.orders.values():
                    if o:
                        feed_trade(o, ts, px, n)
                self.settle_fills(ms, a, ts)
        horizon = ms.trade_cursor - 600
        ms.seen_trades = {k: v for k, v in ms.seen_trades.items() if v >= horizon}

    def settle_fills(self, ms: MState, a: Acct, ts: float) -> None:
        for side, o in list(a.orders.items()):
            if not o:
                continue
            q = order_fill_qty(o, a.mode)
            if q <= 0:
                continue
            fee = ms.fees.maker_fee(o.price, q)
            a.apply_fill(side, q, o.price, fee)
            ms.fill_n += 1
            fid = f"{ms.pick.ticker}:{a.mode}:{ms.fill_n}"
            a.orders[side] = None                      # remainder cancelled; re-quoted next poll
            self.emit(ev="fill", id=fid, tk=ms.pick.ticker, v=a.mode, side=side, px=o.price, n=q,
                      fee=fee, trade_ts=round(ts, 2), mid=ms.mid,
                      yes_qty=a.yes_qty, no_qty=a.no_qty)
            for label, secs in HORIZONS:
                ms.pending_adv.append((ts + secs, fid, a.mode, side, o.price, label))

    def check_adverse(self, ms: MState, now: float, mid: float) -> None:
        keep = []
        for due, fid, v, side, px, label in ms.pending_adv:
            if now < due:
                keep.append((due, fid, v, side, px, label))
                continue
            own = mid if side == "yes" else 1.0 - mid
            self.emit(ev="adv", id=fid, tk=ms.pick.ticker, v=v, h=label, side=side,
                      markout=round(own - px, 4), mid=mid)
        ms.pending_adv = keep

    # -- state log
    def log_snap(self, ms: MState, force: bool = False) -> None:
        now = time.time()
        if not force and now - ms.last_snap_log < self.cfg.snap_log_s:
            return
        ms.last_snap_log = now
        mid = ms.mid if ms.mid is not None else ms.pick.mid
        rec = {"ev": "snap", "tk": ms.pick.ticker, "cat": ms.pick.category, "mid": mid,
               "pool": round(sum(p.pool_day for p in ms.progs if p.active(now)), 2),
               "start": round(ms.first_ts, 0), "settled": ms.settled}
        for v, a in ms.accts.items():
            rec[v] = {"reward": round(a.reward, 5), "fills": a.fills, "yes": a.yes_qty,
                      "no": a.no_qty, "fees": round(a.fees, 4),
                      "tpnl": round(a.trade_pnl(mid), 4), "cap": round(a.capital(), 2),
                      "quoting": sum(1 for o in a.orders.values() if o),
                      "incl_s": round(a.reward_seconds_incl, 0), "sec": round(a.seconds, 0)}
        self.emit(**rec)


# ============================================================================ report
def load_events(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        for line in Path(f).read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass            # torn last line while the recorder is writing
    return out


def summarize(events: list[dict], variant: str = "cons") -> dict:
    """Aggregate event logs -> {'markets': {...}, 'total': {...}, 'categories': {...}}."""
    snaps: dict[str, dict] = {}
    first: dict[str, float] = {}
    caps: dict[str, list[float]] = collections.defaultdict(list)
    cats: dict[str, str] = {}
    marks: dict[tuple, list[float]] = collections.defaultdict(list)
    sess_of_fill: dict[str, str] = {}
    for e in events:
        tk = e.get("tk")
        if e["ev"] == "snap":
            snaps[tk] = e
            first.setdefault(tk, e.get("start", e["t"]))
            cats[tk] = e["cat"]
            caps[tk].append(e[variant]["cap"])
        elif e["ev"] == "adv" and e["v"] == variant:
            marks[(tk, e["h"])].append(e["markout"])
    markets = {}
    for tk, s in snaps.items():
        v = s[variant]
        days = max((s["t"] - first[tk]) / 86400.0, 1e-9)
        pnl = v["reward"] + v["tpnl"]
        markets[tk] = {
            "cat": cats[tk], "pool": s["pool"], "days": days, "reward": v["reward"],
            "reward_day": v["reward"] / days, "fills": v["fills"], "inv": v["yes"] - v["no"],
            "tpnl": v["tpnl"], "net": pnl, "net_day": pnl / days,
            "cap": statistics.mean(caps[tk]) if caps[tk] else 0.0,
            "mark": {h: marks[(tk, h)] for h, _ in HORIZONS}}
    return {"markets": markets}


def _agg(rows: list[dict]) -> dict:
    cap = sum(r["cap"] for r in rows)
    net_day = sum(r["net_day"] for r in rows)
    mk = {h: [x for r in rows for x in r["mark"][h]] for h, _ in HORIZONS}
    return {"n": len(rows), "pool": sum(r["pool"] for r in rows),
            "reward_day": sum(r["reward_day"] for r in rows),
            "fills": sum(r["fills"] for r in rows), "tpnl": sum(r["tpnl"] for r in rows),
            "net_day": net_day, "cap": cap, "days": statistics.mean([r["days"] for r in rows]),
            "mark": {h: (100 * statistics.mean(v) if v else float("nan"), len(v)) for h, v in mk.items()},
            "apy_net": 365 * net_day / cap if cap else float("nan"),
            "apy_reward": 365 * sum(r["reward_day"] for r in rows) / cap if cap else float("nan")}


def print_report(pattern: str) -> None:
    events = load_events(pattern)
    if not any(e["ev"] == "snap" for e in events):
        print(f"No snapshot events yet in {pattern}")
        return
    t0 = min(e["t"] for e in events)
    t1 = max(e["t"] for e in events)
    print(f"LIP paper test: {datetime.fromtimestamp(t0):%Y-%m-%d %H:%M} -> "
          f"{datetime.fromtimestamp(t1):%Y-%m-%d %H:%M}  ({(t1 - t0) / 3600:.1f} h of data)\n")
    for variant, label in (("cons", "CONSERVATIVE fills (strictly through)"),
                           ("opt", "OPTIMISTIC fills (at-or-through)")):
        res = summarize(events, variant)["markets"]
        print(f"=== {label} ===")
        print(f"{'market':44s} {'cat':14s} {'pool$/d':>7s} {'days':>5s} {'rew$/d':>7s} {'fills':>5s} "
              f"{'inv':>5s} {'mk1m':>6s} {'mk1h':>6s} {'mk1d':>6s} {'tpnl$':>7s} {'net$/d':>7s}")

        def mk(r, h):
            v = r["mark"][h]
            return f"{100 * statistics.mean(v):6.1f}" if v else "     -"
        for tk, r in sorted(res.items(), key=lambda kv: -kv[1]["net_day"]):
            print(f"{tk[:44]:44s} {r['cat'][:14]:14s} {r['pool']:7.1f} {r['days']:5.2f} "
                  f"{r['reward_day']:7.3f} {r['fills']:5d} {r['inv']:5d} {mk(r, '1m')} {mk(r, '1h')} "
                  f"{mk(r, '1d')} {r['tpnl']:7.2f} {r['net_day']:7.3f}")
        by_cat: dict[str, list] = collections.defaultdict(list)
        for r in res.values():
            by_cat[r["cat"]].append(r)
        print(f"\n{'category':30s} {'n':>3s} {'rew$/d':>8s} {'fills':>5s} {'mk1m c':>7s} {'mk1h c':>7s} "
              f"{'tpnl$':>8s} {'net$/d':>8s}")
        for c, rows in sorted(by_cat.items()):
            a = _agg(rows)
            print(f"{c[:30]:30s} {a['n']:3d} {a['reward_day']:8.3f} {a['fills']:5d} "
                  f"{a['mark']['1m'][0]:7.2f} {a['mark']['1h'][0]:7.2f} {a['tpnl']:8.2f} {a['net_day']:8.3f}")
        a = _agg(list(res.values()))
        print(f"\nTOTAL  markets={a['n']}  pools=${a['pool']:.0f}/day  reward est=${a['reward_day']:.3f}/day  "
              f"fills={a['fills']}  trade P&L=${a['tpnl']:.2f}  net=${a['net_day']:.3f}/day")
        print(f"       avg capital at risk=${a['cap']:.0f} (held cost + resting bid collateral)")
        print(f"       mark-out per contract (cents, own-side mid minus fill price; negative = adverse): "
              + "  ".join(f"{h}={v:.2f} (n={n})" for h, (v, n) in a["mark"].items()))
        print(f"       conservative annualised return on capital: net {100 * a['apy_net']:.1f}%  "
              f"| reward only {100 * a['apy_reward']:.1f}%\n")
    print(CAVEATS)


CAVEATS = """Caveats (read before believing any number above)
 * Short samples: under ~1 week, daily figures are dominated by book noise; fills are rare events
   so mark-outs rest on a handful of fills; per-market rows are NOT independent (same program
   settings, correlated news, shared events); do not extrapolate a reward rate from a few hours.
 * Rewards assume: pool = period_reward x 1e-4 dollars; all public orders count as competitors
   (market makers are excluded in reality, so true share is probably higher but the inclusion
   test is also less certain); book sampled every ~45 s, not once per second; our own orders
   never move other participants; no payout rounding. The real LIP payout is NOT observed here.
 * Fills: cons = strictly-through only; opt = at-or-through. Real fills lie between. Adverse selection
   on fills is measured only from the fills that occurred; with few fills it is unknown, not zero.
 * Capital counts held cost plus collateral for both resting bids; the return is a paper figure."""


# ============================================================================ CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    d = LipConfig()
    p.add_argument("--report", action="store_true", help="print results from the logs and exit")
    p.add_argument("--select-only", action="store_true", help="show the market selection and exit")
    p.add_argument("--pattern", default=None, help="log glob for --report (default <dir>/events-*.jsonl)")
    for f, v in asdict(d).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, action="store_true", default=v)
        else:
            p.add_argument(flag, type=type(v), default=v)


def cmd_lip_paper(args: argparse.Namespace) -> None:
    cfg = LipConfig(**{k: getattr(args, k) for k in asdict(LipConfig())})
    if args.report:
        print_report(args.pattern or f"{cfg.dir}/events-*.jsonl")
        return
    if args.select_only:
        picks = select_markets(PacedClient(cfg.max_rps), cfg)
        for p in picks:
            print(f"{p.ticker:46s} {p.category[:14]:14s} pool ${p.pool_day:6.1f}/d mid {p.mid:.2f} "
                  f"spr {p.spread:.2f} depth {p.depth_yes:.0f}/{p.depth_no:.0f} vol24 {p.vol24:.0f} "
                  f"close {datetime.fromtimestamp(p.close_ts):%Y-%m-%d} fee={p.fee_type}")
        print(f"{len(picks)} markets")
        return
    runner = LipRunner(cfg)

    def _stop(*_):
        runner.stop = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logging.basicConfig(level=logging.WARNING)
    runner.run()
