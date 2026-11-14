"""Terminal dashboard. Renders a ``State`` object; it never touches the network or the
trading path, so showing more detail cannot slow the bot down.

Panels: order book (with our own resting orders marked), market + sparklines, a "why"
panel (model inputs and the edge arithmetic behind every decision), the trade table
(when / why / at what price / outcome), per-style session stats, and an event log."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class TradeRow:
    ts: float
    ticker: str
    style: str
    side: str
    count: int
    price: float
    quote: float
    fee: float
    edge: float
    secs_left: float
    why: str
    status: str                      # sent | resting | filled | missed | expired | settled
    pnl: Optional[float] = None
    result: str = ""


@dataclass
class Stats:
    trades: int = 0
    open: int = 0                    # filled, waiting for the window to close
    at_risk: float = 0.0             # $ paid (incl. fees) on open positions
    settled: int = 0
    wins: int = 0
    pnl: float = 0.0
    fees: float = 0.0


@dataclass
class State:
    mode: str = "paper"
    env: str = "live market data, simulated fills"
    balance: Optional[float] = None
    ticker: str = "-"
    strike: float = 0.0
    tau_min: float = 0.0
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    ref_price: float = 0.0
    ref_open: float = 0.0
    ref_move: float = 0.0
    ref_age_s: float = 0.0
    vol30: float = 0.0
    status: str = "starting up"
    asks: list = field(default_factory=list)       # [(yes price, size)] nearest the spread first
    bids: list = field(default_factory=list)       # [(yes price, size)] nearest the spread first
    resting: list = field(default_factory=list)    # [(style, side, price, count)] our open orders
    decisions: dict = field(default_factory=dict)  # style -> Decision
    exit_line: str = ""              # early-exit experiment: sell-vs-hold check for open positions
    actions: dict = field(default_factory=dict)    # style -> what the engine is doing about the signal
    open_positions: list = field(default_factory=list)  # [(style, close_ts, $ cost)] awaiting settlement
    blocked: dict = field(default_factory=dict)    # style -> risk pause reason
    stats: dict = field(default_factory=dict)      # style -> Stats
    history: deque = field(default_factory=lambda: deque(maxlen=150))   # (mid, ref_move)
    trades: deque = field(default_factory=lambda: deque(maxlen=40))
    events: deque = field(default_factory=lambda: deque(maxlen=40))

    def stat(self, style: str) -> Stats:
        return self.stats.setdefault(style, Stats())

    def log(self, msg: str) -> None:
        self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")


# --------------------------------------------------------------------------- helpers
def spark(values) -> str:
    vals = list(values)
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in vals)


def _book_panel(s: State) -> Panel:
    ours = {}
    for style, side, price, count in s.resting:           # our NO bid at n sits at YES ask 1-n
        key = ("ask", round(1 - price, 3)) if side == "no" else ("bid", round(price, 3))
        ours[key] = f"◄ {style.upper()} {side.upper()} x{count}"
    biggest = max([sz for _, sz in s.asks + s.bids] or [1])
    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", width=7)
    t.add_column(justify="right", width=8)
    t.add_column(width=8)
    t.add_column()
    for price, size in reversed(s.asks):
        t.add_row(f"[red]{price:.3f}[/]", f"{size:,.0f}", "[red]" + "█" * max(1, int(8 * size / biggest)) + "[/]",
                  f"[yellow]{ours.get(('ask', round(price, 3)), '')}[/]")
    spread = (s.yes_ask - s.yes_bid) if s.yes_ask and s.yes_bid else 0
    t.add_row("", f"[dim]{spread:.3f}[/]", "[dim]spread[/]", "")
    for price, size in s.bids:
        t.add_row(f"[green]{price:.3f}[/]", f"{size:,.0f}", "[green]" + "█" * max(1, int(8 * size / biggest)) + "[/]",
                  f"[yellow]{ours.get(('bid', round(price, 3)), '')}[/]")
    return Panel(t, title="YES book  (asks ↑ bids ↓)")


def _market_panel(s: State) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_row("market", f"{s.ticker[-14:]}   {max(s.tau_min, 0) * 60:.0f}s left   [dim]{s.status}[/]")
    t.add_row("strike", f"${s.strike:,.2f}" if s.strike else "TBD (not published yet)")
    t.add_row("reference", f"${s.ref_price:,.2f}  (age {s.ref_age_s:.1f}s, vol {s.vol30:.2f}$/min)")
    t.add_row("move since open", f"[bold]{s.ref_move:+.2f}$[/]  (opened at ${s.ref_open:,.2f})")
    mids = [m for m, _ in s.history][-40:]
    moves = [x for _, x in s.history][-40:]
    if mids:
        t.add_row("kalshi mid", f"{spark(mids[-24:])} {min(mids):.2f}-{max(mids):.2f} now {mids[-1]:.3f}")
        t.add_row("reference move", f"{spark(moves[-24:])} {min(moves):+.1f}..{max(moves):+.1f} now {moves[-1]:+.1f}")
    if s.exit_line:
        t.add_row("exit check", s.exit_line)
    return Panel(t, title="market  (bars = last 24s, relative low-to-high)")


def _why_panel(s: State) -> Panel:
    if not s.decisions:
        return Panel("waiting for the first signal…", title="why")
    first = next(iter(s.decisions.values())).detail
    out = []
    if "features" in first:
        f = Table(box=None, padding=(0, 1))
        for col in ("model input", "value", "weight", "→ log-odds"):
            f.add_column(col, justify="right" if col != "model input" else "left")
        names = {"logit_mid": "market log-odds", "z_vol": "move / vol (z)",
                 "m3n": "3-min momentum", "m5n": "5-min momentum"}
        for k, v in first["features"].items():
            f.add_row(names[k], f"{v['value']:+.2f}", f"{v['weight']:+.2f}", f"{v['contribution']:+.2f}")
        out += [f, Text(f"model P(YES) = {first['p_yes']:.3f}   vs   market mid {first['mid']:.3f}", style="bold")]
    for style, d in s.decisions.items():
        x = d.detail
        verdict = "[green bold]SIGNAL[/]" if d.side else "[dim]no signal[/]"
        out.append(Text.from_markup(f"\n[bold]{style.upper()}[/]  {verdict}: {d.reason}"))
        if d.side:
            act = s.actions.get(style) or "evaluating"
            color = "green" if act.startswith(("taker order", "maker order")) else "yellow"
            out.append(Text.from_markup(f"  [{color}]-> {act}[/]"))
        if "edge_yes" in x:
            need = x["min_edge"]
            for side, p_side in (("YES", x["p_yes"]), ("NO", 1 - x["p_yes"])):
                k = side.lower()
                edge = x[f"edge_{k}"]
                mark = "[green]✓[/]" if edge >= need else " "
                out.append(Text.from_markup(
                    f"  {mark} {side:<3} win-prob {p_side:.3f} − price {x[f'{k}_price']:.3f} − fee "
                    f"{x[f'fee_{k}']:.3f} = [bold]{edge:+.3f}[/] (need ≥ {need:.2f})"))
    return Panel(Group(*out), title="why (model inputs → edge after fees)")


def _trades_panel(s: State) -> Panel:
    t = Table(box=None, padding=(0, 1), expand=True)
    widths = {"time": 8, "left": 5, "style": 6, "side": 4, "qty": 3, "quote": 6, "fill": 6, "fee": 5, "result": 15}
    for col, w in widths.items():
        t.add_column(col, width=w, no_wrap=True, justify="left" if col in ("time", "style", "side", "result") else "right")
    t.add_column("why", ratio=1, no_wrap=True, overflow="ellipsis")
    for r in list(s.trades)[:8]:
        color = {"settled": "green" if (r.pnl or 0) > 0 else "red", "filled": "cyan",
                 "resting": "yellow", "sent": "yellow", "exiting": "yellow"}.get(r.status, "dim")
        result = f"{r.result.upper()} {r.pnl:+.2f}$" if r.status == "settled" else r.status
        fill = f"{r.price:.3f}" if r.status in ("filled", "settled") else "-"
        t.add_row(time.strftime("%H:%M:%S", time.localtime(r.ts)), f"{r.secs_left:.0f}s", r.style,
                  r.side.upper(), str(r.count), f"{r.quote:.3f}", fill, f"{r.fee:.2f}",
                  f"[{color}]{result}[/]", r.why)
    return Panel(t if s.trades else "no trades yet", title="trades: when · why · price · outcome")


def _session_panel(s: State) -> Panel:
    t = Table(box=None, padding=(0, 1))
    for col in ("style", "fills", "open", "settled", "win%", "P&L", "fees"):
        t.add_column(col, justify="left" if col == "style" else "right", no_wrap=True)
    for style, st in s.stats.items():
        win = f"{st.wins / st.settled:.0%}" if st.settled else "-"
        t.add_row(style, str(st.trades), str(st.open), str(st.settled), win,
                  f"[{'green' if st.pnl >= 0 else 'red'}]{st.pnl:+.2f}$[/]", f"{st.fees:.2f}$")
    notes = ["[dim]P&L = settled trades only; fees paid at fill[/]"]
    now = time.time()
    for style in s.stats:
        mine = [(c, cost) for st_, c, cost in s.open_positions if st_ == style]
        if mine:
            nxt = min(c for c, _ in mine)
            notes.append(f"{style}: {len(mine)} open, ${sum(x for _, x in mine):.2f} at risk, "
                         f"settles {time.strftime('%H:%M:%S', time.localtime(nxt))} (in {max(0, nxt - now):.0f}s)")
    notes += [f"[red]{st} paused: {why}[/]" for st, why in s.blocked.items() if why]
    return Panel(Group(t, *[Text.from_markup(n) for n in notes]), title="session")


def render(s: State):
    color = "red bold" if s.mode == "live" else "yellow"
    bal = "n/a" if s.balance is None else f"${s.balance:,.2f}"
    head = Panel(f"[{color}]{s.mode.upper()}[/] · {s.env} · balance {bal}   (Ctrl-C to stop)", title="kalshi-bot")
    events = Panel("\n".join(list(s.events)[:5]) or "(none yet)", title="events")
    layout = Layout()
    layout.split_column(Layout(head, size=3), Layout(name="top", size=19), Layout(name="row2", size=9),
                        Layout(_trades_panel(s), size=11), Layout(events, minimum_size=5))
    layout["top"].split_row(Layout(_book_panel(s), ratio=4), Layout(_why_panel(s), ratio=7))
    layout["row2"].split_row(Layout(_market_panel(s), ratio=1), Layout(_session_panel(s), ratio=1))
    return layout


def safe_renderable(state: State, render_fn=None):
    """Frame source for Live. The trading thread mutates ``state`` while this renders, so an
    occasional torn read is possible; an exception here would kill Rich's refresh thread and
    freeze the screen. Keep showing the last good frame instead."""
    render_fn = render_fn or render
    last = [Text("starting…")]

    def frame():
        try:
            last[0] = render_fn(state)
        except Exception:
            pass
        return last[0]
    return frame


def live_view(state: State, enabled: bool = True):
    """Context manager: a live-updating full-screen dashboard, or a no-op if disabled."""
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    return Live(get_renderable=safe_renderable(state), refresh_per_second=4, screen=True)
