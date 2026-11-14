import numpy as np

from kalshi_bot.brokers import PaperBroker, tick_ceil, tick_floor
from kalshi_bot.config import Config
from kalshi_bot.engine import parse_book
from kalshi_bot.fees import FeeSchedule
from kalshi_bot.maker import crossing_volume
from kalshi_bot.model import Model, design
from kalshi_bot.report import ci
from kalshi_bot.risk import RiskManager
from kalshi_bot.strategy import Snapshot, decide, explain_trade

# weights: market log-odds 1.0, z 1.0, no momentum
MODEL = Model(weights=(1.0, 1.0, 0.0, 0.0), intercept=0.0)
CFG = Config(contracts=10, min_net_edge=0.03)
FEES = FeeSchedule()


def snap(**kw):
    base = dict(yes_bid=0.49, yes_ask=0.51, yes_bid_size=500, yes_ask_size=500, tau_min=5,
                ref_move=0.0, ref_age_s=1, vol30=1.0, m3=0.0, m5=0.0)
    return Snapshot(**{**base, **kw})


# ---------------------------------------------------------------- fees
def test_fee_peaks_at_fifty_cents_and_rounds_up():
    assert FEES.taker_fee(0.5, 100) == 1.75
    assert FEES.taker_fee(0.5, 1) == 0.02
    assert FEES.taker_fee(0.95, 100) < FEES.taker_fee(0.5, 100)


def test_maker_fee_free_unless_series_charges_it():
    assert FEES.maker_fee(0.5, 100) == 0
    assert FeeSchedule.from_series({"fee_type": "quadratic_with_maker_fees"}).maker_fee(0.5, 100) == 0.44


# --------------------------------------------------------------- model
def test_model_scalar_and_array_paths_agree():
    m = Model(weights=(0.9, 0.34, 0.09, 0.07), intercept=0.01)
    args = (np.array([0.3, 0.6]), np.array([2.0, -1.0]), np.array([5.0, 9.0]),
            np.array([1.2, 0.8]), np.array([0.5, -0.5]), np.array([1.0, -2.0]))
    arr = m.predict(*args)
    for i in range(2):
        assert abs(arr[i] - m.predict(*[a[i] for a in args])) < 1e-12


def test_design_floors_tiny_volatility():
    assert np.isfinite(design(0.5, 1.0, 5.0, 0.0, 0.0, 0.0)).all()


def test_explain_contributions_sum_to_log_odds():
    m = Model(weights=(0.9, 0.34, 0.09, 0.07), intercept=0.0)
    a = (0.4, 1.5, 6.0, 1.0, 0.3, 0.8)
    total = sum(v["contribution"] for v in m.explain(*a).values())
    assert abs(1 / (1 + np.exp(-total)) - m.predict(*a)) < 1e-12


# ------------------------------------------------------------ strategy
def test_no_trade_when_model_agrees_with_market():
    assert decide(snap(), MODEL, FEES, CFG).side is None


def test_buys_yes_when_reference_rallied_but_market_has_not_moved():
    d = decide(snap(ref_move=2.0), MODEL, FEES, CFG)
    assert d.side == "yes" and d.edge >= CFG.min_net_edge
    assert "ref +2.0$" in explain_trade(d)


def test_buys_no_when_reference_dropped():
    assert decide(snap(ref_move=-2.0), MODEL, FEES, CFG).side == "no"


def test_guards_skip_trades():
    assert "stale" in decide(snap(ref_move=2, ref_age_s=60), MODEL, FEES, CFG).reason
    assert decide(snap(ref_move=2, tau_min=0.2), MODEL, FEES, CFG).side is None
    assert decide(snap(ref_move=2, yes_bid=0.40), MODEL, FEES, CFG).reason == "spread too wide"
    assert "size" in decide(snap(ref_move=2, yes_ask_size=1), MODEL, FEES, CFG).reason


def test_maker_rests_inside_the_book_and_pays_no_fee():
    d = decide(snap(ref_move=2.0), MODEL, FEES, CFG, style="maker")
    assert d.side == "yes"
    assert 0.49 < d.price < 0.51 and d.detail["fee_yes"] == 0   # better than the 0.51 ask, no fee
    t = decide(snap(ref_move=2.0), MODEL, FEES, CFG, style="taker")
    assert d.edge > t.edge                                       # cheaper entry => more edge


def test_maker_ignores_touch_size():
    assert decide(snap(ref_move=2, yes_ask_size=0), MODEL, FEES, CFG, style="maker").side == "yes"


# --------------------------------------------------------------- risk
def test_risk_sizing_caps_by_bankroll_and_config():
    r = RiskManager(Config(contracts=10, risk_per_trade_pct=0.005), bankroll=1000)
    assert r.size(0.50) == 10                    # $5 budget / $0.50 = 10, equals cap
    assert RiskManager(Config(contracts=10), bankroll=200).size(0.50) == 2
    assert RiskManager(Config(contracts=10), bankroll=10).size(0.50) == 0


def test_daily_loss_stop_and_cooldown():
    cfg = Config(daily_loss_pct=0.02, max_consecutive_losses=3, cooldown_min=30)
    r = RiskManager(cfg, bankroll=1000)
    assert r.block_reason(now=0) == ""
    for _ in range(3):
        r.record(-1.0, now=100)
    assert "cooldown" in r.block_reason(now=200)
    assert r.block_reason(now=100 + 30 * 60 + 1) == ""
    r.record(-20.0, now=5000)
    assert "daily loss" in r.block_reason(now=6000)


def test_win_resets_consecutive_losses():
    r = RiskManager(Config(), bankroll=1000)
    r.record(-1, now=1); r.record(-1, now=2); r.record(+1, now=3); r.record(-1, now=4)
    assert r.block_reason(now=5) == ""


# -------------------------------------------------------------- brokers
def test_paper_fill_respects_limit_and_settles_correctly():
    market = {"ticker": "T", "yes_ask_dollars": "0.60", "yes_bid_dollars": "0.58",
              "yes_ask_size_fp": "100", "yes_bid_size_fp": "100"}
    broker = PaperBroker(FEES)
    assert broker.buy(market, "yes", 10, limit=0.59) is None
    fill = broker.buy(market, "no", 10, limit=0.43)
    assert fill and abs(fill.price - 0.42) < 1e-9
    assert round(fill.settle_pnl("no"), 4) == round(10 - fill.cost, 4)
    assert round(fill.settle_pnl("yes"), 4) == round(-fill.cost, 4)


def test_tick_rounding():
    assert tick_ceil(0.5512) == 0.56 and tick_ceil(0.9123) == 0.913 and tick_ceil(0.0456) == 0.046
    assert tick_floor(0.505) == 0.5 and tick_floor(0.9124) == 0.912


# --------------------------------------------------------------- maker
def _trade(price, count, created="2026-11-03T20:00:10Z"):
    return {"yes_price_dollars": str(price), "count_fp": str(count), "created_time": created}


def test_maker_fills_only_when_tape_trades_strictly_through_price():
    since = 1793736000.0                                        # 2026-11-03T20:00:00Z
    tape = [_trade(0.50, 5), _trade(0.48, 3), _trade(0.47, 4)]
    assert crossing_volume(tape, "yes", 0.49, since) == 7        # only 0.48 and 0.47 are below 0.49
    assert crossing_volume(tape, "yes", 0.48, since) == 4        # touching 0.48 does not count
    assert crossing_volume([_trade(0.56, 6)], "no", 0.45, since) == 6   # NO bid .45 = YES offer .55
    assert crossing_volume([_trade(0.55, 6)], "no", 0.45, since) == 0


def test_maker_ignores_trades_before_order_reached_book():
    assert crossing_volume([_trade(0.40, 9, "2026-11-03T19:59:59Z")], "yes", 0.49, 1793736000.0) == 0


# ------------------------------------------------------------ book/report
def test_parse_book_converts_no_bids_to_yes_asks():
    ob = {"orderbook_fp": {"yes_dollars": [["0.4800", "10"], ["0.4900", "20"]],
                           "no_dollars": [["0.4800", "30"], ["0.4900", "5"]]}}
    asks, bids = parse_book(ob)
    assert bids[0] == (0.49, 20.0)                  # best bid first
    assert asks[0] == (0.51, 5.0)                   # NO bid 0.49 -> YES ask 0.51, nearest first


def test_confidence_interval():
    mean, lo, hi = ci([1.0] * 50)
    assert mean == 1.0 and lo == hi == 1.0
    assert ci([1.0])[1] == float("-inf")


# ----------------------------------------------------------- engine
def test_one_fill_per_window_even_when_maker_fills_later():
    """Regression: a maker fill lands in a later loop iteration, and must still count."""
    from kalshi_bot.brokers import Fill
    from kalshi_bot.dashboard import State, TradeRow
    from kalshi_bot.engine import Engine
    from kalshi_bot.journal import Journal
    import tempfile
    from pathlib import Path

    eng = Engine(CFG, "paper", None, MODEL, None, None, None, FEES,
                 Journal(Path(tempfile.mkdtemp()), "test"), State(), ["maker"], 1000)
    eng.window_fills = {"maker": 0}
    row = TradeRow(0, "T", "maker", "yes", 5, .5, .5, 0, .05, 300, "why", "resting")
    eng._on_fill(Fill("T", "yes", 5, .5, 0.0, "id", "maker"), row, close_ts=0)
    assert eng.window_fills["maker"] == 1 >= CFG.max_trades_per_window


# ------------------------------------------------------------ live wiring
def test_request_signature_verifies_with_public_key(tmp_path):
    """The exact bytes Kalshi verifies: timestamp + METHOD + path (no query string)."""
    import base64
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from kalshi_bot.kalshi_client import KalshiClient

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "k.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    c = KalshiClient("https://demo-api.kalshi.co/trade-api/v2", "KEYID", str(pem))
    h = c._sign_headers("POST", "/trade-api/v2/portfolio/orders")
    assert h["KALSHI-ACCESS-KEY"] == "KEYID"
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
                            (h["KALSHI-ACCESS-TIMESTAMP"] + "POST/trade-api/v2/portfolio/orders").encode(),
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                            hashes.SHA256())      # raises InvalidSignature if wrong


def test_live_confirmation_prompt_builds(monkeypatch, tmp_path, capsys):
    """Regression: the live prompt referenced a config field that no longer existed."""
    import argparse
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from kalshi_bot import cli

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "k.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    monkeypatch.setenv("KALSHI_KEY_ID", "K")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(pem))
    monkeypatch.setenv("KALSHI_ENV", "demo")
    monkeypatch.setattr("builtins.input", lambda _: "no")
    try:
        cli.cmd_run(argparse.Namespace(live=True, yes=False, no_dashboard=True, standard=False, trades_per_window=0, exit="none", exit_margin=None, series=None))
    except SystemExit as e:
        assert str(e) == "Aborted."
    assert "demo (fake-money)" in capsys.readouterr().out


def test_ed25519_key_signs_directly_and_verifies(tmp_path):
    """Regression: an Ed25519 key crashed the RSA-only signer (sign() takes 1 argument)."""
    import base64
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from kalshi_bot.kalshi_client import KalshiClient

    key = ed25519.Ed25519PrivateKey.generate()
    pem = tmp_path / "ed.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    c = KalshiClient("https://demo-api.kalshi.co/trade-api/v2", "KEYID", str(pem))
    h = c._sign_headers("GET", "/trade-api/v2/portfolio/balance")
    sig = base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"])
    assert len(sig) == 64
    key.public_key().verify(sig, (h["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/v2/portfolio/balance").encode())


# ------------------------------------------------- series / shutdown
def test_reference_product_follows_series():
    assert Config(series="KXGOLD15M").ref_product == "PAXG-USD"
    assert Config(series="KXBTC15M").ref_product == "BTC-USD"
    assert Config(series="KXBTC15M", reference_product="ETH-USD").ref_product == "ETH-USD"
    import pytest
    with pytest.raises(SystemExit):
        Config(series="KXUNKNOWN").ref_product


def _run_child(body: str, signals: int, wait: float):
    """Run a child that installs the watcher, send it SIGINT ``signals`` times, time its exit."""
    import os, signal, subprocess, sys, textwrap, time
    code = "import time\nfrom kalshi_bot import shutdown\nshutdown.install(grace_s=1.5)\nprint('ready', flush=True)\n" + textwrap.dedent(body)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    t0 = time.time()
    for _ in range(signals):
        os.kill(p.pid, signal.SIGINT)
        time.sleep(0.2)
    try:
        rc = p.wait(timeout=wait)
    except Exception:
        p.kill()
        raise AssertionError("child did not exit")
    return rc, time.time() - t0, p.stdout.read()


def test_ctrl_c_stops_a_normal_main_thread_gracefully():
    rc, took, out = _run_child("try:\n    time.sleep(30)\nexcept KeyboardInterrupt:\n    print('graceful')\n", 1, 5)
    assert rc == 0 and "graceful" in out and took < 3


def test_ctrl_c_still_exits_when_main_thread_ignores_it():
    """The reported hang: main thread swallows/blocks, grace timer must force the exit."""
    stuck = "while True:\n    try:\n        time.sleep(0.1)\n    except KeyboardInterrupt:\n        pass\n"
    rc, took, _ = _run_child(stuck, 1, 8)
    assert rc == 130 and took < 5


def test_second_ctrl_c_exits_immediately():
    stuck = "while True:\n    try:\n        time.sleep(0.1)\n    except KeyboardInterrupt:\n        pass\n"
    rc, took, _ = _run_child(stuck, 2, 8)
    assert rc == 130 and took < 1.2


# ------------------------------------------------------ market selection
def _mkt(ticker, open_t, close_t, **kw):
    return {"ticker": ticker, "open_time": open_t, "close_time": close_t, **kw}


def test_picks_current_window_not_markets_listed_a_day_early():
    """Regression: the demo server lists future windows ~22h early, all with open_time in the past
    and no floor_strike. Only the one closing within 15 minutes is the live window."""
    from datetime import datetime, timezone
    from kalshi_bot.engine import pick_active_market

    now = datetime(2026, 11, 4, 20, 0, 40, tzinfo=timezone.utc).timestamp()
    markets = [
        _mkt("A-2045", "2026-11-03T22:34:00Z", "2026-11-04T20:45:00Z"),    # listed early, closes in 45m
        _mkt("A-2030", "2026-11-03T22:34:00Z", "2026-11-04T20:30:00Z"),    # closes in 30m
        _mkt("A-2015", "2026-11-03T22:34:00Z", "2026-11-04T20:15:00Z"),    # closes in 15m -> live window
        _mkt("A-2000", "2026-11-03T22:34:00Z", "2026-11-04T20:00:00Z"),    # already closed
    ]
    assert pick_active_market(markets, now)["ticker"] == "A-2015"
    assert pick_active_market(markets[:2], now) is None                     # nothing live yet: wait


def test_tolerates_markets_without_strike():
    from kalshi_bot.engine import _num
    assert _num({}, "floor_strike") == 0 and _num({"floor_strike": None}, "floor_strike") == 0
    assert _num({"floor_strike": "4058.03"}, "floor_strike") == 4058.03


# ------------------------------------------------------- standard profile
def test_deviations_flag_only_trading_rule_changes():
    assert Config().deviations() == {}
    assert Config(env="prod", execution="maker", paper_bankroll=5000).deviations() == {}   # not rules
    assert Config(min_net_edge=0.01).deviations() == {"min_net_edge": 0.01}
    assert "series" in Config(series="KXBTC15M").deviations()


def test_standard_flag_ignores_env_overrides(monkeypatch):
    monkeypatch.setenv("BOT_MIN_NET_EDGE", "0.0")
    monkeypatch.setenv("KALSHI_SERIES", "KXBTC15M")
    monkeypatch.setenv("BOT_PAPER_BANKROLL", "250")
    assert Config.from_env().deviations() != {}                       # normal run: overrides apply
    std = Config.from_env(standard=True)
    assert std.deviations() == {} and std.paper_bankroll == 250       # rules reset, bankroll kept


def test_report_counts_only_standard_runs(tmp_path):
    import json
    from kalshi_bot.report import settled_trades, standard_files

    def journal(name, start, pnl):
        rows = ([start] if start else []) + [
            {"t": 1.0, "event": "settled", "ticker": "T" + name, "style": "taker", "side": "yes",
             "price": .5, "count": 1, "pnl": pnl, "fee": 0, "quote": .5, "result": "yes"}]
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows))

    journal("paper-a.jsonl", {"event": "start", "t": 0, "standard": True, "deviations": {}}, 1.0)
    journal("paper-b.jsonl", {"event": "start", "t": 0, "standard": False, "deviations": {"min_net_edge": 0.0}}, -9.0)
    journal("paper-c.jsonl", None, -5.0)                               # older run, no settings recorded
    pattern = str(tmp_path / "*.jsonl")
    ok, excluded = standard_files(pattern)
    assert ok == {"paper-a.jsonl"} and set(excluded) == {"paper-b.jsonl", "paper-c.jsonl"}
    rows, _ = settled_trades(pattern, resolve=False)
    assert [r["pnl"] for r in rows] == [1.0]
    assert len(settled_trades(pattern, resolve=False, runs="all")[0]) == 3


# ------------------------------------------------------ threading / latency
class _FakeClient:
    base_url = "http://fake"
    def get_market(self, ticker):
        return {"ticker": ticker, "yes_ask_dollars": "0.51", "yes_bid_dollars": "0.49",
                "yes_ask_size_fp": "500", "yes_bid_size_fp": "500", "close_time": "2026-11-04T20:00:00Z"}


def _engine(tmp_path, latency=2.0):
    from kalshi_bot.dashboard import State
    from kalshi_bot.engine import Engine
    from kalshi_bot.journal import Journal
    cfg = Config(paper_latency_s=latency, min_net_edge=0.03)
    eng = Engine(cfg, "paper", _FakeClient(), MODEL, None, PaperBroker(FEES), None, FEES,
                 Journal(tmp_path, "t"), State(), ["taker"], 1000)
    eng.window_fills = {"taker": 0}
    return eng, cfg


def test_paper_taker_delay_never_blocks_the_loop(tmp_path):
    """Regression: the simulated 2s network delay used to be a time.sleep in the trading loop."""
    import time
    eng, cfg = _engine(tmp_path, latency=2.0)
    d = decide(snap(ref_move=2.0), MODEL, FEES, cfg)
    assert d.side == "yes"
    t0 = time.time()
    eng._execute("taker", _FakeClient().get_market("T"), d, snap(ref_move=2.0), close_ts=time.time() + 300)
    assert time.time() - t0 < 0.2                                  # returned immediately
    assert "taker" in eng.inflight and eng.window_fills["taker"] == 0
    eng._service_inflight()                                          # delay not elapsed yet: nothing happens
    assert "taker" in eng.inflight
    eng.inflight["taker"].due = 0                                    # now elapsed
    eng._service_inflight()
    assert not eng.inflight and eng.window_fills["taker"] == 1
    assert eng.state.trades[0].status == "filled"


def test_dashboard_keeps_last_frame_if_a_render_fails():
    from kalshi_bot.dashboard import State, safe_renderable
    calls = []
    def flaky(state):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("deque mutated during iteration")
        return f"frame{len(calls)}"
    f = safe_renderable(State(), flaky)
    assert f() == "frame1" and f() == "frame1" and f() == "frame3"   # 2nd failed -> reuse frame1


# ------------------------------------------- what the dashboard says vs. what happens
def _sig(tmp_path):
    eng, cfg = _engine(tmp_path)
    d = decide(snap(ref_move=2.0), MODEL, FEES, cfg)
    assert d.side == "yes"
    return eng, cfg, d


def test_signal_explains_why_no_order_is_placed(tmp_path):
    """The 'why' panel used to say TRADE for the whole window after the bot had already traded."""
    eng, cfg, d = _sig(tmp_path)
    assert eng._why_not("taker", d, "", attempts=0) == ""                          # free to trade
    assert "cooldown" in eng._why_not("taker", d, "cooldown after 3 losses", 0)
    eng.window_fills["taker"] = cfg.max_trades_per_window
    assert "already traded this window (1/1)" in eng._why_not("taker", d, "", 0)
    eng.window_fills["taker"] = 0
    assert "attempt limit" in eng._why_not("taker", d, "", cfg.max_attempts_per_window)
    eng.inflight["taker"] = object()
    assert "waiting" in eng._why_not("taker", d, "", 0)
    no_signal = decide(snap(), MODEL, FEES, cfg)
    assert eng._why_not("taker", no_signal, "", 0) == ""                            # nothing to explain


def test_sizing_skip_is_reported_not_silent(tmp_path):
    eng, cfg, d = _sig(tmp_path)
    eng.risk["taker"].bankroll = 5                                                  # 0.5% of $5 buys < 1 contract
    msg = eng._execute("taker", _FakeClient().get_market("T"), d, snap(ref_move=2.0), close_ts=9e9)
    assert msg.startswith("skipped: bankroll")


def test_stats_track_open_positions_fees_at_fill_and_realized_pnl(tmp_path):
    from kalshi_bot.brokers import Fill
    from kalshi_bot.dashboard import TradeRow
    eng, cfg = _engine(tmp_path)
    row = TradeRow(0, "T", "taker", "yes", 10, .5, .5, 0, .05, 300, "why", "sent")
    fill = Fill("T", "yes", 10, 0.50, 0.18, "id", "taker")
    eng._on_fill(fill, row, close_ts=9e9)
    st = eng.state.stats["taker"]
    assert (st.trades, st.open, st.settled) == (1, 1, 0)
    assert abs(st.fees - 0.18) < 1e-9 and abs(st.at_risk - 5.18) < 1e-9 and st.pnl == 0   # fee shows now, P&L later
    assert eng.state.open_positions == [("taker", 9e9, fill.cost)]


def test_dashboard_renders_signal_action_open_positions_and_sparkline_range():
    import time
    from rich.console import Console
    from kalshi_bot.dashboard import State, render
    s = State(ticker="T", tau_min=5, yes_bid=.5, yes_ask=.51)
    s.decisions["taker"] = decide(snap(ref_move=2.0), MODEL, FEES, CFG)
    s.actions["taker"] = "already traded this window (1/1), watching only"
    st = s.stat("taker"); st.trades = st.open = 1; st.at_risk = 5.18
    s.open_positions = [("taker", time.time() + 120, 5.18)]
    for i in range(10):
        s.history.append((.5 + i * .01, i * .3))
    out = Console(width=130, height=46, record=True, file=open("/dev/null", "w"))
    out.print(render(s))
    text = out.export_text()
    for needle in ("SIGNAL", "already traded this window", "1 open", "$5.18 at risk", "P&L = settled",
                   "0.50-0.59", "relative low-to-high"):
        assert needle in text, needle


# ------------------------------------------------------ multi-trade (data-gathering) runs
def test_run_profiles():
    assert Config().profile() == "standard"
    assert Config(max_trades_per_window=3, max_attempts_per_window=5).profile() == "data"
    assert Config(max_trades_per_window=3, min_net_edge=0.0).profile() == "custom"


def test_trades_per_window_is_paper_only(capsys):
    import argparse
    import pytest
    from kalshi_bot import cli
    with pytest.raises(SystemExit) as e:
        cli.cmd_run(argparse.Namespace(live=True, yes=True, no_dashboard=True, standard=True, trades_per_window=3, exit="none", exit_margin=None, series=None))
    assert "paper data-gathering only" in str(e.value)
    with pytest.raises(SystemExit) as e:
        cli.cmd_run(argparse.Namespace(live=False, yes=True, no_dashboard=True, standard=True, trades_per_window=50, exit="none", exit_margin=None, series=None))
    assert "between 1 and 10" in str(e.value)


def test_clustered_ci_does_not_count_trades_in_one_window_as_independent():
    from kalshi_bot.report import ci
    # 3 trades per window that all share the window's outcome: +1,+1,+1 or -1,-1,-1
    vals = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
    naive = ci(vals)
    clustered = ci(vals, groups=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert naive[0] == clustered[0]
    assert clustered[2] - clustered[1] > 1.5 * (naive[2] - naive[1])      # honest interval is much wider


def test_identical_fills_in_one_window_are_each_matched_by_order_id(tmp_path):
    import json
    from kalshi_bot.report import settled_trades

    def ev(kind, oid, **kw):
        return {"t": 1.0, "event": kind, "ticker": "W", "style": "maker", "side": "yes", "price": .5,
                "count": 5, "fee": 0, "quote": .5, "order_id": oid, **kw}

    rows = [{"event": "start", "t": 0, "standard": False, "profile": "data"},
            ev("fill", "a", nth_in_window=1), ev("fill", "b", nth_in_window=2),
            ev("settled", "a", result="yes", pnl=2.5), ev("settled", "b", result="yes", pnl=2.5)]
    (tmp_path / "paper-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    settled, unresolved = settled_trades(str(tmp_path / "*.jsonl"), resolve=False, runs="data")
    assert len(settled) == 2 and unresolved == 0                            # neither dropped nor re-resolved
    assert sorted(r["nth_in_window"] for r in settled) == [1, 2]


def test_report_runs_selects_by_profile(tmp_path):
    import json
    from kalshi_bot.report import standard_files

    def journal(name, **start):
        (tmp_path / name).write_text(json.dumps({"event": "start", "t": 0, **start}))
    journal("paper-std.jsonl", standard=True, profile="standard")
    journal("paper-data.jsonl", standard=False, profile="data")
    journal("paper-custom.jsonl", standard=False, profile="custom")
    pat = str(tmp_path / "*.jsonl")
    assert standard_files(pat, "standard")[0] == {"paper-std.jsonl"}
    assert standard_files(pat, "data")[0] == {"paper-data.jsonl"}
    assert len(standard_files(pat, "all")[0]) == 3


def test_multi_trade_windows_risk_the_same_total_as_single_trade_windows():
    """Regression: 3 trades per window each risked a full 0.5%, so one losing window cost 3x."""
    one = RiskManager(Config(contracts=100, max_trades_per_window=1), bankroll=1000)
    three = RiskManager(Config(contracts=100, max_trades_per_window=3), bankroll=1000)
    assert one.size(0.50) == 10                       # $5 budget / $0.50
    assert three.size(0.50) * 3 <= one.size(0.50)     # 3 trades together never exceed the single-trade budget
    assert three.size(0.50) == 3


def test_cooldown_covers_three_windows_whatever_the_trades_per_window():
    """3 losing windows = 3 trades normally, 9 trades at 3 per window, with no manual setting."""
    single = RiskManager(Config(max_trades_per_window=1), bankroll=1000)
    triple = RiskManager(Config(max_trades_per_window=3), bankroll=1000)
    assert (single.loss_limit, triple.loss_limit) == (3, 9)
    for i in range(8):
        triple.record(-1.0, now=100)
    assert triple.block_reason(now=200) == ""                    # 8 straight losses: not yet
    triple.record(-1.0, now=100)
    assert "cooldown after 9 straight losses" in triple.block_reason(now=200)
    for _ in range(2):
        single.record(-1.0, now=100)
    assert single.block_reason(now=200) == ""
    single.record(-1.0, now=100)
    assert "cooldown after 3 straight losses" in single.block_reason(now=200)


# ------------------------------------------------------------- early-exit experiment
def test_exit_decision_sells_only_when_cash_beats_hold_value_plus_margin():
    from kalshi_bot.strategy import exit_decision
    # YES position; bid 0.70 pays 0.70 - 0.015 fee = 0.685 net; model says the position is worth 0.60
    d = exit_decision("yes", 10, yes_bid=0.70, yes_ask=0.71, p_yes=0.60, fees=FEES, margin=0.02)
    assert d.exit and abs(d.net_sale - 0.685) < 1e-9 and abs(d.gain - 0.085) < 1e-9
    # model agrees with the market: selling only pays the fee, so hold
    assert not exit_decision("yes", 10, 0.70, 0.71, 0.70, FEES, 0.02).exit
    # gain smaller than the margin: hold
    assert not exit_decision("yes", 10, 0.70, 0.71, 0.67, FEES, 0.02).exit      # gain 1.5c < 2c margin
    # NO position sells into the NO bid = 1 - YES ask, and holds the model's NO probability
    d = exit_decision("no", 10, yes_bid=0.29, yes_ask=0.30, p_yes=0.40, fees=FEES, margin=0.0)
    assert abs(d.sell_price - 0.70) < 1e-9 and abs(d.hold_value - 0.60) < 1e-9 and d.exit
    # no bid to sell into: never exit
    assert not exit_decision("yes", 10, 0.0, 0.05, 0.01, FEES, 0.0).exit


def test_exit_profile_and_settings():
    assert Config(exit_rule="model").profile() == "exit"
    assert Config(exit_rule="model", exit_margin=0.03).profile() == "exit"
    assert Config(exit_rule="model", min_net_edge=0.0).profile() == "custom"
    assert Config().exit_rule == "none"


def test_exit_flow_books_pnl_with_both_fees_and_removes_the_position(tmp_path):
    """Buy 10 YES @0.50 (fee 0.18); the market moves up and the model says sell: P&L must include both fees."""
    import json, time
    from kalshi_bot.brokers import Fill
    from kalshi_bot.dashboard import State, TradeRow
    from kalshi_bot.engine import Engine
    from kalshi_bot.journal import Journal

    class Client:
        def get_market(self, ticker):
            return {"ticker": ticker, "status": "active", "yes_bid_dollars": "0.70", "yes_ask_dollars": "0.71",
                    "yes_bid_size_fp": "500", "yes_ask_size_fp": "500"}

    cfg = Config(paper_latency_s=2.0, exit_rule="model", exit_margin=0.02)
    j = Journal(tmp_path, "t")
    eng = Engine(cfg, "paper", Client(), MODEL, None, None, None, FEES, j, State(), ["taker"], 1000)
    eng.window_fills = {"taker": 0}
    row = TradeRow(0, "T", "taker", "yes", 10, .5, .5, 0, .05, 600, "why", "sent")
    fill = Fill("T", "yes", 10, 0.50, 0.18, "oid-1", "taker")
    eng._on_fill(fill, row, close_ts=time.time() + 300)

    # model thinks YES is worth only 0.60 while the bid pays 0.70 -> sell
    eng._consider_exits("T", snap(yes_bid=0.70, yes_ask=0.71, tau_min=5), p_yes=0.60)
    assert "oid-1" in eng.exit_inflight and row.status == "exiting"
    assert "SELL" in eng.state.exit_line
    eng._service_exits()                                   # delay not elapsed: nothing happens yet
    assert eng.pending
    eng.exit_inflight["oid-1"].due = 0
    eng._service_exits()
    # proceeds 10*0.70 = 7.00, sale fee 0.15, cost 5.00+0.18 -> +1.67
    assert not eng.pending and not eng.exit_inflight
    assert abs(row.pnl - 1.67) < 1e-9 and row.result == "exit" and row.status == "settled"
    st = eng.state.stats["taker"]
    assert (st.settled, st.open, st.wins) == (1, 0, 1) and abs(st.fees - 0.33) < 1e-9 and abs(st.at_risk) < 1e-9
    events = [json.loads(l) for l in open(j.path)]
    settled = [e for e in events if e["event"] == "settled"][0]
    assert settled["result"] == "exit" and settled["exited"] and abs(settled["fee"] - 0.33) < 1e-9


def test_exit_is_skipped_when_the_bid_drops_before_the_simulated_sale(tmp_path):
    import time
    from kalshi_bot.brokers import Fill
    from kalshi_bot.dashboard import State, TradeRow
    from kalshi_bot.engine import Engine
    from kalshi_bot.journal import Journal

    class Client:                                          # the bid collapsed from 0.70 to 0.60 during the delay
        def get_market(self, ticker):
            return {"ticker": ticker, "status": "active", "yes_bid_dollars": "0.60", "yes_ask_dollars": "0.61",
                    "yes_bid_size_fp": "500", "yes_ask_size_fp": "500"}

    cfg = Config(exit_rule="model", exit_margin=0.02)
    eng = Engine(cfg, "paper", Client(), MODEL, None, None, None, FEES, Journal(tmp_path, "t"), State(), ["taker"], 1000)
    eng.window_fills = {"taker": 0}
    row = TradeRow(0, "T", "taker", "yes", 10, .5, .5, 0, .05, 600, "why", "sent")
    eng._on_fill(Fill("T", "yes", 10, 0.50, 0.18, "oid-2", "taker"), row, close_ts=time.time() + 300)
    eng._consider_exits("T", snap(yes_bid=0.70, yes_ask=0.71, tau_min=5), p_yes=0.60)
    eng.exit_inflight["oid-2"].due = 0
    eng._service_exits()
    assert len(eng.pending) == 1 and row.status == "filled"           # still holding, nothing booked
    assert eng.state.stats["taker"].settled == 0


def test_exit_is_off_by_default_and_never_runs_live(tmp_path):
    eng, cfg = _engine(tmp_path)
    eng._consider_exits("T", snap(), p_yes=0.5)
    assert eng.state.exit_line == "" and not eng.exit_inflight


def test_compare_pairs_exit_run_against_hold_run_by_window(tmp_path, capsys):
    import json
    from kalshi_bot.report import compare

    def trade(oid, ticker, pnl, result="yes", nth=1):
        return {"t": 1.0, "event": "settled", "ticker": ticker, "style": "taker", "side": "yes", "price": .5,
                "count": 10, "fee": .2, "quote": .5, "order_id": oid, "result": result, "pnl": pnl, "nth_in_window": nth}
    hold = [{"event": "start", "t": 0, "standard": True, "profile": "standard"}] + \
           [trade(f"h{i}", f"W{i}", -5.0, "no") for i in range(6)]            # every window lost the full stake
    exit_ = [{"event": "start", "t": 0, "standard": False, "profile": "exit"}] + \
            [trade(f"e{i}", f"W{i}", -2.0, "exit") for i in range(6)]         # sold early, lost less
    (tmp_path / "paper-hold.jsonl").write_text("\n".join(map(json.dumps, hold)))
    (tmp_path / "paper-exit.jsonl").write_text("\n".join(map(json.dumps, exit_)))
    compare(str(tmp_path / "*.jsonl"))
    out = capsys.readouterr().out
    assert "matched pairs (same window, style, side): 6" in out and "6 sold early" in out
    assert "paired gain +0.3000" in out                                       # +3.0 per trade / 10 contracts


def test_polling_interval_is_not_a_trading_rule_and_feed_reports_trade_age():
    assert Config(poll_s=0.25).deviations() == {} and Config(poll_s=0.25).profile() == "standard"
    import time
    from kalshi_bot.reference import CoinbaseFeed
    f = CoinbaseFeed()
    assert f.trade_age() == float("inf")                     # nothing seen yet
    f._last_trade_ts = time.time() - 42
    assert 41 < f.trade_age() < 44


def test_book_reader_survives_a_truncated_recorder_file(tmp_path):
    import gzip
    from kalshi_bot.backtest.record_book import read_book
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        fh.write('{"t":1}\n{"t":2}\n')
    good = p.read_bytes()
    p.write_bytes(good[:-8])                               # chop the gzip trailer: a killed recorder
    assert [r["t"] for r in read_book(p)] in ([1, 2], [1])  # never raises; keeps what decoded
