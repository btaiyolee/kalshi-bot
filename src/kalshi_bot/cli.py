"""Command-line interface.

    kalshi-bot run                 paper trade against live data (default, no credentials needed)
    kalshi-bot run --live          place REAL orders (demo or prod, per KALSHI_ENV)
    kalshi-bot collect             download backtest data
    kalshi-bot fit                 fit the model on collected data
    kalshi-bot backtest            walk-forward edge study (after fees)
    kalshi-bot auth-check          diagnose API key problems (tries demo and prod, checks clock)
    kalshi-bot chart               live net-P&L chart of the paper runs (opens a local web page)
    kalshi-bot compare             early-exit experiment vs hold-to-settlement, paired by window
    kalshi-bot trades [glob]       every trade: when, why, at what price, outcome
    kalshi-bot report [glob]       proof gates: P&L with confidence intervals and cost stress tests
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import logging
import sys
from pathlib import Path

from kalshi_bot.config import Config


def cmd_run(args: argparse.Namespace) -> None:
    from kalshi_bot import shutdown
    from kalshi_bot.brokers import LiveBroker, PaperBroker
    from kalshi_bot.maker import LiveMaker, PaperMaker
    from kalshi_bot.dashboard import State, live_view
    from kalshi_bot.engine import Engine
    from kalshi_bot.fees import FeeSchedule
    from kalshi_bot.journal import Journal
    from kalshi_bot.kalshi_client import KalshiClient
    from kalshi_bot.model import DEFAULT_MODEL_PATH, Model
    from kalshi_bot.reference import CoinbaseFeed

    cfg = Config.from_env(standard=args.standard)
    if args.series:
        cfg = replace(cfg, series=args.series)
    mode = "live" if args.live else "paper"
    if args.exit != "none":
        if mode == "live":
            sys.exit("--exit is a paper-only experiment for now (live needs sell-order handling).")
        cfg = replace(cfg, exit_rule=args.exit,
                      **({"exit_margin": args.exit_margin} if args.exit_margin is not None else {}))
    if args.trades_per_window:
        if mode == "live":
            sys.exit("--trades-per-window is for paper data-gathering only; live runs keep the 1-per-window rule.")
        if not 1 <= args.trades_per_window <= 10:
            sys.exit("--trades-per-window must be between 1 and 10.")
        cfg = replace(cfg, max_trades_per_window=args.trades_per_window,
                      max_attempts_per_window=args.trades_per_window + 2)
    # Paper mode watches REAL production markets (public data, no credentials, no orders).
    # Live mode talks to whichever environment KALSHI_ENV selects.
    from kalshi_bot.kalshi_client import PROD_URL
    client = KalshiClient(cfg.base_url if mode == "live" else PROD_URL,
                          cfg.key_id or None, cfg.private_key_path or None)

    if mode == "live":
        if not client.can_trade:
            sys.exit("Live mode needs KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH (see .env.example).")
        if not args.yes:
            money = "REAL-MONEY" if cfg.env == "prod" else "demo (fake-money)"
            print(f"About to place {money} {cfg.execution} orders on Kalshi '{cfg.env}': up to "
                  f"{cfg.contracts} contracts per trade, {cfg.risk_per_trade_pct:.1%} of bankroll "
                  f"per trade, daily loss stop {cfg.daily_loss_pct:.0%}.")
            if input("Type 'trade' to continue: ").strip() != "trade":
                sys.exit("Aborted.")
    from kalshi_bot.series import spec_for
    model_path = spec_for(cfg.series).model_path
    if not model_path.exists():
        sys.exit(f"No model at {model_path}. Run `kalshi-bot collect --series {cfg.series}` then "
                 f"`kalshi-bot fit --series {cfg.series}` first.")

    fees = FeeSchedule.from_series(client.get_series(cfg.series))
    if mode == "live":
        taker, maker = LiveBroker(client), LiveMaker(client, fees)
        styles, bankroll = [cfg.execution], 0.0
        try:
            bankroll = client.get_balance()["balance"] / 100
        except Exception as exc:
            sys.exit(f"Could not read your balance (check credentials / KALSHI_ENV): {exc}")
    else:                       # paper: run both styles side by side on identical signals
        taker, maker = PaperBroker(fees), PaperMaker(client, fees, cfg.paper_latency_s)
        styles, bankroll = ["taker", "maker"], cfg.paper_bankroll
    shutdown.install()                      # reliable Ctrl-C; must run before any thread starts
    feed = CoinbaseFeed(cfg.ref_product)
    feed.start()
    journal = Journal(mode=mode)
    where = cfg.env if mode == "live" else "live market data, simulated fills"
    state = State(mode=mode, env=f"{where} | {cfg.series} vs {cfg.ref_product}")
    state.balance = bankroll if mode == "live" else None
    dev = cfg.deviations()
    journal.write("start", mode=mode, series=cfg.series, styles=styles, standard=not dev,
                  profile=cfg.profile(),
                  deviations=dev, config={k: v for k, v in asdict(cfg).items()
                                          if k not in ("key_id", "private_key_path")})
    if cfg.profile() == "exit":
        state.log(f"EXIT EXPERIMENT: entries identical to standard; sells early when selling beats holding by "
                  f"{cfg.exit_margin:.2f}. Report: `kalshi-bot report --runs exit`, compare: `kalshi-bot compare`.")
    elif cfg.profile() == "data":
        state.log(f"DATA-GATHERING run: up to {cfg.max_trades_per_window} trades per window per style. "
                  f"Report it with `kalshi-bot report --runs data` (trades in one window share one outcome).")
    elif dev:
        state.log(f"NON-STANDARD settings {dev}: `kalshi-bot report` will EXCLUDE this run "
                  f"(use `run --standard` for a clean test)")
    else:
        state.log("STANDARD production settings: this run counts toward `kalshi-bot report`")
    state.log(f"fees: makers_pay={fees.makers_pay} multiplier={fees.multiplier}; styles={styles}; "
              f"journal {journal.path}")
    if cfg.series != "KXGOLD15M":
        state.log(f"NOTE: the model was fitted on gold. {cfg.series} is for plumbing tests, not profit.")
    if mode == "live" and cfg.env == "demo":
        state.log("demo books are thin: taker orders need size at the touch, so try BOT_CONTRACTS=2 or BOT_EXECUTION=maker")
    engine = Engine(cfg, mode, client, Model.load(model_path), feed, taker, maker, fees, journal, state,
                    styles, bankroll)
    try:
        with live_view(state, enabled=not args.no_dashboard):
            engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()
        feed.stop()
        journal.close()
    print("\nStopped.")
    for style, st in state.stats.items():
        print(f"  {style:<6} fills {st.trades}  settled {st.settled}  net P&L ${st.pnl:+.2f}")
    print(f"Journal: {journal.path}\n  kalshi-bot trades   (every trade: when / why / price / outcome)\n"
          "  kalshi-bot report   (profit with confidence intervals)")


def cmd_collect(args: argparse.Namespace) -> None:
    from kalshi_bot.backtest import collect_kalshi, collect_reference
    sys.argv = [sys.argv[0], "--series", args.series]
    collect_kalshi.main()
    collect_reference.main()


def cmd_fit(args: argparse.Namespace) -> None:
    from kalshi_bot.backtest.dataset import build
    from kalshi_bot.model import fit
    from kalshi_bot.series import spec_for
    spec = spec_for(args.series)
    model = fit(build(args.series))
    model.save(spec.model_path)
    print(f"Saved {spec.model_path}: {model}")


def cmd_backtest(args: argparse.Namespace) -> None:
    from kalshi_bot.backtest.edge_study import main
    main(args.series)


def cmd_ticks(args: argparse.Namespace) -> None:
    from kalshi_bot.backtest import collect_ticks
    collect_ticks.main(args.series, args.hours)


def cmd_record_book(args: argparse.Namespace) -> None:
    from kalshi_bot.backtest.record_book import record
    record(args.series or ["KXGOLD15M", "KXBTC15M"], args.poll_s)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from kalshi_bot.backtest.evaluate import evaluate
    evaluate(args.series, args.thr, args.perms, args.source)


def cmd_report(args: argparse.Namespace) -> None:
    from kalshi_bot import report
    report.report(args.pattern or "logs/*.jsonl", runs="all" if args.all else args.runs, series=args.series)


def cmd_auth_check(_: argparse.Namespace) -> None:
    """Try your key against BOTH Kalshi environments and say what is wrong."""
    import email.utils
    import time

    import requests
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from kalshi_bot.kalshi_client import DEMO_URL, PROD_URL, KalshiClient, KalshiError

    cfg = Config.from_env()
    if not cfg.key_id or not cfg.private_key_path:
        sys.exit("Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env first.")
    kid = cfg.key_id
    print(f"Key ID: {kid[:8]}...{kid[-4:]}  (length {len(kid)}; Kalshi IDs are normally 36 chars)")
    if kid != kid.strip() or any(c in kid for c in "\"' "):
        print("  WARNING: the Key ID contains spaces or quotes - remove them in .env")
    try:
        client = KalshiClient(PROD_URL, kid, cfg.private_key_path)
    except Exception as exc:
        sys.exit(f"Could not load the private key at {cfg.private_key_path}: {exc}")
    kind = "Ed25519" if isinstance(client._private_key, ed25519.Ed25519PrivateKey) else "RSA"
    print(f"Private key: loaded OK ({kind})")

    try:
        date = requests.head(PROD_URL, timeout=8).headers["Date"]
        skew = time.time() - email.utils.parsedate_to_datetime(date).timestamp()
        print(f"Clock skew vs Kalshi: {skew:+.1f}s" + ("   <-- too large, fix your system clock" if abs(skew) > 30 else "   (fine)"))
    except Exception as exc:
        print(f"Clock check skipped: {exc}")

    results = {}
    for name, url in (("demo", DEMO_URL), ("prod", PROD_URL)):
        try:
            bal = KalshiClient(url, kid, cfg.private_key_path).get_balance()
            results[name] = "ok"
            print(f"  {name}: AUTHENTICATED, balance ${bal['balance'] / 100:,.2f}")
        except KalshiError as exc:
            results[name] = str(exc)
            print(f"  {name}: rejected - {str(exc)[:160]}")

    print()
    if results["demo"] == "ok":
        print("Your key works on demo. Use KALSHI_ENV=demo." + ("" if results["prod"] != "ok" else " (It also works on prod.)"))
    elif results["prod"] == "ok":
        print("This is a PRODUCTION key. Demo needs its own key from demo.kalshi.co (separate account).\n"
              "To use this key anyway, set KALSHI_ENV=prod - that trades REAL money.")
    else:
        print("Neither environment accepts this key/ID pair. Most likely the private key file does not\n"
              "belong to this Key ID (re-download or regenerate the key and update BOTH values), or the\n"
              "Key ID was copied incorrectly.")


def cmd_chart(args: argparse.Namespace) -> None:
    from kalshi_bot import chart
    chart.serve(args.pattern, args.port, open_browser=not args.no_open, series=args.series)


def cmd_compare(args: argparse.Namespace) -> None:
    from kalshi_bot import report
    report.compare(args.pattern or "logs/*.jsonl")


def cmd_trades(args: argparse.Namespace) -> None:
    from kalshi_bot import report
    report.trades(args.pattern or "logs/*.jsonl", runs="all" if args.all else args.runs, series=args.series)


def main() -> None:
    ap = argparse.ArgumentParser(prog="kalshi-bot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="trade (paper by default)")
    run.add_argument("--live", action="store_true", help="place real orders")
    run.add_argument("--yes", action="store_true", help="skip the live-trading confirmation")
    run.add_argument("--no-dashboard", action="store_true")
    run.add_argument("--trades-per-window", type=int, default=0, metavar="N",
                     help="PAPER ONLY: allow up to N trades per window per style to gather execution data faster")
    run.add_argument("--series", default=None, help="KXGOLD15M (default) or KXBTC15M (paper-test a new series)")
    run.add_argument("--exit", choices=("none", "model"), default="none",
                     help="PAPER ONLY experiment: sell early when the model says selling beats holding")
    run.add_argument("--exit-margin", type=float, default=None, metavar="DOLLARS",
                     help="with --exit model: required gain per contract from selling (default 0.02)")
    run.add_argument("--standard", action="store_true",
                     help="ignore .env/shell overrides of the trading rules: production defaults only")
    run.set_defaults(fn=cmd_run)
    for name, fn, helptext in (("collect", cmd_collect, "download backtest data"),
                               ("fit", cmd_fit, "fit the model"),
                               ("backtest", cmd_backtest, "walk-forward study")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--series", default="KXGOLD15M", help="KXGOLD15M (default) or KXBTC15M")
        sp.set_defaults(fn=fn)
    tk = sub.add_parser("ticks", help="download/extend raw public trades of the reference product (honest features)")
    tk.add_argument("--series", default="KXGOLD15M")
    tk.add_argument("--hours", type=float, default=None, help="only fetch this many recent hours (use for BTC)")
    tk.set_defaults(fn=cmd_ticks)
    rb = sub.add_parser("record-book", help="record the live Kalshi order book + trades (builds the dataset structure strategies need)")
    rb.add_argument("--series", action="append", default=None, help="repeatable; default gold and BTC")
    rb.add_argument("--poll-s", type=float, default=1.0)
    rb.set_defaults(fn=cmd_record_book)
    ev = sub.add_parser("evaluate", help="honest full evaluation of a series (splits, frictions, placebo, gates)")
    ev.add_argument("--series", default="KXGOLD15M")
    ev.add_argument("--thr", type=float, default=0.04, help="required edge per contract after fees")
    ev.add_argument("--perms", type=int, default=200, help="placebo shuffles")
    ev.add_argument("--source", choices=("auto", "candles", "ticks"), default="auto",
                    help="reference-price features from raw public trades (auto = ticks when recorded) or 1-min candles")
    ev.set_defaults(fn=cmd_evaluate)
    rep = sub.add_parser("report", help="realised results")
    rep.add_argument("pattern", nargs="?")
    rep.add_argument("--runs", choices=("standard", "data", "exit", "all"), default="standard",
                     help="which runs to count: standard (default), data (multi-trade runs), or all")
    rep.add_argument("--all", action="store_true", help="same as --runs all")
    rep.add_argument("--series", default="KXGOLD15M")
    rep.set_defaults(fn=cmd_report)
    sub.add_parser("auth-check", help="diagnose API key problems").set_defaults(fn=cmd_auth_check)
    ch = sub.add_parser("chart", help="live net-P&L chart of the paper runs (local web page)")
    ch.add_argument("--port", type=int, default=8765)
    ch.add_argument("--no-open", action="store_true", help="don't open the browser automatically")
    ch.add_argument("--pattern", default="logs/paper-*.jsonl")
    ch.add_argument("--series", default="KXGOLD15M", help="which market's runs to plot")
    ch.set_defaults(fn=cmd_chart)
    cmp_ = sub.add_parser("compare", help="early-exit experiment vs hold, paired by window")
    cmp_.add_argument("pattern", nargs="?")
    cmp_.set_defaults(fn=cmd_compare)
    tr = sub.add_parser("trades", help="every trade: when, why, price, outcome")
    tr.add_argument("pattern", nargs="?")
    tr.add_argument("--runs", choices=("standard", "data", "exit", "all"), default="standard")
    tr.add_argument("--all", action="store_true", help="same as --runs all")
    tr.add_argument("--series", default="KXGOLD15M")
    tr.set_defaults(fn=cmd_trades)
    from kalshi_bot import lip
    lp = sub.add_parser("lip-paper", help="paper-test Kalshi Liquidity Incentive Program quoting (--report for results)")
    lip.add_arguments(lp)
    lp.set_defaults(fn=lip.cmd_lip_paper)
    from kalshi_bot import weather
    weather.add_cli(sub)
    args = ap.parse_args()
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.WARNING, filename="logs/bot.log" if args.cmd == "run" else None)
    args.fn(args)


if __name__ == "__main__":
    main()
