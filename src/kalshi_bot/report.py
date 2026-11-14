"""Results from journal files. This is the evidence for 'is it actually profitable?'.

``report`` applies proof gates: profit after fees must be statistically above zero, must
survive pessimistic cost scenarios, and must not rest on one lucky day.
``trades`` lists every trade: when, why, at what price, and how it ended."""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

from kalshi_bot.kalshi_client import PROD_URL, KalshiClient, KalshiError

# Extra cost per contract subtracted from every settled trade, to stress the result.
SCENARIOS = (("normal (as recorded)", 0.0), ("conservative (+1c/contract)", 0.01),
             ("stress (+2c/contract)", 0.02))


def load(pattern: str, event: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                e = json.loads(line)
                if e["event"] == event:
                    e["_file"] = os.path.basename(path)
                    rows.append(e)
    return rows


def _legacy_key(e: dict) -> tuple:
    return (e["ticker"], e["style"], e["side"], e["price"])


def _same_trade_index(settled: list[dict]) -> tuple[set, set]:
    """What counts as 'already settled'. Newer events carry the order id; older ones don't, so
    those fall back to (window, style, side, price)."""
    return ({e["order_id"] for e in settled if e.get("order_id")},
            {_legacy_key(e) for e in settled if not e.get("order_id")})


def _find(settled_rows: list[dict], fill: dict):
    for e in settled_rows:
        if e.get("order_id") and e["order_id"] == fill.get("order_id"):
            return e
    return next((e for e in settled_rows if not e.get("order_id") and _legacy_key(e) == _legacy_key(fill)), None)


def standard_files(pattern: str, runs: str = "standard", series: str | None = "KXGOLD15M") -> tuple[set[str], dict[str, str]]:
    """(journals to count, {other journal: why it is excluded}).
    runs: "standard" = default-settings runs, "data" = multi-trade data-gathering runs, "all"."""
    ok, excluded = set(), {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        start = next((e for e in map(json.loads, open(path)) if e["event"] == "start"), None)
        if series and start is not None and start.get("series", "KXGOLD15M") != series:
            excluded[name] = f"different series ({start.get('series')})"
        elif series and start is None and series != "KXGOLD15M":
            excluded[name] = "no settings recorded (older run)"
        elif runs == "all":
            ok.add(name)
        elif start is None:
            excluded[name] = "no settings recorded (older run)"
        elif (runs in ("data", "exit") and start.get("profile") == runs) or (runs == "standard" and start.get("standard")):
            ok.add(name)
        else:
            excluded[name] = f"{start.get('profile', 'custom')} run, settings {start.get('deviations')}"
    return ok, excluded


def settled_trades(pattern: str, resolve: bool = True, runs: str = "standard",
                   series: str | None = "KXGOLD15M") -> tuple[list[dict], int]:
    """Settled trades from the journals, plus trades that settled AFTER the bot was stopped.

    A bot stopped mid-window never records the result of its open positions. Paper fills
    are on production markets, so their final results are public: look them up and settle
    them here. Returns (settled rows, number still unresolved)."""
    keep = standard_files(pattern, runs, series)[0]
    allowed = lambda e: e["_file"] in keep
    settled = [e for e in load(pattern, "settled") if allowed(e)]
    fills = [f for f in load(pattern, "fill") if allowed(f)]
    nth = {f["order_id"]: f.get("nth_in_window", 1) for f in fills if f.get("order_id")}
    for e in settled:                                   # which trade of its window was this?
        e.setdefault("nth_in_window", nth.get(e.get("order_id"), 1))
    have_ids, have_legacy = _same_trade_index(settled)
    pending = [f for f in fills if f["_file"].startswith("paper-")
               and f.get("order_id") not in have_ids and _legacy_key(f) not in have_legacy]
    if not pending:
        return settled, 0
    results: dict[str, str] = {}
    if resolve:
        try:
            client = KalshiClient(PROD_URL)
            for ticker in {f["ticker"] for f in pending}:
                results[ticker] = client.get_market(ticker).get("result", "")
        except KalshiError as exc:
            print(f"(could not look up results for unsettled trades: {exc})", file=sys.stderr)
    unresolved = 0
    for f in pending:
        result = results.get(f["ticker"], "")
        if result not in ("yes", "no"):
            unresolved += 1
            continue
        payout = f["count"] if result == f["side"] else 0.0
        settled.append({**f, "event": "settled", "result": result, "backfilled": True,
                        "nth_in_window": f.get("nth_in_window", 1),
                        "pnl": payout - f["price"] * f["count"] - f["fee"]})
    return settled, unresolved


def ci(values: list[float], groups: list | None = None) -> tuple[float, float, float]:
    """(mean, lower, upper) of a 95% confidence interval for the average.

    With ``groups`` (the window each trade belongs to) the interval is clustered: trades in one
    window share a single outcome, so they count as one piece of evidence, not several."""
    n = len(values)
    mean = sum(values) / n
    if groups is None:
        if n < 2:
            return mean, float("-inf"), float("inf")
        se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) / math.sqrt(n)
    else:
        sums, counts = defaultdict(float), defaultdict(int)
        for v, g in zip(values, groups):
            sums[g] += v
            counts[g] += 1
        g_n = len(sums)
        if g_n < 2:
            return mean, float("-inf"), float("inf")
        resid = sum((sums[g] - mean * counts[g]) ** 2 for g in sums)
        se = math.sqrt(resid * g_n / (g_n - 1)) / n
    return mean, mean - 1.96 * se, mean + 1.96 * se


def style_report(style: str, rows: list[dict]) -> bool:
    """Print one style's results; return True if it passes every gate."""
    n = len(rows)
    per_contract = [r["pnl"] / r["count"] for r in rows]
    groups = [r["ticker"] for r in rows]
    windows = len(set(groups))
    print(f"\n=== {style.upper()}  ({n} settled trades in {windows} windows) ===")
    if n > windows:
        print("(several trades share a window, so confidence intervals count each WINDOW once)")
    print(f"net P&L after fees   ${sum(r['pnl'] for r in rows):+.2f}   fees paid ${sum(r['fee'] for r in rows):.2f}   "
          f"win rate {sum(r['pnl'] > 0 for r in rows) / n:.1%}")
    slip = [r["price"] - r["quote"] for r in rows]
    print(f"avg price vs quote   {sum(slip) / n:+.4f} per contract (negative = better than quoted)")

    gates = []
    print("cost scenarios (avg $/contract, 95% CI):")
    for label, extra in SCENARIOS:
        mean, lo, hi = ci([x - extra for x in per_contract], groups)
        gates.append(lo > 0)
        print(f"  {label:<30} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {'PASS' if lo > 0 else 'fail'}")

    by_day = defaultdict(float)
    for r in rows:
        by_day[time.strftime("%Y-%m-%d", time.localtime(r["t"]))] += r["pnl"]
    green = sum(v > 0 for v in by_day.values())
    gates.append(len(by_day) >= 5 and green / len(by_day) >= 0.6)
    print(f"stability: {green}/{len(by_day)} days profitable "
          f"{'(need >=5 days and >=60% green)' if len(by_day) < 5 or green / len(by_day) < 0.6 else ''}")

    mean, lo, hi = ci(per_contract, groups)
    if windows > 1 and mean > 0:
        se = (hi - mean) / 1.96
        print(f"windows needed for the normal-case CI to exclude zero at this effect: "
              f"~{math.ceil(windows * (1.96 * se / mean) ** 2)} (have {windows})")
    later = [r for r in rows if r.get("nth_in_window", 1) > 1]
    if later:
        first = [r for r in rows if r.get("nth_in_window", 1) <= 1]
        avg = lambda rs: sum(r["pnl"] / r["count"] for r in rs) / len(rs) if rs else float("nan")
        print(f"first trade in a window: {len(first)} trades, avg {avg(first):+.4f}/contract | "
              f"later trades: {len(later)}, avg {avg(later):+.4f}/contract")
    ok = all(gates)
    print("VERDICT: " + ("passes all gates - eligible for small live size" if ok else
          "NOT proven. " + ("Not distinguishable from luck yet." if lo <= 0 else "Positive, but fails a stress or stability gate.")))
    return ok


def report(pattern: str, runs: str = "standard", series: str = "KXGOLD15M") -> None:
    print(f"series: {series}")
    ok, excluded = standard_files(pattern, runs, series)
    label = {"standard": "standard-settings", "data": "data-gathering (multi-trade)",
             "exit": "early-exit experiment", "all": "ALL"}[runs]
    if runs == "all":
        print("INCLUDING every run (--runs all): not valid as evidence.")
    else:
        print(f"Counting {len(ok)} {label} run(s); excluding {len(excluded)} other(s)"
              f"{' (see --runs data / --runs all)' if excluded else ''}.")
    rows, unresolved = settled_trades(pattern, runs=runs, series=series)
    if not rows:
        print(f"No settled trades found in {label} runs matching {pattern}.")
        return
    backfilled = sum(1 for r in rows if r.get("backfilled"))
    if backfilled or unresolved:
        print(f"({backfilled} trades were settled after the bot stopped, using Kalshi's published results; "
              f"{unresolved} still waiting for their window to resolve)")
    by_style = defaultdict(list)
    for r in rows:
        by_style[r.get("style", "taker")].append(r)
    for style, rs in sorted(by_style.items()):
        style_report(style, rs)
    if len(by_style) == 2:
        print("\nMaker vs taker: compare the 'normal' CIs above. Maker's edge counts only if its CI "
              "clears zero AFTER the fills it missed and the adverse selection it suffered, both of "
              "which are already in these numbers.")


def trades(pattern: str, limit: int = 40, runs: str = "standard", series: str = "KXGOLD15M") -> None:
    keep = standard_files(pattern, runs, series)[0]
    fills = {(e["ticker"], e["style"], e["order_id"]): e for e in load(pattern, "fill") if e["_file"] in keep}
    settled_rows = settled_trades(pattern, runs=runs, series=series)[0]
    print(f"{'time':8} {'left':>5} {'window':9} {'style':6} {'side':4} {'qty':>3} {'quote':>6} {'fill':>6} "
          f"{'fee':>5} {'result':>14}  why")
    rows = sorted(fills.values(), key=lambda e: e["t"])[-limit:]
    for e in rows:
        s = _find(settled_rows, e)
        result = f"{s['result'].upper()} {s['pnl']:+.2f}$" if s else "pending"
        left = f"{e['secs_left']:.0f}s" if "secs_left" in e else ""
        print(f"{time.strftime('%H:%M:%S', time.localtime(e['t'])):8} {left:>5} {e['ticker'][-9:]:9} "
              f"{e['style']:6} {e['side'].upper():4} {e['count']:>3} {e['quote']:>6.3f} {e['price']:>6.3f} "
              f"{e['fee']:>5.2f} {result:>14}  {e.get('why', '')}")
    if not rows:
        print("(no fills yet)")


def compare(pattern: str) -> None:
    """Early-exit experiment vs hold-to-settlement, paired window by window.

    Both runs use the same entry rule, so in a window where both bought the same side of the same
    style, any P&L difference per contract comes from the exit rule (plus a few seconds of timing
    and different sizes, which per-contract numbers wash out). Confidence intervals count each
    window once. The baseline is the FIRST trade per window of the standard and data runs."""
    base = [r for runs in ("standard", "data") for r in settled_trades(pattern, runs=runs)[0]
            if r.get("nth_in_window", 1) <= 1]
    exp = [r for r in settled_trades(pattern, runs="exit")[0] if r.get("nth_in_window", 1) <= 1]
    if not exp:
        print("No exit-experiment trades yet. Start one with: kalshi-bot run --standard --exit model")
        return
    by_base = {}
    for r in base:
        by_base.setdefault((r["ticker"], r["style"], r["side"]), r)
    pairs = [(by_base[k], r) for r in exp if (k := (r["ticker"], r["style"], r["side"])) in by_base]
    print(f"exit-experiment trades: {len(exp)} ({sum(r.get('result') == 'exit' for r in exp)} sold early) | "
          f"hold baseline trades: {len(base)} | matched pairs (same window, style, side): {len(pairs)}")
    if len(pairs) < 2:
        print("Not enough matched windows yet: keep both runs going at the same time.")
        return
    per = lambda r: r["pnl"] / r["count"]
    for label, subset in (("ALL", pairs), ("taker", [p for p in pairs if p[1]["style"] == "taker"]),
                          ("maker", [p for p in pairs if p[1]["style"] == "maker"])):
        if len(subset) < 2:
            continue
        diffs = [per(e) - per(b) for b, e in subset]
        groups = [e["ticker"] for _, e in subset]
        mean, lo, hi = ci(diffs, groups)
        sold = sum(e.get("result") == "exit" for _, e in subset)
        verdict = "exit HELPS" if lo > 0 else "exit HURTS" if hi < 0 else "no detectable difference yet"
        print(f"{label:>6}: {len(subset):3d} pairs, {sold:3d} sold early | hold {sum(per(b) for b, _ in subset) / len(subset):+.4f} "
              f"vs exit {sum(per(e) for _, e in subset) / len(subset):+.4f} per contract | "
              f"paired gain {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "logs/*.jsonl")
