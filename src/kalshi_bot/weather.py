"""Kalshi weather: live-observation recorder + paper tester for daily-temperature brackets.

Idea under test: on the settlement day the running max (min) of station observations is a hard-ish
lower (upper) bound on the final daily high (low), so brackets below it are dead and the remaining
rise is a function of local hour. A 61-day x 20-city backtest (see README) found the market already
prices this (no edge net of fees); this module is the *forward* test and a data recorder.

Never places orders. Only unauthenticated public GETs (Kalshi market data, aviationweather.gov METAR,
api.weather.gov). Pure functions at the top are unit-tested in tests/test_weather.py.

Time convention: Kalshi settles the daily high on the NWS CLI day = LOCAL STANDARD TIME midnight to
midnight (no DST shift), so in summer the settlement day runs 01:00 -> 01:00 local clock time.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import requests

from .fees import FeeSchedule
from .weather_gapdata import GAP_COUNTS

log = logging.getLogger(__name__)

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
AWC = "https://aviationweather.gov/api/data/metar"
NWS = "https://api.weather.gov/stations/{}/observations"
UA = "kalshi-bot-weather-research/0.1 (public-data research; no trading)"

# series -> (ICAO station, IANA tz, low-series ticker or None). Station = the NWS CLI site named in the
# market rules (e.g. CLINYC = Central Park KNYC, CLICHI = Midway KMDW).
CITIES: dict[str, tuple[str, str, Optional[str]]] = {
    "KXHIGHNY": ("KNYC", "America/New_York", "KXLOWTNYC"),
    "KXHIGHCHI": ("KMDW", "America/Chicago", "KXLOWTCHI"),
    "KXHIGHMIA": ("KMIA", "America/New_York", "KXLOWTMIA"),
    "KXHIGHAUS": ("KAUS", "America/Chicago", "KXLOWTAUS"),
    "KXHIGHLAX": ("KLAX", "America/Los_Angeles", "KXLOWTLAX"),
    "KXHIGHDEN": ("KDEN", "America/Denver", "KXLOWTDEN"),
    "KXHIGHPHIL": ("KPHL", "America/New_York", "KXLOWTPHIL"),
    "KXHIGHTDAL": ("KDFW", "America/Chicago", "KXLOWTDAL"),
    "KXHIGHTHOU": ("KHOU", "America/Chicago", "KXLOWTHOU"),
    "KXHIGHTNOLA": ("KMSY", "America/Chicago", "KXLOWTNOLA"),
    "KXHIGHTOKC": ("KOKC", "America/Chicago", "KXLOWTOKC"),
    "KXHIGHTPHX": ("KPHX", "America/Phoenix", "KXLOWTPHX"),
    "KXHIGHTLV": ("KLAS", "America/Los_Angeles", "KXLOWTLV"),
    "KXHIGHTSEA": ("KSEA", "America/Los_Angeles", "KXLOWTSEA"),
    "KXHIGHTSFO": ("KSFO", "America/Los_Angeles", "KXLOWTSFO"),
    "KXHIGHTBOS": ("KBOS", "America/New_York", "KXLOWTBOS"),
    "KXHIGHTDC": ("KDCA", "America/New_York", "KXLOWTDC"),
    "KXHIGHTATL": ("KATL", "America/New_York", "KXLOWTATL"),
    "KXHIGHTMIN": ("KMSP", "America/Chicago", "KXLOWTMIN"),
    "KXHIGHTSATX": ("KSAT", "America/Chicago", "KXLOWTSATX"),
}
HOURLY_SERIES = "KXTEMPNYCHS"  # hourly directional NYC temp (Synoptic "Kalshi Weather Index"); logged only
HOURLY_EXTRA_STATIONS = ("KLGA", "KJFK", "KEWR")

# paper rule (from the backtest's V1 model): taker, edge threshold after fee+slippage, hour window
RULE = dict(edge=0.06, hmin=12, hmax=22, slip=0.01, max_obs_age_s=5400)


# ====================================================================== pure core
def std_offset_hours(tz: str) -> float:
    """Standard-time UTC offset (hours, negative west). Min of Jan/Jul offsets = standard time."""
    z = ZoneInfo(tz)
    offs = [z.utcoffset(dt.datetime(2026, m, 1)).total_seconds() / 3600 for m in (1, 7)]
    return min(offs)


def lst_day_bounds(ts: float, tz: str) -> tuple[float, float, dt.date]:
    """UTC bounds [start, end) and calendar date of the local-STANDARD-time day containing ts."""
    off = std_offset_hours(tz)
    local = dt.datetime.fromtimestamp(ts + off * 3600, dt.timezone.utc)
    d = local.date()
    start = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() - off * 3600
    return start, start + 86400, d


def lst_hour(ts: float, tz: str) -> int:
    off = std_offset_hours(tz)
    return dt.datetime.fromtimestamp(ts + off * 3600, dt.timezone.utc).hour


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def running_extremes(obs: Iterable[tuple[float, float]], ts: float, tz: str) -> Optional[dict]:
    """Running max/min (deg F) of observations inside the LST day containing ts, up to ts.
    obs = (unix_ts, temp_F). Returns None when no observation yet today."""
    start, end, d = lst_day_bounds(ts, tz)
    xs = [(t, f) for t, f in obs if start <= t <= min(ts, end - 1e-6)]
    if not xs:
        return None
    last_t, last_f = max(xs)
    ts_sorted = sorted(t for t, _ in xs)
    edges = [start] + ts_sorted + [ts]
    max_gap = max(b - a for a, b in zip(edges, edges[1:]))
    return dict(first_ts=ts_sorted[0], max_gap_s=max_gap, rmax=max(f for _, f in xs), rmin=min(f for _, f in xs), n=len(xs), last_ts=last_t,
                last_f=last_f, age_s=ts - last_t, date=d.isoformat())


def coverage_ok(ex: Optional[dict], day_start: float, max_gap_s: float = 7500, first_within_s: float = 4 * 3600) -> bool:
    """Running max/min is only trustworthy if obs cover the day from (near) its start without big holes
    (e.g. a freshly started recorder, or KNYC's sparse feed, would otherwise understate the extreme)."""
    return bool(ex) and ex["first_ts"] - day_start <= first_within_s and ex["max_gap_s"] <= max_gap_s


def parse_brackets(markets: list[dict]) -> list[dict]:
    """Normalise a Kalshi event's markets into integer-F brackets sorted ascending.
    less(cap)=> <=cap-1 ; greater(floor)=> >=floor+1 ; between(floor,cap)=> floor..cap."""
    out = []
    for m in markets:
        st = m.get("strike_type")
        if st == "less":
            lo, hi = -999, int(m["cap_strike"]) - 1
        elif st == "greater":
            lo, hi = int(m["floor_strike"]) + 1, 999
        elif st == "between":
            lo, hi = int(m["floor_strike"]), int(m["cap_strike"])
        else:
            continue
        f = lambda k: float(m.get(k) or 0)
        out.append(dict(ticker=m["ticker"], lo=lo, hi=hi, bid=f("yes_bid_dollars"), ask=f("yes_ask_dollars"),
                        bid_size=f("yes_bid_size_fp"), ask_size=f("yes_ask_size_fp"), oi=f("open_interest_fp"),
                        result=m.get("result") or "", exp=m.get("expiration_value") or ""))
    return sorted(out, key=lambda b: (b["lo"], b["hi"]))


def eliminated(b: dict, kind: str, extreme: float, margin: float = 0.0) -> bool:
    """High market: a bracket whose upper bound is below the running max (minus margin) cannot win.
    Low market: bracket whose lower bound is above the running min (plus margin) cannot win."""
    if kind == "high":
        return b["hi"] < extreme - margin
    return b["lo"] > extreme + margin


def _gap_table() -> dict[int, tuple[list[float], list[int]]]:
    t = {}
    for h, d in GAP_COUNTS.items():
        keys = sorted(float(k) for k in d)
        cum, run = [], 0
        for k in keys:
            run += d[str(k) if str(k) in d else k]
            cum.append(run)
        t[int(h)] = (keys, cum)
    return t


_TABLE = _gap_table()


def gap_cdf(hour: int, x: float) -> float:
    """P(final_high - running_max <= x | LST hour), empirical (pooled 20 cities, 61 days)."""
    h = min(max(hour, min(_TABLE)), max(_TABLE))
    keys, cum = _TABLE[h]
    i = bisect.bisect_right(keys, x)
    return (cum[i - 1] / cum[-1]) if i else 0.0


def bracket_probs(brackets: list[dict], rmax: float, hour: int) -> list[float]:
    """Model probability for each (high) bracket given the running max and the LST hour."""
    ps = []
    for b in brackets:
        up = 1.0 if b["hi"] >= 999 else gap_cdf(hour, b["hi"] - rmax)
        dn = 0.0 if b["lo"] <= -999 else gap_cdf(hour, b["lo"] - 1 - rmax)
        ps.append(max(up - dn, 0.0))
    return ps


def taker_fee(price: float, contracts: int) -> float:
    return FeeSchedule().taker_fee(price, contracts)


def decide(brackets: list[dict], probs: list[float], hour: int, age_s: float, max_contracts: int,
           already: set[str], rule: dict = RULE) -> list[dict]:
    """Would-be taker trades. YES buys at ask+slip, NO buys at (1-bid)+slip; needs model prob minus
    all-in cost (price + per-contract fee) > rule['edge']; one entry per bracket per day; size
    capped by displayed top-of-book size."""
    if not (rule["hmin"] <= hour <= rule["hmax"]) or age_s > rule["max_obs_age_s"]:
        return []
    out = []
    for b, p in zip(brackets, probs):
        if b["ticker"] in already:
            continue
        cands = []
        if 0 < b["ask"] < 1 and b["ask_size"] >= 1:
            cands.append(("yes", min(b["ask"] + rule["slip"], 0.99), p, b["ask_size"]))
        if 0 < b["bid"] < 1 and b["bid_size"] >= 1:
            cands.append(("no", min(1 - b["bid"] + rule["slip"], 0.99), 1 - p, b["bid_size"]))
        best = None
        for side, px, pm, size in cands:
            n = int(min(max_contracts, size))
            if n < 1:
                continue
            edge = pm - px - taker_fee(px, n) / n
            if edge > rule["edge"] and (best is None or edge > best["edge"]):
                best = dict(ticker=b["ticker"], side=side, price=round(px, 4), contracts=n,
                            fee=taker_fee(px, n), p_model=round(pm, 4), edge=round(edge, 4))
        if best:
            out.append(best)
    return out


def settle_pnl(side: str, price: float, contracts: int, fee: float, result: str) -> float:
    """Net dollars: contracts*(payoff - price) - fee. result is the YES-market outcome."""
    win = (result == "yes") == (side == "yes")
    return round(contracts * ((1.0 if win else 0.0) - price) - fee, 6)


def event_date(event_ticker: str) -> Optional[dt.date]:
    m = re.search(r"-(\d\d[A-Z]{3}\d\d)", event_ticker)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%y%b%d").date()
    except ValueError:
        return None


def metar_obs(rec: dict) -> Optional[tuple[str, float, float]]:
    """AviationWeather JSON record -> (icao, unix_ts, temp_F)."""
    if rec.get("temp") is None or rec.get("obsTime") is None:
        return None
    return rec["icaoId"], float(rec["obsTime"]), c_to_f(float(rec["temp"]))


def nws_obs(feat: dict) -> Optional[tuple[float, float]]:
    p = feat.get("properties", {})
    v = (p.get("temperature") or {}).get("value")
    if v is None or not p.get("timestamp"):
        return None
    t = dt.datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")).timestamp()
    return t, c_to_f(float(v))


# ====================================================================== I/O shell
class Http:
    """<= 1 request/sec across all APIs; exponential back-off on 429/5xx."""

    def __init__(self, min_gap: float = 1.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.min_gap, self.last, self.n = min_gap, 0.0, 0

    def get(self, url: str, params: Optional[dict] = None, tries: int = 6):
        for a in range(tries):
            wait = self.min_gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.n += 1
            try:
                r = self.s.get(url, params=params, timeout=20)
            except requests.RequestException as e:
                log.warning("http error %s", e)
                time.sleep(min(2 ** a, 60))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                time.sleep(min(float(ra) if ra and ra.isdigit() else 2 ** (a + 1), 120))
                continue
            if r.status_code == 204:
                return None
            if r.status_code != 200:
                log.warning("http %s %s", r.status_code, url)
                return None
            return r.json()
        return None


def _append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


class WeatherRecorder:
    def __init__(self, outdir: str = "data/weather", paper: bool = True, lows: bool = True, hourly: bool = True,
                 max_contracts: int = 10, cities: Optional[list[str]] = None):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.http = Http()
        self.paper, self.lows, self.hourly, self.maxc = paper, lows, hourly, max_contracts
        self.cities = {k: v for k, v in CITIES.items() if not cities or k in cities}
        self.obs: dict[str, dict[float, float]] = defaultdict(dict)  # icao -> {ts: F}
        self.state = json.loads((self.dir / "state.json").read_text()) if (self.dir / "state.json").exists() else {}
        self.state.setdefault("entered", [])
        self.state.setdefault("open_trades", [])
        self.state.setdefault("settled_events", [])
        self.state.setdefault("dayclose", {})
        self.state.setdefault("seen", [])  # series that ever had an open event (skip dead low tickers)
        self.state.setdefault("lastpoll", {})
        self._load_obs_cache()
        self._backfilled = False

    # ---- obs
    def _load_obs_cache(self):
        p = self.dir / "obscache.json"
        if p.exists():
            for icao, d in json.loads(p.read_text()).items():
                self.obs[icao] = {float(t): f for t, f in d.items()}

    def _save(self):
        cut = time.time() - 60 * 3600
        for icao in self.obs:
            self.obs[icao] = {t: f for t, f in self.obs[icao].items() if t >= cut}
        (self.dir / "obscache.json").write_text(json.dumps({k: {str(t): f for t, f in v.items()} for k, v in self.obs.items()}))
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.dir / "state.json")

    def fetch_obs(self):
        stations = sorted({v[0] for v in self.cities.values()} - {"KNYC"} | (set(HOURLY_EXTRA_STATIONS) if self.hourly else set()))
        hours = 3
        batches = [stations]
        if not self._backfilled:  # AWC caps ~400 records/call: backfill ~30h in small batches
            hours = 48
            batches = [stations[i:i + 4] for i in range(0, len(stations), 4)]
        for b in batches:
            d = self.http.get(AWC, {"ids": ",".join(b), "format": "json", "hours": hours})
            for rec in d or []:
                o = metar_obs(rec)
                if o:
                    self.obs[o[0]][o[1]] = o[2]
        if any(v[0] == "KNYC" for v in self.cities.values()):
            start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48 if not self._backfilled else 3)).strftime("%Y-%m-%dT%H:%M:%SZ")
            d = self.http.get(NWS.format("KNYC"), {"start": start, "limit": 500})
            for f in (d or {}).get("features", []):
                o = nws_obs(f)
                if o:
                    self.obs["KNYC"][o[0]] = o[1]
        self._backfilled = True

    # ---- markets
    def open_markets(self, series: str) -> dict[str, list[dict]]:
        d = self.http.get(f"{KALSHI}/markets", {"series_ticker": series, "status": "open", "limit": 200})
        ev: dict[str, list[dict]] = defaultdict(list)
        for m in (d or {}).get("markets", []):
            ev[m["event_ticker"]].append(m)
        return ev

    def cycle(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        self.fetch_obs()
        stats = dict(ts=now, events=0, trades=0, settled=0)
        for series, (icao, tz, low) in self.cities.items():
            ex = running_extremes(self.obs[icao].items(), now, tz)
            _, _, today = lst_day_bounds(now, tz)
            _append(self.dir / "obs.jsonl", dict(ts=now, station=icao, series=series, **(ex or {})))
            for kind, ser in (("high", series), ("low", low if self.lows else None)):
                if not ser:
                    continue
                for et, ms in self.open_markets(ser).items():
                    if event_date(et) != today:
                        continue
                    br = parse_brackets(ms)
                    if len(br) < 3:
                        continue
                    stats["events"] += 1
                    if ser not in self.state["seen"]:
                        self.state["seen"].append(ser)
                    rec = dict(ts=now, kind=kind, series=ser, event=et, date=today.isoformat(), hour=lst_hour(now, tz),
                               obs=ex, brackets=br)
                    if ex and kind == "high":
                        probs = bracket_probs(br, ex["rmax"], rec["hour"])
                        for b, p in zip(br, probs):
                            b["p"] = round(p, 4)
                        for b in br:
                            b["dead"] = eliminated(b, "high", ex["rmax"])
                        if self.paper and coverage_ok(ex, lst_day_bounds(now, tz)[0]):
                            for t in decide(br, probs, rec["hour"], ex["age_s"], self.maxc, set(self.state["entered"])):
                                t.update(ts=now, event=et, series=ser, rmax=ex["rmax"], hour=rec["hour"], date=today.isoformat())
                                self.state["entered"].append(t["ticker"])
                                self.state["open_trades"].append(t)
                                _append(self.dir / "paper_trades.jsonl", t)
                                stats["trades"] += 1
                    elif ex and kind == "low":
                        for b in br:
                            b["dead"] = eliminated(b, "low", ex["rmin"])
                    _append(self.dir / "books.jsonl", rec)
        if self.hourly:
            for et, ms in self.open_markets(HOURLY_SERIES).items():
                _append(self.dir / "hourly.jsonl", dict(
                    ts=now, event=et, obs={s: max(self.obs[s].items()) if self.obs[s] else None for s in ("KNYC",) + HOURLY_EXTRA_STATIONS},
                    mk=[dict(t=m["ticker"], k=m.get("floor_strike"), bid=m.get("yes_bid_dollars"), ask=m.get("yes_ask_dollars"),
                             bs=m.get("yes_bid_size_fp"), as_=m.get("yes_ask_size_fp"), close=m.get("close_time")) for m in ms]))
        stats["settled"] = self.settle(now)
        self._save()
        stats["http"] = self.http.n
        return stats

    # ---- settlement + obs-vs-Kalshi mismatch log
    def settle(self, now: float) -> int:
        n = 0
        for series, (icao, tz, low) in self.cities.items():
            start, _, today = lst_day_bounds(now, tz)
            yday = today - dt.timedelta(days=1)
            key = f"{series}:{yday}"
            if key not in self.state["dayclose"]:  # snapshot obs-derived extremes of the finished LST day
                ex = running_extremes(self.obs[icao].items(), start - 1, tz)
                ok = coverage_ok(ex, start - 86400, max_gap_s=9000) and ex["age_s"] <= 9000
                self.state["dayclose"][key] = dict(rmax=ex["rmax"], rmin=ex["rmin"], n=ex["n"]) if ok else None
            for kind, ser in (("high", series), ("low", low if self.lows else None)):
                if not ser:
                    continue
                evt = f"{ser}-{yday.strftime('%y%b%d').upper()}"
                if (evt in self.state["settled_events"] or key not in self.state["dayclose"] or ser not in self.state["seen"]
                        or now - self.state["lastpoll"].get(evt, 0) < 1800):
                    continue
                self.state["lastpoll"][evt] = now
                d = self.http.get(f"{KALSHI}/markets", {"event_ticker": evt, "limit": 20})
                ms = (d or {}).get("markets", [])
                if not ms or any(m.get("status") not in ("finalized", "settled") or not m.get("result") for m in ms):
                    continue
                br = parse_brackets(ms)
                dc = self.state["dayclose"][key]  # None when obs coverage was incomplete
                ev = br[0]["exp"]
                try:
                    val = float(ev)
                except ValueError:
                    val = None
                obsv = (dc["rmax"] if kind == "high" else dc["rmin"]) if dc else None
                _append(self.dir / "settlements.jsonl", dict(ts=now, event=evt, kind=kind, expiration_value=val, obs_extreme=obsv,
                                                              results={b["ticker"]: b["result"] for b in br}))
                if val is not None and dc:
                    _append(self.dir / "mismatch.jsonl", dict(event=evt, kind=kind, kalshi=val, obs=obsv, diff=round(val - obsv, 2), n_obs=dc["n"]))
                res = {b["ticker"]: b["result"] for b in br}
                keep = []
                for t in self.state["open_trades"]:
                    if t["ticker"] in res:
                        _append(self.dir / "paper_settled.jsonl", dict(ts=now, **t, result=res[t["ticker"]],
                                pnl=settle_pnl(t["side"], t["price"], t["contracts"], t["fee"], res[t["ticker"]])))
                    else:
                        keep.append(t)
                self.state["open_trades"] = keep
                self.state["settled_events"].append(evt)
                n += 1
        return n

    def run(self, interval: float = 600, once: bool = False):
        while True:
            t0 = time.time()
            try:
                s = self.cycle()
                log.info("cycle ok: %s", s)
                print(json.dumps(s), flush=True)
            except Exception:  # keep the recorder alive
                log.exception("cycle failed")
            if once:
                return
            time.sleep(max(5.0, interval - (time.time() - t0)))


# ====================================================================== report
def report(outdir: str = "data/weather") -> str:
    d = Path(outdir)
    tr, st, mm = _read(d / "paper_settled.jsonl"), _read(d / "settlements.jsonl"), _read(d / "mismatch.jsonl")
    trades, books = _read(d / "paper_trades.jsonl"), _read(d / "books.jsonl")
    L = [f"paper trades placed: {len(trades)}   settled: {len(tr)}   open: {len(trades) - len(tr)}"]
    if tr:
        pnl = sum(t["pnl"] for t in tr)
        cost = sum(t["price"] * t["contracts"] + t["fee"] for t in tr)
        wins = sum(1 for t in tr if t["pnl"] > 0)
        byday: dict[str, float] = defaultdict(float)
        for t in tr:
            byday[t["date"]] += t["pnl"]
        v = list(byday.values())
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
        L.append(f"net P&L ${pnl:.2f} on ${cost:.2f} staked ({pnl / cost:+.1%}); win rate {wins / len(tr):.1%}; fees ${sum(t['fee'] for t in tr):.2f}")
        L.append(f"per-day P&L mean ${m:.2f} sd ${sd:.2f} over {len(v)} days (cluster unit = day)")
    # calibration: last pre-settlement model prob per bracket vs outcome, hours >= 12
    res = {}
    for s in st:
        res.update(s["results"])
    seen: dict[str, dict] = {}
    for b in books:
        if b["kind"] != "high" or b["hour"] < 12:
            continue
        for x in b["brackets"]:
            if "p" in x and x["ticker"] in res:
                seen[(x["ticker"], b["hour"])] = x
    if seen:
        bm = sum((x["p"] - (res[t] == "yes")) ** 2 for (t, h), x in seen.items()) / len(seen)
        mk = [((x["bid"] + x["ask"]) / 2, res[t] == "yes") for (t, h), x in seen.items() if x["ask"] > 0]
        bk = sum((p - y) ** 2 for p, y in mk) / max(len(mk), 1)
        L.append(f"calibration (hour>=12, n={len(seen)} bracket-hours): Brier model {bm:.4f} vs market-mid {bk:.4f}")
    if mm:
        diffs = [m["diff"] for m in mm if m["kind"] == "high"]
        if diffs:
            c = defaultdict(int)
            for x in diffs:
                c[round(x)] += 1
            L.append(f"mismatch kalshi_high - obs_running_max: n={len(diffs)} mean {sum(diffs) / len(diffs):+.2f}F  hist {dict(sorted(c.items()))}")
        lo = [m["diff"] for m in mm if m["kind"] == "low"]
        if lo:
            L.append(f"mismatch kalshi_low - obs_running_min: n={len(lo)} mean {sum(lo) / len(lo):+.2f}F")
    return "\n".join(L)


# ====================================================================== CLI glue
def add_cli(sub) -> None:
    rec = sub.add_parser("weather-record", help="record Kalshi daily-temp brackets + station obs; paper-test the running-max rule (no orders)")
    rec.add_argument("--interval", type=float, default=600, help="seconds between cycles (default 600)")
    rec.add_argument("--once", action="store_true", help="run a single cycle and exit")
    rec.add_argument("--no-paper", action="store_true", help="record only")
    rec.add_argument("--no-lows", action="store_true")
    rec.add_argument("--no-hourly", action="store_true")
    rec.add_argument("--max-contracts", type=int, default=10)
    rec.add_argument("--cities", nargs="*", default=None, help="series tickers, e.g. KXHIGHNY KXHIGHCHI (default all 20)")
    rec.add_argument("--dir", default="data/weather")
    rec.set_defaults(fn=lambda a: (logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"),
                                   WeatherRecorder(a.dir, not a.no_paper, not a.no_lows, not a.no_hourly, a.max_contracts, a.cities)
                                   .run(a.interval, a.once)))
    rp = sub.add_parser("weather-report", help="summarise weather paper trades, P&L, calibration and obs-vs-Kalshi mismatch")
    rp.add_argument("--dir", default="data/weather")
    rp.set_defaults(fn=lambda a: print(report(a.dir)))
