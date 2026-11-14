import datetime as dt
import math

from kalshi_bot import weather as w


def utc(y, m, d, h=0, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=dt.timezone.utc).timestamp()


def test_standard_offsets_ignore_dst():
    assert w.std_offset_hours("America/New_York") == -5
    assert w.std_offset_hours("America/Chicago") == -6
    assert w.std_offset_hours("America/Denver") == -7
    assert w.std_offset_hours("America/Los_Angeles") == -8
    assert w.std_offset_hours("America/Phoenix") == -7


def test_lst_day_in_summer_runs_01_to_01_clock_time():
    # July (EDT): LST midnight is 05:00Z = 01:00 EDT
    s, e, d = w.lst_day_bounds(utc(2026, 7, 10, 12), "America/New_York")
    assert s == utc(2026, 7, 10, 5) and e == utc(2026, 7, 11, 5) and d == dt.date(2026, 7, 10)
    # 04:30Z on Jul 10 is still LST Jul 9 (23:30 EST)
    assert w.lst_day_bounds(utc(2026, 7, 10, 4, 30), "America/New_York")[2] == dt.date(2026, 7, 9)


def test_lst_day_november_standard_time_and_arizona():
    s, _, d = w.lst_day_bounds(utc(2026, 11, 8, 3), "America/New_York")  # 22:00 EST Nov 7
    assert d == dt.date(2026, 11, 7) and s == utc(2026, 11, 7, 5)
    # Phoenix never shifts: 06:59Z Aug 1 is Jul 31 23:59 MST
    assert w.lst_day_bounds(utc(2026, 8, 1, 6, 59), "America/Phoenix")[2] == dt.date(2026, 7, 31)
    assert w.lst_hour(utc(2026, 11, 8, 18), "America/Chicago") == 12


def test_running_extremes_only_counts_today_lst_and_past():
    now = utc(2026, 11, 8, 20)  # 15:00 EST
    obs = [(utc(2026, 11, 8, 4, 59), 80.0),   # 23:59 EST previous day -> excluded
           (utc(2026, 11, 8, 6), 50.0), (utc(2026, 11, 8, 15), 61.5), (utc(2026, 11, 8, 19, 51), 58.0),
           (utc(2026, 11, 8, 21), 90.0)]      # future -> excluded
    ex = w.running_extremes(obs, now, "America/New_York")
    assert ex["rmax"] == 61.5 and ex["rmin"] == 50.0 and ex["n"] == 3 and ex["last_f"] == 58.0
    assert ex["age_s"] == 9 * 60
    assert w.running_extremes([], now, "America/New_York") is None


def mk(strike, **kw):
    d = dict(ticker="T", yes_bid_dollars="0.10", yes_ask_dollars="0.12", yes_bid_size_fp="50", yes_ask_size_fp="40")
    d.update(strike_type=strike, **kw)
    return d


def six():
    ms = [mk("less", cap_strike=60, ticker="L"), mk("between", floor_strike=60, cap_strike=61, ticker="B60"),
          mk("between", floor_strike=62, cap_strike=63, ticker="B62"), mk("between", floor_strike=64, cap_strike=65, ticker="B64"),
          mk("between", floor_strike=66, cap_strike=67, ticker="B66"), mk("greater", floor_strike=67, ticker="G")]
    return w.parse_brackets(list(reversed(ms)))


def test_parse_and_elimination():
    br = six()
    assert [(b["lo"], b["hi"]) for b in br][0] == (-999, 59) and (br[-1]["lo"], br[-1]["hi"]) == (68, 999)
    dead = [b["ticker"] for b in br if w.eliminated(b, "high", 64.4)]
    assert dead == ["L", "B60", "B62"]  # hi<64.4 ; B64 (64..65) alive
    deadlow = [b["ticker"] for b in br if w.eliminated(b, "low", 63.0)]
    assert deadlow == ["B64", "B66", "G"]


def test_gap_cdf_monotone_and_late_hours_tighter():
    for h in (12, 17, 22):
        assert 0 <= w.gap_cdf(h, -5) <= w.gap_cdf(h, 0) <= w.gap_cdf(h, 3) <= w.gap_cdf(h, 99) == 1.0
    assert w.gap_cdf(21, 2.0) > w.gap_cdf(12, 2.0)  # less rise left late in the day


def test_bracket_probs_sum_to_one_and_dead_have_near_zero():
    br = six()
    ps = w.bracket_probs(br, 64.4, 18)
    assert abs(sum(ps) - 1) < 1e-9
    assert ps[0] < 0.01 and ps[1] < 0.01 and ps[2] < 0.01  # dead by the running max


def test_taker_fee_rounds_up_per_order():
    assert w.taker_fee(0.5, 1) == 0.02            # 0.0175 -> 2c
    assert w.taker_fee(0.5, 10) == 0.18           # 0.175 -> 18c
    assert w.taker_fee(0.01, 1) == 0.01
    assert math.isclose(w.taker_fee(0.9, 100), 0.63)


def test_decide_buys_underpriced_yes_with_slippage_and_caps_size():
    br = [dict(ticker="X", lo=64, hi=65, bid=0.10, ask=0.12, bid_size=50, ask_size=4)]
    out = w.decide(br, [0.60], 16, 600, 10, set())
    assert len(out) == 1 and out[0]["side"] == "yes" and out[0]["price"] == 0.13 and out[0]["contracts"] == 4
    assert out[0]["fee"] == w.taker_fee(0.13, 4)
    assert w.decide(br, [0.60], 16, 600, 10, {"X"}) == []        # one entry per bracket
    assert w.decide(br, [0.60], 9, 600, 10, set()) == []         # outside hour window
    assert w.decide(br, [0.60], 16, 99999, 10, set()) == []      # stale obs


def test_decide_buys_no_when_model_says_dead():
    br = [dict(ticker="X", lo=60, hi=61, bid=0.20, ask=0.25, bid_size=30, ask_size=30)]
    out = w.decide(br, [0.0], 16, 60, 10, set())
    assert out[0]["side"] == "no" and out[0]["price"] == 0.81 and out[0]["contracts"] == 10
    assert w.decide(br, [0.22], 16, 60, 10, set()) == []         # no edge


def test_settle_pnl_accounting():
    assert w.settle_pnl("yes", 0.13, 4, 0.01, "yes") == round(4 * 0.87 - 0.01, 6)
    assert w.settle_pnl("yes", 0.13, 4, 0.01, "no") == round(-4 * 0.13 - 0.01, 6)
    assert w.settle_pnl("no", 0.81, 10, 0.12, "no") == round(10 * 0.19 - 0.12, 6)
    assert w.settle_pnl("no", 0.81, 10, 0.12, "yes") == round(-8.1 - 0.12, 6)


def test_event_date_and_obs_parsers():
    assert w.event_date("KXHIGHNY-26NOV05") == dt.date(2026, 11, 5)
    assert w.event_date("garbage") is None
    assert w.metar_obs({"icaoId": "KMDW", "obsTime": 100, "temp": 10.0}) == ("KMDW", 100.0, 50.0)
    assert w.metar_obs({"icaoId": "KMDW", "obsTime": 100, "temp": None}) is None
    assert w.nws_obs({"properties": {"timestamp": "2026-11-08T12:00:00+00:00", "temperature": {"value": 20.0}}}) == (utc(2026, 11, 8, 12), 68.0)
    assert w.nws_obs({"properties": {"timestamp": "2026-11-08T12:00:00+00:00", "temperature": {"value": None}}}) is None


def test_coverage_guard_rejects_partial_days():
    now = utc(2026, 11, 8, 20)
    start = w.lst_day_bounds(now, "America/New_York")[0]
    full = [(start + h * 3600 + 3060, 50.0 + h) for h in range(0, 15)]
    assert w.coverage_ok(w.running_extremes(full, now, "America/New_York"), start)
    late_start = [(start + 8 * 3600, 50.0), (start + 14 * 3600, 60.0)]   # recorder started mid-day
    assert not w.coverage_ok(w.running_extremes(late_start, now, "America/New_York"), start)
    holey = [(start + 3600, 50.0), (start + 8 * 3600, 60.0), (start + 14 * 3600, 61.0)]
    assert not w.coverage_ok(w.running_extremes(holey, now, "America/New_York"), start)
    assert not w.coverage_ok(None, start)
