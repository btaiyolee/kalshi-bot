"""LIP paper tester: pure-function core, no network."""

import pytest

from kalshi_bot import lip
from kalshi_bot.lip import (Acct, Order, desired_price, distance_mult, feed_trade, interval_reward,
                            jump_detected, order_fill_qty, parse_book, pull_reasons,
                            reference_price, reward_share)

TARGET, DF = 1000.0, 0.5
YES = {0.50: 100.0, 0.49: 100.0, 0.48: 400.0, 0.45: 500.0}      # cum: 100, 200 (ref 0.49), 600, 1100
NO = {0.45: 300.0, 0.44: 300.0, 0.40: 500.0}


def test_parse_book():
    y, n = parse_book({"orderbook_fp": {"yes_dollars": [["0.4900", "10.00"], ["0.5000", "5"]],
                                         "no_dollars": [["0.4000", "7"]]}})
    assert y == {0.49: 10.0, 0.5: 5.0} and n == {0.4: 7.0}


def test_reference_price_is_fifth_of_target():
    assert reference_price(YES, TARGET) == 0.49          # 100 < 200 <= 200
    assert reference_price({0.5: 199.0}, TARGET) is None  # never reaches 200


def test_discount_by_ticks():
    assert distance_mult(0.49, 0.49, DF) == 1.0
    assert distance_mult(0.60, 0.49, DF) == 1.0          # better than reference
    assert distance_mult(0.48, 0.49, DF) == 0.5
    assert distance_mult(0.45, 0.49, DF) == pytest.approx(0.0625)


def test_reward_share_synthetic_book():
    # our 100 @0.49 joins the yes book: ref moves nowhere (still 0.49, cum 200 -> 300)
    res = reward_share(YES, NO, (0.49, 100), (0.44, 100), TARGET, DF)
    assert res.included
    # yes side: total = 100 + (100+100) + 400*.5 + 500*.0625 = 531.25 ; mine = 100
    assert res.score_yes == pytest.approx(100 / 531.25)
    # no side with ours: 0.45:300, 0.44:400, 0.40:500 -> ref 0.44? 300<200? cum 300>=200 at 0.45
    assert res.ref_no == 0.45
    no_total = 300 + 400 * 0.5 + 500 * 0.5 ** 5
    assert res.score_no == pytest.approx(100 * 0.5 / no_total)
    assert res.score == pytest.approx(res.score_yes + res.score_no)


def test_excluded_when_side_below_target():
    thin = {0.5: 600.0}
    res = reward_share(thin, NO, (0.5, 100), (0.44, 100), TARGET, DF)
    assert not res.included and res.score == 0.0
    # our size can lift a 900 side over the target
    res = reward_share({0.5: 900.0}, NO, (0.5, 100), (0.44, 100), TARGET, DF)
    assert res.included


def test_interval_reward_whole_day_is_pool_times_score_over_two():
    # sole participant on both sides with score 2.0 earns the full pool
    full = lip.RewardResult(1.0, 1.0, True)
    assert interval_reward(full, 14.0, 86400) == pytest.approx(14.0)
    half = lip.RewardResult(0.1, 0.1, True)
    assert interval_reward(half, 14.0, 43200) == pytest.approx(14.0 * 0.5 * 0.1)
    assert interval_reward(lip.RewardResult(0.5, 0.5, False), 14.0, 3600) == 0.0


def test_desired_price_offset_and_no_cross():
    assert desired_price(YES, NO, TARGET, 0) == 0.49
    assert desired_price(YES, NO, TARGET, 1) == 0.50
    # opposite best bid 0.51 => yes ask 0.49 => yes bid must stay <= 0.48
    assert desired_price(YES, {0.51: 500.0}, TARGET, 0) == 0.48


def test_strict_through_fill_yes_and_no():
    o = Order("yes", 0.49, 100, 0.0, 10.0)
    feed_trade(o, 5.0, 0.40, 500)          # before active: ignored
    feed_trade(o, 11.0, 0.49, 500)         # AT our price: not through
    feed_trade(o, 12.0, 0.50, 500)         # above: ignored
    assert order_fill_qty(o, "cons") == 0
    assert order_fill_qty(o, "opt") == 100
    feed_trade(o, 13.0, 0.48, 30.9)        # strictly through
    assert order_fill_qty(o, "cons") == 30
    feed_trade(o, 14.0, 0.40, 500)
    assert order_fill_qty(o, "cons") == 100  # capped at size
    n = Order("no", 0.40, 100, 0.0, 0.0)   # NO bid .40 == YES offer .60
    feed_trade(n, 1.0, 0.60, 100)
    assert order_fill_qty(n, "cons") == 0
    feed_trade(n, 2.0, 0.61, 40)
    assert order_fill_qty(n, "cons") == 40


def test_pull_rules_inventory_close_and_jump():
    base = dict(status="active", secs_to_close=10 * 86400, pulled_until=0, now=1000.0, net=0,
                gross_total=0, size=100, cap=200, total_cap=1000, pull_hours=24)
    assert pull_reasons(**base) == {"yes": None, "no": None}
    r = pull_reasons(**{**base, "net": 150})
    assert r["yes"] == "market_inventory_cap" and r["no"] is None   # 250 > cap; selling side ok
    r = pull_reasons(**{**base, "net": -150})
    assert r["no"] == "market_inventory_cap" and r["yes"] is None
    r = pull_reasons(**{**base, "gross_total": 950, "net": 0})
    assert r["yes"] == r["no"] == "total_inventory_cap"
    assert pull_reasons(**{**base, "secs_to_close": 3600})["yes"] == "near_close"
    assert pull_reasons(**{**base, "pulled_until": 2000.0})["no"] == "cooldown_after_jump"
    assert pull_reasons(**{**base, "status": "closed"})["yes"] == "status=closed"
    assert jump_detected([(0, 0.50), (1, 0.56)], 0.05)
    assert not jump_detected([(0, 0.50), (1, 0.53)], 0.05)


def test_accounting_fill_mtm_and_settlement():
    a = Acct("cons")
    a.apply_fill("yes", 100, 0.40, 0.0)
    assert a.net == 100 and a.spent == pytest.approx(40.0)
    assert a.trade_pnl(0.45) == pytest.approx(5.0)       # marked at mid
    a.apply_fill("no", 100, 0.55, 0.0)                   # hedged pair costs .95, pays 1
    assert a.trade_pnl(0.5) == pytest.approx(5.0)
    pnl = a.settle("yes")
    assert pnl == pytest.approx(5.0) and a.yes_qty == a.no_qty == 0
    b = Acct("cons")
    b.apply_fill("yes", 100, 0.40, 0.25)
    assert b.fees == 0.25
    assert b.settle("no") == pytest.approx(-40.25)


def test_maker_fee_only_when_series_charges():
    from kalshi_bot.fees import FeeSchedule
    assert FeeSchedule(1.0, False).maker_fee(0.5, 100) == 0.0
    assert FeeSchedule(1.0, True).maker_fee(0.5, 100) == 0.44   # ceil(0.0175*100*.25 = 0.4375)


def test_capital_includes_resting_collateral_and_held_cost():
    a = Acct("cons")
    a.orders["yes"] = Order("yes", 0.5, 100, 0, 0)
    a.orders["no"] = Order("no", 0.4, 100, 0, 0)
    assert a.capital() == pytest.approx(90.0)
    a.apply_fill("yes", 100, 0.5, 0.0)
    assert a.capital() == pytest.approx(50.0 + 90.0)


def test_balanced_order_interleaves_categories():
    import random
    order = lip.balanced_order({"A1": "a", "A2": "a", "A3": "a", "B1": "b"}, random.Random(1))
    assert order[0][0] != order[1][0]        # first two come from different categories


def test_group_programs_units():
    raw = [{"market_ticker": "X-1", "incentive_type": "liquidity", "paid_out": False,
            "start_date": "2026-11-08T00:00:00Z", "end_date": "2026-11-15T00:00:00Z",
            "period_reward": 700000, "target_size_fp": "1000.00", "discount_factor_bps": 5000}]
    p = lip.group_programs(raw)["X-1"][0]
    assert p.pool_day == pytest.approx(10.0) and p.df == 0.5 and p.target == 1000.0
