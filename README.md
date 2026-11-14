# kalshi-bot

> **Status (Nov 2026): no demonstrated edge.** An earlier backtest was contaminated by revised candle data; see
> "What the evidence says" below. This repository is a research/paper-trading framework, not a profitable bot.

Automated, fee-aware trading for Kalshi's **15-minute gold** market (`KXGOLD15M`).
Paper trading is the default. Real orders need `--live` plus a typed confirmation.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/kalshi-bot run                 # paper trade on live data: no credentials needed
```

Paper mode watches real production markets and simulates fills, running a **taker** and a
**maker** strategy side by side on identical signals so you can compare them.

For real orders, create an API key in Kalshi (use the demo site first), copy `.env.example`
to `.env`, fill in `KALSHI_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH`, set `KALSHI_ENV=demo` or `prod`, then:

```bash
.venv/bin/kalshi-bot run --live          # one style, set by BOT_EXECUTION=taker|maker
```

| Command | What it does |
|---|---|
| `kalshi-bot run [--live]` | Trading loop with a full-screen live dashboard (`--no-dashboard` for plain logs) |
| `kalshi-bot trades` | Every trade: when, why, at what price, and how it ended |
| `kalshi-bot chart` | Live net-P&L chart of the paper runs (local web page at http://127.0.0.1:8765, refreshes itself) |
| `kalshi-bot compare` | Early-exit experiment vs hold-to-settlement, paired window by window |
| `kalshi-bot report` | Proof gates: P&L with confidence intervals, cost stress tests, per-day stability |
| `kalshi-bot collect` / `fit` / `backtest` | Download ~67 days of data, fit the model, walk-forward study |

## What you see while it runs

* **Order book**: the YES ladder (asks above, bids below, sizes as bars) with your own resting orders marked.
* **Market**: strike, time left, gold reference price, move since open, and sparklines of Kalshi's mid and the reference.
* **Why**: every model input with its weight and contribution, the model's P(YES) against the market's, and for each
  style the edge arithmetic: `win-probability - price - fee = edge` against the threshold.
* **Trades**: time, seconds left, style, side, quantity, quoted price, fill price, fee, outcome, and a one-line reason
  such as `ref +4.2$ (z +1.8, mom3 +1.8) -> model 0.71 vs market 0.55; edge +0.051 after fees`.
* **Session**: per-style fills, win rate, net P&L, fees, and any risk pause.

Everything is also written to `logs/<mode>-<time>.jsonl` (every second's quote and decision, a full book sample every
5 seconds, each signal with its numbers, orders, fills, settlements). The dashboard only reads in-memory state, so
showing more cannot slow trading.

## The market and fees

"Will Pyth's gold price at the end of the window be at or above its price at the start?" A coin flip at the open
(49.1% YES over 4,535 windows), very liquid (~317k contracts per window).

Taker fee = `ceil(0.07 x contracts x P x (1-P))`: 1.75c per contract at 50c (3.5% of cost), rounded up per order.
This series is `fee_type: quadratic`, which appears to mean takers only, so **resting maker orders pay nothing**.
Verify on your own fills: the journal records the fee Kalshi actually charged.

## Strategy

The market already prices the window well (a pure volatility model with no market input scores log-loss 0.527 against
the market's own 0.456), so the bot never tries to predict gold on its own. It asks one question: **has the gold price
moved in a way Kalshi's quote has not absorbed yet?**

`P(yes) = sigmoid(w . [market log-odds, move/vol, 3-min momentum/vol, 5-min momentum/vol] + b)`

* `vol` = standard deviation of the last 30 one-minute gold moves (volatility normalisation).
* **Taker**: buy the cheaper side at the ask if expected profit after fees is at least `BOT_MIN_NET_EDGE` (4c).
* **Maker**: rest an order one step inside the book (free of fees), cancel after 60s if unfilled.
* Skipped when: spread is wide, the reference price is stale, <45s remain, or risk rules pause trading.

### Risk rules (per execution style)

Max cost per window 0.5% of bankroll (`BOT_RISK_PER_TRADE_PCT`; split across trades if `--trades-per-window` > 1, since they share one outcome), stop for the day after losing 2%
(`BOT_DAILY_LOSS_PCT`), 30-minute cooldown after 3 straight losing windows (3 trades normally, 9 at `--trades-per-window 3`), absolute `BOT_MAX_OPEN_COST` cap per order.
Paper bankroll defaults to $1,000 (`BOT_PAPER_BANKROLL`); live uses your real balance.

## What the evidence says (read this before using real money)

**There is no demonstrated edge, on gold or on BTC. Do not trade real money on this.**

An earlier version of this section quoted backtest results of +6.5c to +8.9c per contract. **Those numbers were wrong.**
They were measured on Coinbase's 1-minute *candle* file, and candles are not final when a minute closes: only about
half are complete 2.5 minutes later, 81% after 3-5 minutes, 99% after 5+ (late-reported trades keep being added;
for thin PAXG 60% of minutes carry volume that is not on the public trade tape). A backtest that downloads candles
afterwards therefore reads 1-5 minutes of the future, and a live bot can never see it.

Measured honestly, with features built from the raw public trades the live bot actually reads (`kalshi-bot ticks`,
`kalshi-bot evaluate`):

| Test | Result |
|---|---|
| Same 1,987 gold windows, same model: candle features | +6.9c [+4.5, +9.4] (artifact) |
| Same windows, raw-tick (live-style) features | -0.7c [-3.2, +1.8] |
| Frozen production model, 255 unseen windows, tick features, 4c edge | -3.5c [-10.8, +3.9] |
| Live paper trading (simulated fills), ~5 days | about 0 +/- 5c |
| BTC15M: Kalshi mid vs diffusion fair value from Coinbase + exact strike | Kalshi is sharper at every minutes-left; features add ~0.0001 log-loss; all gates fail |
| ~180 gold signal variants from public-tape microstructure | none beats the Kalshi mid on held-out data |
| Passive market making (validated fill model, 1.8M tape trades) | about -1.6c per filled contract: quotes are run over |
| ~1,100 Kalshi-internal price/time/series cells (6 series) | none survives held-out + correction |

Taker-side, Kalshi gold is efficient before fees (gross -0.2c), so takers lose about the 1c fee. The paper-P&L figure
of "$20-50 a day" was a sizing cap on a strategy whose edge is unproven, not a result.

What would be needed for a real edge (none of it exists here): a gold price faster/closer to settlement than PAXG
(Pyth itself; paid), millisecond Kalshi order-book data, and maker mechanics with a measured fill/queue model.

Use `kalshi-bot evaluate --series ...` (with `--source ticks`) for any new idea. What it does now:

- Walk-forward with a burn-in (first >= 500 windows never traded), then holdout and forward blocks, all out of sample.
- **Primary statistic**: pooled out-of-sample mean net P&L per contract with a one-sided 95% day-block bootstrap
  bound, reported at the 1c trading threshold and the 4c threshold, next to its **MDE** (2.8 x standard error: the
  smallest true edge this sample would detect 80% of the time).
- Frictions: +1c / +2c slippage and a ~2 s latency fill (+0.14c, from journal ask drift). The old 60 s
  next-minute fill and the per-block / days-positive gates are printed as diagnostics only (no power / noisy).
- Two placebos with the same fold design: reference features shuffled, and outcomes shuffled within quote-price x
  minutes-left cells. Both need p < 0.02.
- Verdict: **SUPPORTED** (bound > 0 at 1c, positive after slippage and latency, both placebos), **DEAD** (upper
  bound < +1c), **INCONCLUSIVE** (interval still includes +2c: not evidence of no edge), else NOT SUPPORTED.
  On ~2,000 gold windows the MDE is about 3-4c, so most real-data runs will be INCONCLUSIVE or DEAD, not SUPPORTED.

Never trust a backtest whose reference prices come from a candle endpoint for a thin product.

## Running experiments side by side (paper only)

Every run writes its own journal, so several can run at once in separate terminals without touching each other.
The report separates them by what each run was testing:

```bash
caffeinate -i .venv/bin/kalshi-bot run --standard                          # default settings: the clean evidence
caffeinate -i .venv/bin/kalshi-bot run --standard --trades-per-window 3    # more data on fills/slippage ("data" run)
caffeinate -i .venv/bin/kalshi-bot run --standard --exit model             # early-exit experiment ("exit" run)
.venv/bin/kalshi-bot chart                                                 # watch all of them live
.venv/bin/kalshi-bot report --runs exit                                    # results of one kind of run
.venv/bin/kalshi-bot compare                                               # exit vs hold, paired by window
```

**Early exit** (`--exit model`, optional `--exit-margin 0.02`): entries are identical to the default strategy, but
each open position is re-checked every second. A contract held to settlement is worth the model's probability that
its side wins; selling now nets the bid minus a second taker fee. When the sale beats holding by the margin, the
bot sells (after the same simulated delay as a taker entry, against a fresh book, within the slippage cap). Exits are
booked as `EXIT` with both fees included. Backtest evidence for it is in the research notes; the forward paired
comparison (`compare`) is the real test. It is paper-only: live trading has no sell-order path yet.

**The chart** plots cumulative net P&L after fees from settled trades, one line per run type and style, with a fixed
colour per run type, hover/keyboard readout, a table view, and time-range filters. It reads the journals
incrementally, so it stays cheap as they grow, and it listens on `127.0.0.1` only. Dollar totals are dominated by
position size and by several trades sharing one outcome; use `report` (per-contract, counted per window) for evidence.

## Layout

```
src/kalshi_bot/
  fees.py  model.py  strategy.py  risk.py   pure logic (unit-tested)
  kalshi_client.py  reference.py            Kalshi API (RSA-PSS signing), Coinbase PAXG feed
  brokers.py  maker.py  engine.py           taker + maker execution, trading loop
  journal.py  dashboard.py  report.py       audit log, live view, proof gates
  chart.py                                  live P&L chart (local web page)
  backtest/                                 data collection, dataset, edge study
tests/test_core.py
```

Not financial advice. Prediction-market trading can lose money; the loss limits are guardrails, not guarantees.

## Liquidity-reward paper test: what it measures, what it can't

`kalshi-bot lip-paper` simulates (never places) a YES bid and a NO bid of 100 contracts in ~30 slow,
long-dated markets that carry a Kalshi Liquidity Incentive Program (LIP) pool, using only public endpoints
(`/incentive_programs`, `/series`, `/markets`, orderbook, trades; <= ~1.4 requests/s). Event logs go to
`data/lip/events-*.jsonl`; `kalshi-bot lip-paper --report` prints per-market / per-category / total
estimated reward per day, fills, mark-outs (1 min / 1 h / 1 day after each fill), trade P&L and a
capital-based annualised return. `--select-only` shows the market selection. Formula, units and
assumptions are in the `kalshi_bot/lip.py` docstring.

Measures: the reward share a small quoter would get from the real book depth (reference price at the
level where cumulative size reaches Target/5, DiscountFactor^ticks below it, both sides must hold
Target Size), plus fills and adverse selection under a conservative (strictly-through) and an
optimistic (at-or-through) fill rule.

Cannot measure: the actual LIP payout (we only estimate it; `period_reward` unit of 1e-4 dollars is an
inference), who is an excluded market maker, 1-second snapshot timing, true queue priority, or rare
jump risk (early-close or news events) without weeks of data. Per-market results are correlated and
small-sample; treat anything under ~1 week as anecdotal.

## Weather recorder / paper tester (`weather-record`, `weather-report`)

Records, every 10 min (<= 1 req/s, backs off on 429), the Kalshi bracket book for today's daily-high and daily-low
markets of 20 US cities plus the hourly NYC directional market (`KXTEMPNYCHS`), the latest station observations
(AviationWeather METAR, api.weather.gov for KNYC), the running max/min since LOCAL STANDARD midnight (the NWS CLI
convention, no DST shift), and the model's bracket probabilities (empirical "final high minus running max" by hour,
fit on 61 days x 20 cities). Paper mode logs would-be taker trades (rule: model prob minus price+1c slippage minus
fee > 6c, hours 12-22 LST, <= 10 contracts, one entry per bracket) and settles them from Kalshi results. It never places
orders. Files in `data/weather/` (gitignored): `books.jsonl obs.jsonl hourly.jsonl paper_trades.jsonl
paper_settled.jsonl settlements.jsonl mismatch.jsonl state.json obscache.json`.

    nohup .venv/bin/kalshi-bot weather-record > logs/weather.out 2>&1 &   # run
    .venv/bin/kalshi-bot weather-report                                   # P&L, calibration, Kalshi-vs-ASOS mismatch
    pkill -f weather-record                                               # stop

Backtest verdict (Nov 2026, held-out last 40% of 61 days): the market already prices the running max; every variant loses
2-4c/contract net of fees. See the report in the project history; the recorder exists to collect forward evidence.
