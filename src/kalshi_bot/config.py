"""Runtime configuration: environment variables (optionally from a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv

from kalshi_bot.kalshi_client import DEMO_URL, PROD_URL


# Which Coinbase product stands in for each Kalshi series' settlement price.
REFERENCE_PRODUCTS = {"KXGOLD15M": "PAXG-USD", "KXBTC15M": "BTC-USD"}


# Fields that may differ without making a run "non-standard" (see Config.deviations).
NON_RULE_FIELDS = {"env", "key_id", "private_key_path", "execution", "paper_bankroll", "poll_s"}


@dataclass(frozen=True)
class Config:
    # --- connection -------------------------------------------------------
    env: str = "demo"                      # "demo" (fake money) or "prod" (real money)
    key_id: str = ""
    private_key_path: str = ""
    series: str = "KXGOLD15M"              # KXGOLD15M (what the model is fitted on) or KXBTC15M (plumbing tests)
    reference_product: str = ""            # Coinbase product; default chosen from the series
    # --- execution --------------------------------------------------------
    execution: str = "taker"               # live style: "taker" or "maker". Paper runs BOTH side by side.
    maker_improve: float = 0.01            # rest this far above the best bid (below the best ask for NO)
    maker_rest_s: float = 60.0             # cancel an unfilled resting order after this long
    max_attempts_per_window: int = 3       # order attempts per style per window (missed/expired ones retry)
    # --- entry rules ------------------------------------------------------
    contracts: int = 10                    # CAP on contracts per trade (risk sizing may use fewer)
    min_net_edge: float = 0.04             # required edge per contract AFTER fees ($)
    slippage_cap: float = 0.01             # worst taker price we accept vs. the quote we saw ($)
    max_spread: float = 0.03               # skip thin / jumpy books
    min_tau_min: float = 0.75              # don't trade in the final 45s (settlement noise)
    max_tau_min: float = 14.0              # ...or before the window has a reference move
    max_ref_age_s: float = 5.0             # skip if the reference price is stale
    max_trades_per_window: int = 1
    # --- risk rules (per execution style) ---------------------------------
    paper_bankroll: float = 1000.0         # pretend account size for paper runs ($)
    risk_per_trade_pct: float = 0.005      # max cost per WINDOW, as a fraction of bankroll (split if several trades/window)
    daily_loss_pct: float = 0.02           # stop for the day after losing this fraction
    max_consecutive_losses: int = 3        # ...then cool down: counted in WINDOWS (x trades per window)
    cooldown_min: float = 30.0
    max_open_cost: float = 50.0            # absolute $ cap on a single order
    # --- early exit (paper experiment) -------------------------------------
    exit_rule: str = "none"                # "none" = hold to settlement; "model" = sell early when the
                                           # model says selling beats holding (paper only)
    exit_margin: float = 0.02              # ...by at least this much per contract ($)
    poll_s: float = 0.5                    # how often the quote and reference price are re-read (seconds)
    # --- paper realism ----------------------------------------------------
    paper_latency_s: float = 2.0           # delay between decision and simulated fill / order arrival

    def deviations(self) -> dict:
        """Trading-rule settings that differ from the production defaults (empty = standard).
        Connection details, the live execution style, and the paper bankroll size don't count:
        they don't change which trades the strategy takes, only where it runs or how big it is."""
        default = Config()
        ignore = NON_RULE_FIELDS
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in ignore and getattr(self, f.name) != getattr(default, f.name)}

    def profile(self) -> str:
        """"standard" (all defaults), "data" (only the trades-per-window limits raised: the
        first trade in each window is exactly the standard strategy), "exit" (only the early-exit
        rule differs: same entries as standard), or "custom"."""
        dev = set(self.deviations())
        if not dev:
            return "standard"
        if dev <= {"max_trades_per_window", "max_attempts_per_window"}:
            return "data"
        if dev <= {"exit_rule", "exit_margin"}:
            return "exit"          # entries identical to standard; only the exit differs
        return "custom"

    @property
    def ref_product(self) -> str:
        product = self.reference_product or REFERENCE_PRODUCTS.get(self.series, "")
        if not product:
            raise SystemExit(f"No reference price known for series {self.series}. Set BOT_REFERENCE_PRODUCT "
                             f"to a Coinbase product (e.g. BTC-USD). Known series: {', '.join(REFERENCE_PRODUCTS)}.")
        return product

    @property
    def base_url(self) -> str:
        return PROD_URL if self.env == "prod" else DEMO_URL

    @classmethod
    def from_env(cls, standard: bool = False) -> "Config":
        """Every field can be overridden by env var BOT_<FIELD> (KALSHI_* for connection).
        ``standard=True`` ignores every override of the trading rules, so a test is guaranteed
        to use the production defaults no matter what is in .env or the shell."""
        load_dotenv()
        names = {"env": "KALSHI_ENV", "key_id": "KALSHI_KEY_ID",
                 "private_key_path": "KALSHI_PRIVATE_KEY_PATH", "series": "KALSHI_SERIES"}
        default = cls()
        kwargs = {}
        for f in fields(cls):
            if standard and f.name not in NON_RULE_FIELDS:
                continue
            raw = os.environ.get(names.get(f.name, f"BOT_{f.name.upper()}"))
            if raw not in (None, ""):
                kwargs[f.name] = type(getattr(default, f.name))(raw)
        return cls(**kwargs)
