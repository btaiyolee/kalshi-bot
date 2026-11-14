"""Risk rules, one instance per execution style: position sizing from bankroll,
a daily loss stop, and a cooldown after consecutive losses."""

from __future__ import annotations

import math
import time
from datetime import date

from kalshi_bot.config import Config


class RiskManager:
    def __init__(self, cfg: Config, bankroll: float):
        self.cfg = cfg
        self.bankroll = bankroll
        self.day = date.today()
        self.day_pnl = 0.0
        self.consecutive_losses = 0
        self.cooldown_until = 0.0

    @property
    def loss_limit(self) -> int:
        """Consecutive losing TRADES that trigger the cooldown. ``max_consecutive_losses`` is meant
        as losing windows (the original design: 1 trade per window), so it is scaled by trades per
        window: 3 windows is 3 trades normally and 9 trades at 3 trades per window."""
        return self.cfg.max_consecutive_losses * max(1, self.cfg.max_trades_per_window)

    def size(self, price: float) -> int:
        """Contracts to buy. ``risk_per_trade_pct`` is the most we risk in ONE WINDOW: trades in the
        same window share one outcome, so with several trades per window it is split between them
        (a window of 3 trades risks the same total as a window of 1)."""
        per_trade_pct = self.cfg.risk_per_trade_pct / max(1, self.cfg.max_trades_per_window)
        by_risk = math.floor(self.bankroll * per_trade_pct / max(price, 0.01))
        by_cap = math.floor(self.cfg.max_open_cost / max(price, 0.01))
        return max(0, min(self.cfg.contracts, by_risk, by_cap))

    def block_reason(self, now: float | None = None) -> str:
        """Empty string if trading is allowed, else why it is paused."""
        now = time.time() if now is None else now
        if date.today() != self.day:                      # new day: reset the daily counters
            self.day, self.day_pnl = date.today(), 0.0
        limit = self.cfg.daily_loss_pct * self.bankroll
        if self.day_pnl <= -limit:
            return f"daily loss stop (-${limit:.2f})"
        if now < self.cooldown_until:
            return f"cooldown after {self.loss_limit} straight losses ({self.cooldown_until - now:.0f}s left)"
        return ""

    def record(self, pnl: float, now: float | None = None) -> None:
        """Feed in each settled trade's P&L."""
        now = time.time() if now is None else now
        self.bankroll += pnl
        self.day_pnl += pnl
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
        if self.consecutive_losses >= self.loss_limit:
            self.cooldown_until = now + self.cfg.cooldown_min * 60
            self.consecutive_losses = 0
