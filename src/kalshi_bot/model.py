"""Probability model: P(window resolves YES) given the market's own quote and what the
reference price has done.

    p = sigmoid( w . features + intercept )

    features
      logit_mid  market-implied log-odds (the market already knows most things)
      z_vol      move since window open, in units of expected remaining volatility:
                 x / (vol30 * sqrt(minutes_left + 1))
      m3n, m5n   3- and 5-minute momentum in units of volatility

``vol30`` is the standard deviation of the last 30 one-minute reference moves ($).
The same ``design`` function builds features for the backtest (arrays) and the live bot
(scalars), so there is exactly one definition of every feature.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path("models/model.json")
VOL_FLOOR = 0.05          # $/sqrt-minute; avoids divide-by-~zero when the thin PAXG tape is flat
FEATURES = ("logit_mid", "z_vol", "m3n", "m5n")


def design(mid, x, tau, vol, m3, m5) -> np.ndarray:
    """Feature matrix (or vector, for scalar inputs). Last axis follows ``FEATURES``."""
    mid = np.clip(np.asarray(mid, dtype=float), 1e-3, 1 - 1e-3)
    vol = np.maximum(np.asarray(vol, dtype=float), VOL_FLOOR)
    tau = np.asarray(tau, dtype=float)
    cols = [np.log(mid / (1 - mid)),
            np.asarray(x, dtype=float) / (vol * np.sqrt(tau + 1.0)),
            np.asarray(m3, dtype=float) / (vol * np.sqrt(3.0)),
            np.asarray(m5, dtype=float) / (vol * np.sqrt(5.0))]
    return np.stack(np.broadcast_arrays(*cols), axis=-1)


@dataclass(frozen=True)
class Model:
    weights: tuple[float, ...]
    intercept: float

    def predict(self, mid, x, tau, vol, m3, m5):
        s = design(mid, x, tau, vol, m3, m5) @ np.asarray(self.weights) + self.intercept
        p = 1.0 / (1.0 + np.exp(-s))
        return float(p) if np.ndim(p) == 0 else p

    def explain(self, mid, x, tau, vol, m3, m5) -> dict:
        """Per-feature value and contribution to the log-odds, for the dashboard."""
        f = design(mid, x, tau, vol, m3, m5)
        return {name: {"value": float(v), "weight": float(w), "contribution": float(v * w)}
                for name, v, w in zip(FEATURES, f, self.weights)}

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**asdict(self), "features": FEATURES}, indent=2))

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "Model":
        raw = json.loads(path.read_text())
        if tuple(raw.get("features", ())) != FEATURES:
            raise SystemExit(f"{path} was fitted with different features; run `kalshi-bot fit`.")
        return cls(tuple(raw["weights"]), raw["intercept"])


def fit(df) -> Model:
    """Fit on a dataset from ``backtest.dataset.build`` (mid, x, tau, vol30, m3, m5, yes)."""
    from sklearn.linear_model import LogisticRegression  # dev dependency; fitting only

    X = design(df.mid.to_numpy(), df.x.to_numpy(), df.tau.to_numpy(),
               df.vol30.to_numpy(), df.m3.to_numpy(), df.m5.to_numpy())
    lr = LogisticRegression(C=100, max_iter=500).fit(X, df.yes)
    return Model(tuple(float(w) for w in lr.coef_[0]), float(lr.intercept_[0]))


def predict_df(model: Model, df) -> np.ndarray:
    return model.predict(df.mid.to_numpy(), df.x.to_numpy(), df.tau.to_numpy(),
                         df.vol30.to_numpy(), df.m3.to_numpy(), df.m5.to_numpy())
