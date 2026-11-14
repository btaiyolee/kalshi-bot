"""Per-series settings: where its data and model live, and which Coinbase product stands in for
the settlement price. Gold paths are unchanged so existing data, models and journals keep working."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeriesSpec:
    series: str
    product: str              # Coinbase product used as the reference price
    data_dir: Path            # markets.csv, candles.csv, ref_coinbase_1m.csv live here
    model_path: Path
    settles_on: str           # for display / documentation
    default_source: str = "candles"   # reference-price features: 'ticks' (raw public trades) or 'candles'

    @property
    def markets_csv(self) -> Path:
        return self.data_dir / "markets.csv"

    @property
    def candles_csv(self) -> Path:
        return self.data_dir / "candles.csv"

    @property
    def ref_csv(self) -> Path:
        return self.data_dir / "ref_coinbase_1m.csv"

    @property
    def ticks_csv(self) -> Path:
        """Raw public trades of the reference product: what the live bot actually sees."""
        return self.data_dir / ("paxg_ticks.csv" if self.series == "KXGOLD15M" else "ticks.csv")


SERIES = {
    "KXGOLD15M": SeriesSpec("KXGOLD15M", "PAXG-USD", Path("data"), Path("models/model.json"),
                            "Pyth gold, 1-minute candle close", default_source="ticks"),
    "KXBTC15M": SeriesSpec("KXBTC15M", "BTC-USD", Path("data/btc"), Path("models/model_btc.json"),
                           "CF Benchmarks BRTI, simple average of the 60 seconds before the window edge"),
}


def spec_for(series: str) -> SeriesSpec:
    if series not in SERIES:
        raise SystemExit(f"Unknown series {series}. Known: {', '.join(SERIES)}")
    return SERIES[series]
