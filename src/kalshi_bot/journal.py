"""Append-only JSONL event log. Every snapshot, decision, fill, and settlement is
recorded so a run can be audited and re-analysed later (see ``report.py``)."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Journal:
    def __init__(self, directory: Path = Path("logs"), mode: str = "paper"):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._fh = open(self.path, "a", buffering=1)   # line-buffered: survives crashes

    def write(self, event: str, **fields) -> None:
        self._fh.write(json.dumps({"t": round(time.time(), 3), "event": event, **fields}) + "\n")

    def close(self) -> None:
        self._fh.close()
