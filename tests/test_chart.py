import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from kalshi_bot.chart import PAGE, build_payload


def _journal(path: Path, profile, rows):
    start = [] if profile is None else [{"t": 0, "event": "start", "profile": profile, "standard": profile == "standard"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in start + rows))      # real journals end every line with \n


def test_payload_groups_by_profile_and_style_with_fixed_colors(tmp_path):
    now = time.time()
    fill = lambda oid, st: {"t": now - 30, "event": "fill", "ticker": "KXGOLD15M-X", "style": st, "order_id": oid, "secs_left": 600}
    settled = lambda oid, st, pnl, t: {"t": t, "event": "settled", "ticker": "KXGOLD15M-X", "style": st, "order_id": oid,
                                       "pnl": pnl, "result": "yes", "count": 5}
    _journal(tmp_path / "paper-a.jsonl", "exit", [fill("a1", "taker"), settled("a1", "taker", 1.5, now - 10)])
    _journal(tmp_path / "paper-b.jsonl", "data", [fill("b1", "taker"), fill("b2", "maker"), settled("b1", "taker", -2.0, now - 5)])
    _journal(tmp_path / "paper-c.jsonl", "custom", [settled("c1", "taker", 9.0, now - 5)])
    _journal(tmp_path / "paper-d.jsonl", None, [settled("d1", "maker", 1.0, now - 5)])      # old run, no settings recorded
    by = {s["key"]: s for s in build_payload(str(tmp_path / "paper-*.jsonl"))["series"]}
    assert by["exit_taker"]["slot"] == 2 and by["data_taker"]["slot"] == 1                      # colour follows the run type
    assert by["exit_taker"]["points"][0]["pnl"] == 1.5 and by["exit_taker"]["open"] == 0
    assert by["data_maker"]["open"] == 1                                                         # fill with no result, window still running
    assert by["custom_taker"]["excluded"] and by["none_maker"]["excluded"] and not by["data_taker"]["excluded"]
    assert by["custom_taker"]["slot"] == 0                                                       # grey, not a categorical colour


def test_fills_from_stopped_runs_are_not_shown_as_open_forever(tmp_path):
    old = {"t": time.time() - 3 * 3600, "event": "fill", "ticker": "T", "style": "taker", "order_id": "x", "secs_left": 300}
    _journal(tmp_path / "paper-a.jsonl", "standard", [old])
    s = build_payload(str(tmp_path / "paper-*.jsonl"))["series"][0]
    assert s["open"] == 0 and s["unrecorded"] == 1


def test_page_has_hover_legend_and_table_view():
    for needle in ("pointermove", "keydown", "aria-pressed", "textContent", "Table", "include custom"):
        assert needle in PAGE
    assert "innerHTML" not in PAGE.replace("host.innerHTML = '<div class=\"empty\">", "")     # labels are never injected as HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_chart_javascript_helpers():
    js = re.search(r"<script>(.*)</script>", PAGE, re.S).group(1)
    helpers = js[js.index("/* ---- pure helpers"):js.index("/* ---- state ----")]
    checks = Path(__file__).with_name("chart_helpers_check.js").read_text()
    r = subprocess.run(["node", "-e", helpers + checks], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_payload_reads_journals_incrementally_and_matches_a_full_read(tmp_path):
    from kalshi_bot import chart
    now = time.time()
    line = lambda o: json.dumps(o)
    p = tmp_path / "paper-inc.jsonl"
    snap = line({"t": now, "event": "snapshot", "ticker": "T", "style": "taker", "bid": .5})
    st = line({"t": 0, "event": "start", "profile": "standard", "standard": True})
    s1 = line({"t": now - 5, "event": "settled", "ticker": "T-1", "style": "taker", "order_id": "a", "pnl": 1.0, "result": "yes", "count": 5})
    p.write_text("\n".join([st, snap, snap, s1]) + "\n")
    first = chart.build_payload(str(p))["series"][0]
    assert len(first["points"]) == 1
    # a half-written trailing line must not be parsed or lost
    s2 = line({"t": now - 3, "event": "settled", "ticker": "T-2", "style": "taker", "order_id": "b", "pnl": -2.0, "result": "no", "count": 5})
    with open(p, "a") as fh:
        fh.write(snap + "\n" + s2[:20])
    assert len(chart.build_payload(str(p))["series"][0]["points"]) == 1            # still only the complete line
    with open(p, "a") as fh:
        fh.write(s2[20:] + "\n")
    inc = chart.build_payload(str(p))["series"][0]
    assert [pt["pnl"] for pt in inc["points"]] == [1.0, -2.0]
    chart._CACHE.pop(str(p))                                                        # a from-scratch read agrees
    assert chart.build_payload(str(p))["series"][0]["points"] == inc["points"]
