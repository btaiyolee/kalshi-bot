"""Live net-P&L chart for the paper runs.

    kalshi-bot chart            # opens http://127.0.0.1:8765

A tiny local web server (standard library only). It re-reads the journal files every time the
page asks, so the chart keeps growing while the bots run. Nothing here talks to Kalshi or touches
the running bots; it only reads ``logs/paper-*.jsonl``.

What is plotted: cumulative net P&L after fees from SETTLED trades (including early exits), one
line per run type and execution style. Positions still open are counted in the table but are
not in the lines, because their result is not known yet.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# What each run profile is called on screen, and whether it counts as evidence by default.
GROUP_NAMES = {"standard": "Default settings", "data": "Multi-trade (hold)", "exit": "Early exit",
               "custom": "Custom settings", "none": "Older runs"}
EXCLUDED_BY_DEFAULT = {"custom", "none"}
# Colour follows the run type, never its rank: a fixed slot per (profile, style), so hiding a
# line never repaints the others. Slots 1-6 of the validated categorical palette; 0 = neutral grey.
COLOR_SLOT = {("data", "taker"): 1, ("exit", "taker"): 2, ("data", "maker"): 3,
              ("exit", "maker"): 4, ("standard", "taker"): 5, ("standard", "maker"): 6}


_CACHE: dict[str, dict] = {}     # journal path -> what we have read so far
_WANTED = (b'"event": "settled"', b'"event": "fill"', b'"event": "start"')


def _read_new(path: str) -> dict:
    """Parse only what was appended since the last call, and only the lines we need.

    Journals are dominated by one-second snapshot and book lines (tens of thousands per day).
    Those are skipped with a cheap substring test instead of being parsed, and everything already
    read is remembered, so a refresh costs about the same on day 7 as on day 1."""
    c = _CACHE.setdefault(path, {"offset": 0, "profile": "none", "series": "KXGOLD15M", "settled": [], "fills": []})
    size = os.path.getsize(path)
    if size < c["offset"]:                                  # file was replaced: start over
        c.update(offset=0, profile="none", series="KXGOLD15M", settled=[], fills=[])
    with open(path, "rb") as fh:
        fh.seek(c["offset"])
        data = fh.read()
    whole = data.rfind(b"\n") + 1                           # never parse a half-written last line
    for line in data[:whole].splitlines():
        if not any(w in line for w in _WANTED):
            continue
        e = json.loads(line)
        if e["event"] == "start":
            c["profile"] = e.get("profile") or ("standard" if e.get("standard") else "custom")
            c["series"] = e.get("series", "KXGOLD15M")
        elif e["event"] == "settled":
            c["settled"].append(e)
        else:
            c["fills"].append(e)
    c["offset"] += whole
    return c


def build_payload(pattern: str = "logs/paper-*.jsonl", series: str = "KXGOLD15M") -> dict:
    """Everything the page needs: one series per (run profile, style), with every settled trade."""
    lines: dict[str, dict] = {}
    now = time.time()
    for path in sorted(glob.glob(pattern)):
        c = _read_new(path)
        if c["series"] != series:
            continue
        group = c["profile"]
        settled_ids = {e.get("order_id") for e in c["settled"]}
        for e in c["settled"] + c["fills"]:
            style = e.get("style", "taker")
            key = f"{group}_{style}"
            s = lines.setdefault(key, {
                "key": key, "group": group, "style": style, "label": f"{GROUP_NAMES.get(group, group)} · {style}",
                "slot": COLOR_SLOT.get((group, style), 0), "excluded": group in EXCLUDED_BY_DEFAULT,
                "points": [], "open": 0, "unrecorded": 0})
            if e["event"] == "settled":
                s["points"].append({"t": e["t"], "pnl": round(e["pnl"], 4), "window": e["ticker"][-9:],
                                    "result": e.get("result", ""), "count": e.get("count", 0)})
            elif e.get("order_id") not in settled_ids:      # filled, but no result recorded
                if e["t"] + e.get("secs_left", 0) + 120 > now:
                    s["open"] += 1                         # its window is still running (or just ended)
                else:
                    s["unrecorded"] += 1                   # that bot stopped first; `report` can look the result up
    for s in lines.values():
        s["points"].sort(key=lambda p: p["t"])
    order = sorted(lines.values(), key=lambda s: (s["slot"] == 0, s["slot"], s["key"]))
    return {"now": now, "series": order, "market": series}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kalshi-bot: paper P&amp;L</title>
<style>
:root { color-scheme: light dark; }
.viz-root {
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6b6a66;
  --grid: #ebeae6; --axis: #cfcec8; --ghost: #f0efec;
  --s0: #8a8985; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --grid: #2a2a28; --axis: #3d3d3a; --ghost: #262625;
    --s0: #77766f; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  }
}
body { margin: 0; background: var(--surface-1); }
.viz-root { background: var(--surface-1); color: var(--text-primary); min-height: 100vh; padding: 24px 28px 40px;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; box-sizing: border-box; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 2px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.filters { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.seg { display: inline-flex; border: 1px solid var(--axis); border-radius: 8px; overflow: hidden; }
.seg button { background: none; border: 0; padding: 5px 12px; color: var(--text-secondary); font: inherit; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ghost); color: var(--text-primary); font-weight: 600; }
.seg button:hover { background: var(--ghost); }
button:focus-visible, svg:focus-visible, input:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
label.chk { color: var(--text-secondary); display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 10px; margin-bottom: 10px; }
.legend button { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--axis); background: none;
  color: var(--text-primary); border-radius: 999px; padding: 3px 12px 3px 9px; font: inherit; cursor: pointer; }
.legend button[aria-pressed="false"] { opacity: .45; }
.key { width: 16px; height: 0; border-top: 2px solid; border-radius: 2px; display: inline-block; }
#plot { position: relative; transition: opacity .2s; }
#plot svg { display: block; width: 100%; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--axis); border-radius: 8px;
  padding: 8px 10px; box-shadow: 0 4px 16px rgba(0,0,0,.14); min-width: 190px; display: none; z-index: 2; }
.tip .h { color: var(--text-secondary); margin-bottom: 4px; }
.tip .r { display: flex; align-items: center; gap: 8px; justify-content: space-between; }
.tip .r .n { display: inline-flex; align-items: center; gap: 8px; color: var(--text-secondary); }
.tip .r .v { font-weight: 600; color: var(--text-primary); font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; margin-top: 18px; }
th { text-align: right; color: var(--text-secondary); font-weight: 500; padding: 6px 10px; border-bottom: 1px solid var(--axis); }
td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
.pos { color: var(--text-primary); } .empty { color: var(--text-secondary); padding: 40px 0; text-align: center; }
.note { color: var(--text-muted); margin-top: 14px; font-size: 12px; max-width: 80ch; }
.axis text { fill: var(--text-secondary); font-size: 12px; }
</style></head>
<body><div class="viz-root">
  <h1 id="title">Paper trading: net P&amp;L over time</h1>
  <p class="sub" id="sub">loading…</p>
  <div class="filters">
    <div class="seg" id="range" role="group" aria-label="Time range">
      <button data-h="0" aria-pressed="true">All</button><button data-h="24" aria-pressed="false">24 h</button>
      <button data-h="6" aria-pressed="false">6 h</button><button data-h="1" aria-pressed="false">1 h</button></div>
    <div class="seg" id="view" role="group" aria-label="View">
      <button data-v="chart" aria-pressed="true">Chart</button><button data-v="table" aria-pressed="false">Table</button></div>
    <label class="chk"><input type="checkbox" id="excl"> include custom &amp; older runs (not valid as evidence)</label>
  </div>
  <div class="legend" id="legend"></div>
  <div id="plot"></div>
  <div id="tablewrap"></div>
  <p class="note">Lines show cumulative net P&amp;L after fees from <b>settled</b> trades only (early exits count when sold).
  Open positions are listed in the table; \"result not recorded\" are fills from runs that stopped before their window settled (run `kalshi-bot report` to look those up). For a selected time range, each line starts at $0 at the left edge.
  Colour is fixed per run type. Hover or use ← → on the chart to read exact values.</p>
</div>
<script>
const SLOT_VAR = ['--s0','--s1','--s2','--s3','--s4','--s5','--s6'];
const colorOf = s => `var(${SLOT_VAR[s.slot] || '--s0'})`;
const fmtMoney = (v, sign) => (v < 0 ? '-' : sign && v > 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2);

/* ---- pure helpers (also exercised by tests/node) ---- */
function niceTicks(lo, hi, want) {                                     // round numbers: 1, 2, 5 x 10^k
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const base = Math.pow(10, Math.floor(Math.log10((hi - lo) / want)) - 1);
  const step = [1, 2, 5, 10, 20, 50, 100].map(m => m * base).find(s => (hi - lo) / s <= want + 1) || 100 * base;
  const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
const TIME_STEPS = [300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800];
function timeTicks(t0, t1, max) {
  const step = TIME_STEPS.find(s => (t1 - t0) / s <= max) || 604800, out = [];
  const off = new Date(t0 * 1000).getTimezoneOffset() * 60;           // align ticks to local time
  for (let t = Math.ceil((t0 - off) / step) * step + off; t <= t1; t += step) out.push(t);
  return out;
}
function cumulative(points, start) {                                   // [{t,c}] re-based to 0 at `start`
  let c = 0; const out = [];
  for (const p of points) if (p.t >= start) { c += p.pnl; out.push({ t: p.t, c: +c.toFixed(4), p }); }
  return out;
}
function valueAt(cum, t) { let v = 0; for (const q of cum) { if (q.t <= t) v = q.c; else break; } return v; }
function stepPath(cum, x0, x1, X, Y) {                                 // horizontal-then-vertical steps
  let d = `M${X(x0)},${Y(0)}`, prev = 0;
  for (const q of cum) { d += `L${X(q.t)},${Y(prev)}L${X(q.t)},${Y(q.c)}`; prev = q.c; }
  return d + `L${X(x1)},${Y(prev)}`;
}

/* ---- state ---- */
let DATA = null, rangeH = 0, view = 'chart', showExcl = false, hidden = new Set(), cursor = null;
const $ = id => document.getElementById(id);

function visibleSeries() {
  return DATA.series.filter(s => (showExcl || !s.excluded) && !hidden.has(s.key));
}
function rangeStart() {
  if (rangeH) return DATA.now - rangeH * 3600;
  const all = DATA.series.filter(s => showExcl || !s.excluded).flatMap(s => s.points.map(p => p.t));
  return all.length ? Math.min(...all) - 120 : DATA.now - 3600;
}

function renderLegend() {
  const el = $('legend'); el.replaceChildren();
  const shown = DATA.series.filter(s => showExcl || !s.excluded);
  for (const s of shown) {
    const b = document.createElement('button'); b.setAttribute('aria-pressed', String(!hidden.has(s.key)));
    const k = document.createElement('span'); k.className = 'key'; k.style.borderTopColor = colorOf(s);
    const t = document.createElement('span'); t.textContent = s.label;
    b.append(k, t); b.onclick = () => { hidden.has(s.key) ? hidden.delete(s.key) : hidden.add(s.key); render(); };
    el.append(b);
  }
}

function renderChart() {
  const host = $('plot'); host.replaceChildren();
  const vis = visibleSeries(), start = rangeStart(), x1 = DATA.now;
  const cums = vis.map(s => ({ s, cum: cumulative(s.points, start) }));
  if (!cums.some(c => c.cum.length)) { host.innerHTML = '<div class="empty">No settled trades in the selected runs and range yet.</div>'; return; }
  const W = Math.max(520, host.clientWidth || 900), H = 380, m = { l: 62, r: 78, t: 12, b: 30 };
  const allC = cums.flatMap(c => c.cum.map(q => q.c)).concat([0]);
  let lo = Math.min(...allC), hi = Math.max(...allC); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const X = t => m.l + (t - start) / (x1 - start) * (W - m.l - m.r), Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  const NS = 'http://www.w3.org/2000/svg', el = (n, a, txt) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; return e; };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, tabindex: 0, role: 'img',
    'aria-label': 'Cumulative net profit and loss over time for each run type. Use the Table view for exact values.' });
  const ax = el('g', { class: 'axis' }); svg.append(ax);
  for (const v of niceTicks(lo, hi, 5)) {                               // horizontal grid + y labels
    ax.append(el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: v === 0 ? 'var(--axis)' : 'var(--grid)', 'stroke-width': 1 }));
    ax.append(el('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, fmtMoney(v)));
  }
  const spanDays = (x1 - start) > 86400;
  for (const t of timeTicks(start, x1, 6)) {
    const d = new Date(t * 1000), hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    ax.append(el('text', { x: X(t), y: H - 8, 'text-anchor': 'middle' }, spanDays ? d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + hm : hm));
  }
  for (const { s, cum } of cums) {                                      // lines (2px) + end dots with surface ring
    svg.append(el('path', { d: stepPath(cum, start, x1, X, Y), fill: 'none', stroke: colorOf(s),
      'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    if (cum.length) { const q = cum[cum.length - 1];
      svg.append(el('circle', { cx: X(q.t), cy: Y(q.c), r: 4.5, fill: colorOf(s), stroke: 'var(--surface-1)', 'stroke-width': 2 })); }
  }
  const ends = cums.filter(c => c.cum.length).map(c => ({ c, y: Y(c.cum[c.cum.length - 1].c), v: c.cum[c.cum.length - 1].c }))
    .sort((a, b) => a.y - b.y);                                         // direct end labels, only if they don't collide
  if (ends.length <= 4) { let lastY = -99;                              // label each end that has room; collisions fall back to legend + tooltip
    for (const e of ends) if (e.y - lastY >= 15) { lastY = e.y;
      svg.append(el('text', { x: W - m.r + 8, y: e.y + 4, fill: 'var(--text-secondary)', style: 'font-size:12px;font-weight:600' }, fmtMoney(e.v, true))); } }
  const guide = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--axis)', 'stroke-width': 1, visibility: 'hidden' }); svg.append(guide);
  const dots = cums.map(({ s }) => { const d = el('circle', { r: 4.5, fill: colorOf(s), stroke: 'var(--surface-1)', 'stroke-width': 2, visibility: 'hidden' }); svg.append(d); return d; });
  const times = [...new Set(cums.flatMap(c => c.cum.map(q => q.t)))].sort((a, b) => a - b);
  const tip = document.createElement('div'); tip.className = 'tip';
  function show(t) {                                                    // crosshair snaps to the nearest settlement time
    if (!times.length) return; cursor = t;
    guide.setAttribute('x1', X(t)); guide.setAttribute('x2', X(t)); guide.setAttribute('visibility', 'visible');
    tip.replaceChildren();
    const h = document.createElement('div'); h.className = 'h'; h.textContent = new Date(t * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    tip.append(h);
    cums.forEach(({ s, cum }, i) => {
      const v = valueAt(cum, t); dots[i].setAttribute('cx', X(t)); dots[i].setAttribute('cy', Y(v)); dots[i].setAttribute('visibility', 'visible');
      const r = document.createElement('div'); r.className = 'r';
      const n = document.createElement('span'); n.className = 'n';
      const k = document.createElement('span'); k.className = 'key'; k.style.borderTopColor = colorOf(s);
      const nm = document.createElement('span'); nm.textContent = s.label; n.append(k, nm);
      const val = document.createElement('span'); val.className = 'v'; val.textContent = fmtMoney(v, true);
      r.append(n, val); tip.append(r);
    });
    tip.style.display = 'block';
    const px = X(t), w = tip.offsetWidth; tip.style.left = Math.min(Math.max(px + 14, 0), host.clientWidth - w - 4) + 'px'; tip.style.top = '8px';
  }
  function hide() { guide.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); tip.style.display = 'none'; }
  const nearest = px => { const t = start + (px - m.l) / (W - m.l - m.r) * (x1 - start);
    return times.reduce((b, c) => Math.abs(c - t) < Math.abs(b - t) ? c : b, times[0]); };
  svg.addEventListener('pointermove', e => { const r = svg.getBoundingClientRect(); show(nearest((e.clientX - r.left) * W / r.width)); });
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('keydown', e => { if (!times.length) return;
    const i = cursor == null ? times.length - 1 : times.indexOf(cursor);
    if (e.key === 'ArrowLeft') { show(times[Math.max(0, i - 1)]); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { show(times[Math.min(times.length - 1, i + 1)]); e.preventDefault(); }
    else if (e.key === 'Escape') hide(); });
  svg.addEventListener('blur', hide);
  host.append(svg, tip);
}

function cell(tr, text, tag) { const c = document.createElement(tag || 'td'); c.textContent = text; tr.append(c); return c; }
function renderTable() {
  const wrap = $('tablewrap'); wrap.replaceChildren();
  const vis = DATA.series.filter(s => showExcl || !s.excluded), start = rangeStart();
  const tbl = document.createElement('table'), head = document.createElement('tr');
  ['Run type', 'Settled', 'Wins', 'Win rate', 'Net P&L', 'Open now', 'Result not recorded'].forEach(h => cell(head, h, 'th')); tbl.append(head);
  for (const s of vis) {
    const pts = s.points.filter(p => p.t >= start), wins = pts.filter(p => p.pnl > 0).length;
    const tr = document.createElement('tr'), c0 = cell(tr, '');
    const k = document.createElement('span'); k.className = 'key'; k.style.borderTopColor = colorOf(s); k.style.marginRight = '8px';
    c0.append(k, document.createTextNode(s.label));
    cell(tr, pts.length); cell(tr, wins); cell(tr, pts.length ? (100 * wins / pts.length).toFixed(0) + '%' : '-');
    cell(tr, fmtMoney(pts.reduce((a, p) => a + p.pnl, 0), true)); cell(tr, s.open); cell(tr, s.unrecorded); tbl.append(tr);
  }
  wrap.append(tbl);
  if (view !== 'table') return;
  const rows = vis.flatMap(s => cumulative(s.points, start).map(q => ({ s, q }))).sort((a, b) => b.q.t - a.q.t).slice(0, 300);
  const t2 = document.createElement('table'), h2 = document.createElement('tr');
  ['Settled at', 'Run type', 'Window', 'Result', 'Trade P&L', 'Cumulative'].forEach(h => cell(h2, h, 'th')); t2.append(h2);
  for (const { s, q } of rows) { const tr = document.createElement('tr');
    cell(tr, new Date(q.t * 1000).toLocaleString()); cell(tr, s.label); cell(tr, q.p.window);
    cell(tr, (q.p.result || '').toUpperCase()); cell(tr, fmtMoney(q.p.pnl, true)); cell(tr, fmtMoney(q.c, true)); t2.append(tr); }
  wrap.append(t2);
}

function render() {
  if (!DATA) return;
  const start = rangeStart(), age = Math.max(0, Math.round(Date.now() / 1000 - DATA.now));
  $('title').textContent = `Paper trading (${DATA.market || 'KXGOLD15M'}): net P&L over time`;
  $('sub').textContent = `Cumulative, after fees, settled trades only · since ${new Date(start * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} · updated ${age}s ago`;
  renderLegend();
  $('plot').style.display = view === 'chart' ? '' : 'none';
  if (view === 'chart') renderChart();
  renderTable();
}
for (const b of document.querySelectorAll('#range button')) b.onclick = () => { rangeH = +b.dataset.h; document.querySelectorAll('#range button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); render(); };
for (const b of document.querySelectorAll('#view button')) b.onclick = () => { view = b.dataset.v; document.querySelectorAll('#view button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); render(); };
$('excl').onchange = e => { showExcl = e.target.checked; render(); };
window.addEventListener('resize', () => render());
async function load() {                                                  // refetch keeps the frame: dim, don't blank
  $('plot').style.opacity = .6;
  try { DATA = await (await fetch('/data.json')).json(); } catch (e) { $('sub').textContent = 'cannot reach the chart server: ' + e; }
  $('plot').style.opacity = 1; render();
}
load(); setInterval(load, 5000);
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    pattern = "logs/paper-*.jsonl"
    series = "KXGOLD15M"

    def do_GET(self):
        if self.path.split("?")[0] == "/data.json":
            body, kind = json.dumps(build_payload(self.pattern, self.series)).encode(), "application/json"
        elif self.path.split("?")[0] == "/":
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):          # keep the terminal quiet
        pass


def serve(pattern: str = "logs/paper-*.jsonl", port: int = 8765, open_browser: bool = True,
          series: str = "KXGOLD15M") -> None:
    _Handler.pattern = pattern
    _Handler.series = series
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)     # local only: never exposed to the network
    url = f"http://127.0.0.1:{port}"
    print(f"Paper P&L chart: {url}   (Ctrl-C to stop; the page refreshes itself every 5 seconds)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
