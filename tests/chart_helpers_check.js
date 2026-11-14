const assert = require('assert');
// y ticks: round numbers that include zero, 4-8 of them
let t = niceTicks(-3.2, 7.9, 5); assert.deepStrictEqual(t, [-2, 0, 2, 4, 6]); 
t = niceTicks(0, 0, 5); assert.ok(t.length >= 2, 'flat range still produces ticks');
t = niceTicks(-120, 340, 5); assert.deepStrictEqual(t, [-100, 0, 100, 200, 300]);
// time ticks: at most max ticks, inside the range, evenly spaced
const t0 = 1_800_000_000, t1 = t0 + 6 * 3600; const tt = timeTicks(t0, t1, 6);
assert.ok(tt.length >= 3 && tt.length <= 7 && tt.every(x => x >= t0 && x <= t1)); 
assert.ok(tt.every((x, i) => i === 0 || x - tt[i - 1] === tt[1] - tt[0]));
// cumulative: re-based at the range start
const pts = [{t: 10, pnl: 2}, {t: 20, pnl: -5}, {t: 30, pnl: 1.5}];
let c = cumulative(pts, 0); assert.deepStrictEqual(c.map(q => q.c), [2, -3, -1.5]);
c = cumulative(pts, 15); assert.deepStrictEqual(c.map(q => q.c), [-5, -3.5]);          // starts again from zero
assert.strictEqual(valueAt(cumulative(pts, 0), 25), -3); assert.strictEqual(valueAt(cumulative(pts, 0), 5), 0);
// step path: starts at (x0, 0), every move is horizontal then vertical, ends at x1 with the last value
const X = v => v, Y = v => -v; const d = stepPath(cumulative(pts, 0), 0, 40, X, Y);
assert.strictEqual(d, 'M0,0L10,0L10,-2L20,-2L20,3L30,3L30,1.5L40,1.5');
console.log('all JS helper checks passed');
