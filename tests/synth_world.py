"""Minimal synthetic Kalshi-like world with controlled truth (adapted from the harness audit).
lam: market under-reaction to the last-L-minute reference move, in sigma units (a FEATURE-borne edge).
lam=0 is an efficient (null) market."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def gen(n, lam=0.0, L=1, h=0.4, seed=0, step=900):
    rng = np.random.default_rng(seed)
    pad = 40
    a = np.zeros(n)
    a[0] = rng.normal(0, .5)
    eps = rng.normal(0, .156, n)
    for i in range(1, n):
        a[i] = .95 * a[i - 1] + eps[i]
    sig = np.exp(a)
    inc = rng.normal(0, 1, pad + n * 15)
    sig_min = np.concatenate([np.full(pad, sig[0]), np.repeat(sig, 15)])
    R = np.concatenate([[0], np.cumsum(inc * sig_min)])
    vol30 = pd.Series(R).diff().rolling(30).std().to_numpy()
    H = np.cumsum(rng.normal(0, 1, (n, 15)) * sig[:, None] * h, axis=1)
    i0 = pad + 15 * np.arange(n)
    ks = np.arange(1, 15)
    idx = i0[:, None] + ks[None, :]
    x = R[idx] - R[i0][:, None]
    D = x + H[:, :14]
    yes = ((R[i0 + 15] - R[i0]) + H[:, 14] >= 0).astype(int)
    tau = (15 - ks)[None, :] * np.ones((n, 1))
    sS = (sig * np.sqrt(1 + h * h))[:, None]
    mm = np.clip(norm.cdf((D - lam * (R[idx] - R[idx - L])) / (sS * np.sqrt(tau))), 0.004, 0.996)
    sp = rng.choice([0.01, 0.02, 0.03], size=mm.shape, p=[0.5, 0.35, 0.15])
    bid = np.round(mm - sp / 2, 3)
    ask = np.round(bid + sp, 3)
    ok = (bid > 0) & (ask < 1)
    w = np.repeat(np.arange(n), 14).reshape(n, 14)
    open_ts = (1.8e9 + w * step).astype(np.int64)
    df = pd.DataFrame(dict(ticker=np.char.add("W", w.ravel().astype(str)), open_ts=open_ts.ravel(),
                           end_ts=(open_ts + 60 * ks[None, :]).ravel(), bid=bid.ravel(), ask=ask.ravel(),
                           x=x.ravel(), vol30=vol30[idx].ravel(), m3=(R[idx] - R[idx - 3]).ravel(),
                           m5=(R[idx] - R[idx - 5]).ravel(), tau=tau.ravel(), yes=np.repeat(yes, 14)))
    df["mid"] = (df.bid + df.ask) / 2
    return df[ok.ravel()].sort_values(["open_ts", "end_ts"]).reset_index(drop=True)
