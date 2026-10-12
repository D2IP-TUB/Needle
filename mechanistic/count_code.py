#!/usr/bin/env python
"""Where the count is born: the shape of the count code along the layers (the 1/j paragraph of the paper).

On 16 Markdown tables of 8 x 64 (cell-locating query), the residual stream at the pipe before every body cell
is collected at several layers. Per layer, the count direction is the first principal direction of the
per-column means; the per-column mean projected on it, m(j), is fitted with a + b/j (a head that averages
uniformly over the j delimiters seen so far writes exactly this), with a + b log j, and with a hinge
a + b min(j, k). The log-log slope of the steps m(j+1) - m(j) is -2 for the 1/j code.

    python mechanistic/count_code.py           # -> results/mechanistic/heads/count_code.json
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as sz  # noqa: E402
from common import OUT, chat, load_hf_model, synth_rows, toks_in, token_ids, units  # noqa: E402

OUT_DIR = OUT / "heads"
OUT_DIR.mkdir(parents=True, exist_ok=True)
tok, model = load_hf_model()
LAYERS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 24, 30]
R, C, N = 8, 64, 16
rng = random.Random(83)
H = {L: [[] for _ in range(C)] for L in LAYERS}
for n in range(N):
    rows = synth_rows(rng, R, C)
    p, f = sz.build(rows, 2, 2, "markdown")
    text = chat(tok, p) + f
    ids, offs = token_ids(tok, text)
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
    for row, lab, pos, sp in units(text, "markdown"):
        if row == 1:
            continue
        td = toks_in(offs, sp["delim"])[-1]
        for L in LAYERS:
            H[L][pos - 1].append(out.hidden_states[L][0, td].float().cpu().numpy())
    print(n, flush=True)

j = np.arange(1, C + 1, dtype=float)
inv = lambda j, a, b: a + b / j  # noqa: E731
lg = lambda j, a, b: a + b * np.log(j)  # noqa: E731
hinge = lambda j, a, b, k: a + b * np.minimum(j, k)  # noqa: E731


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


curves, noise, stats = {}, {}, {}
for L in LAYERS:
    mu = np.stack([np.mean(H[L][k], 0) for k in range(C)])
    M = mu - mu.mean(0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    d = Vt[0]
    m = M @ d
    m = m if np.corrcoef(m, j)[0, 1] > 0 else -m
    sd = np.array([np.std(np.stack(H[L][k]) @ d) for k in range(C)])     # per-column spread of the projection
    curves[L], noise[L] = m, sd
    pi, _ = curve_fit(inv, j, m, p0=[m.max(), -(m.max() - m.min())])
    pl, _ = curve_fit(lg, j, m, p0=[m.min(), 1])
    try:
        ph, _ = curve_fit(hinge, j, m, p0=[m.min(), (m.max() - m.min()) / 16, 16], maxfev=20000)
    except Exception:
        ph = [np.nan] * 3
    st = np.diff(m)
    ok = (st > 0) & (j[1:] <= 32)
    sl = np.polyfit(np.log(j[1:][ok]), np.log(st[ok]), 1)[0] if ok.sum() > 4 else np.nan
    stats[L] = {"r2_inv": r2(m, inv(j, *pi)), "r2_log": r2(m, lg(j, *pl)),
                "r2_hinge": r2(m, hinge(j, *ph)) if not np.isnan(ph[0]) else np.nan, "knee": float(ph[2]), "step_slope": float(sl)}
    print(f"L{L}: a+b/j R2 {stats[L]['r2_inv']:.2f}  a+b log j R2 {stats[L]['r2_log']:.2f}  hinge R2 {stats[L]['r2_hinge']:.2f} "
          f"(knee {stats[L]['knee']:.1f})  step slope {stats[L]['step_slope']:.2f}", flush=True)
json.dump({"stats": {str(k): v for k, v in stats.items()}, "curves": {str(k): v.tolist() for k, v in curves.items()},
           "noise": {str(k): v.tolist() for k, v in noise.items()}}, open(OUT_DIR / "count_code.json", "w"))
print("done")
