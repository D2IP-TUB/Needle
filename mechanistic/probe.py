#!/usr/bin/env python
"""Linear probes for the running row and column count (the probe figure and the probe numbers of the paper).

A ridge regression from the layer-17 residual stream at the token that opens each cell (the pipe before a
Markdown cell; with ``--tagged`` the '[' of a CCS cell) to the cell's absolute row and column index. A probe
for size N is fitted on 400 tables whose counted axis is drawn from 4 to N (the other axis from 4 to 16) and
read out on twelve held-out tables of N columns (8 rows) and N rows (8 columns); the 16 x 16 probe is fitted
on 1,000 tables of 4-16 x 4-16.

    python mechanistic/probe.py --sizes 64 128          # as in the paper: p16 + probes for 64 and 128
    python mechanistic/probe.py --sizes 32              # the 32-column / 32-row panels
    python mechanistic/probe.py --sizes 64 128 --tagged # the same probes on CCS tables

Writes results/mechanistic/probe/probe_<R>x<C>[_tagged].npy (per cell: true index and read-outs) and merges
the summary (mean absolute error, share within half a cell) into probe/probe_sizes[_tagged].json.
"""

import argparse
import bisect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as sz  # noqa: E402
from common import OUT, chat, synth_rows, token_ids, units  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
ap.add_argument("--tagged", action="store_true", help="read the probe at the '[' of each CCS cell")
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--no-p16", action="store_true", help="skip the 16 x 16 probe (e.g. for the 32 run)")
args = ap.parse_args()
FMT = "tagged" if args.tagged else "markdown"
SUF = "_tagged" if args.tagged else ""
OUT_DIR = OUT / "probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
tok, model = sz.load()
D = model.config.hidden_size
L = args.layer
LAMBDA = 1.0


@torch.no_grad()
def feats(rows, tr, tc):
    p, f = sz.build(rows, tr, tc, FMT)
    text = chat(tok, p) + f
    ids, offs = token_ids(tok, text)
    H = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states[L][0].float()
    starts = [x for x, y in offs]
    rc, ix = [], []
    for row, lab, pos, sp in units(text, FMT):
        a = sp["delim"][0] if FMT == "markdown" else sp["tag"][0] - 1     # the pipe / the opening bracket
        ix.append(bisect.bisect_right(starts, a) - 1)
        rc.append((row, lab))
    return np.array(rc), torch.cat([H[ix], torch.ones(len(ix), 1, device="cuda")], 1)


def fit(n, rmax, cmax):
    XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64, device="cuda")
    Xty = torch.zeros(D + 1, 2, dtype=torch.float64, device="cuda")
    rng = random.Random(1)
    for _ in range(n):
        R, C = rng.randint(4, rmax), rng.randint(4, cmax)
        rc, X = feats(synth_rows(rng, R, C), rng.randint(1, R), rng.randint(1, C))
        X = X.double()
        XtX += X.T @ X
        Xty += X.T @ torch.tensor(rc, dtype=torch.float64, device="cuda")
    I = torch.eye(D + 1, dtype=torch.float64, device="cuda") * LAMBDA
    I[D, D] = 0
    return torch.linalg.solve(XtX + I, Xty).float()   # (D+1) x 2: row, column


W, PLAN = {}, {}
sizes = [n for n in args.sizes if n != 16]
if not args.no_p16:   # the 16 x 16 probe, read on 16 x 16, 8 x 16 and 16 x 8 tables
    W["p16"] = fit(1000, 16, 16)
    print("probe p16 fitted", flush=True)
    PLAN.update({(16, 16): ["p16"], (8, 16): ["p16"], (16, 8): ["p16"]})
for N in sizes:
    W[f"col{N}"] = fit(400, 16, N)
    W[f"row{N}"] = fit(400, N, 16)
    print("probes", N, "fitted", flush=True)
    PLAN.setdefault((8, N), []).append(f"col{N}")
    PLAN.setdefault((N, 8), []).append(f"row{N}")
    if (16, 16) in PLAN:
        PLAN[(16, 16)] += [f"col{N}", f"row{N}"]
res_path = OUT_DIR / f"probe_sizes{SUF}.json"
res = json.load(open(res_path)) if res_path.exists() else {}
for (R, C), names in PLAN.items():
    per = []
    for k in range(12):
        rc, X = feats(synth_rows(random.Random(2024 + k), R, C), 2, 2)
        per.append((rc, {m: (X @ W[m]).cpu().numpy() for m in names}))
    np.save(OUT_DIR / f"probe_{R}x{C}{SUF}.npy", {"rc": per[0][0], **{m: per[0][1][m] for m in names}}, allow_pickle=True)
    for m in names:
        for ax, nm in ((0, "row"), (1, "col")):
            if (m.startswith("col") and ax == 0) or (m.startswith("row") and ax == 1):
                continue
            g = np.concatenate([rc[:, ax] for rc, _ in per])
            e = np.abs(np.concatenate([P[m][:, ax] for _, P in per]) - g)
            res[f"{R}x{C}_{m}_{nm}"] = {"mae": float(e.mean()), "half": float((e < 0.5).mean())}
            bands = [(1, 8), (9, 16), (17, 32), (33, 64), (65, 128)]
            bd = "  ".join(f"{a}-{b}: {e[(g >= a) & (g <= b)].mean():.2f}/{(e[(g >= a) & (g <= b)] < 0.5).mean():.2f}"
                           for a, b in bands if ((g >= a) & (g <= b)).any())
            print(f"{R}x{C} {nm} index, probe {m:<7}: mean |error| {e.mean():.2f}  within half a cell {(e < 0.5).mean():.2f}"
                  f"   by index band (error/within half): {bd}", flush=True)
json.dump(res, open(res_path, "w"), indent=1)
print("done")
