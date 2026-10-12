#!/usr/bin/env python
"""Single heads reading one small table (the stage figure of the paper): the full attention matrix of the
chosen heads on the 5 x 5 orders table, target cell (4,4), under Markdown and CCS, with every token labelled
by its cell and kind so that any delimiter or value can serve as the query token in the figure.

    python mechanistic/stages.py                                   # the four heads of the paper's figure
    python mechanistic/stages.py --heads L1H31 L9H6 L18H3 L29H11

Writes results/mechanistic/heads/stages.npz ({fmt}_L{l}H{h}: T x T, fp16) and stages.json (texts, offsets,
token labels, predicted first token).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as sz  # noqa: E402
from common import ORDERS, OUT, chat, load_hf_model, token_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--heads", nargs="+", default=["L1H31", "L9H6", "L18H3", "L29H11"])
ap.add_argument("--target", type=int, nargs=2, default=[4, 4])
args = ap.parse_args()
HEADS = [tuple(int(x) for x in re.match(r"L(\d+)H(\d+)", h).groups()) for h in args.heads]
TR, TC = args.target
OUT_DIR = OUT / "heads"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_lines(rows, fmt):
    """The table lines of a serialization and, per character, (row, column, kind); the same text as
    synthetic.table_text / the CCS transform."""
    out, lab = [], []

    def add(s, r, c, k):
        if s:
            out.append(s)
            lab.extend([(r, c, k)] * len(s))

    def prefix(i):
        add("row ", i, 0, "pre")
        add(str(i), i, 0, "pre_dig")
        add(" (header): " if i == 1 else ": ", i, 0, "pre")

    R, C = len(rows), len(rows[0])
    if fmt == "markdown":
        w = [max(len(r[j]) for r in rows) for j in range(C)]
        for i, r in enumerate(rows, 1):
            for j, v in enumerate(r, 1):
                add("|", i, j, "open")
                add(" ", i, j, "pad")
                add(v, i, j, "value")
                add(" " * (w[j - 1] - len(v)) + " ", i, j, "pad")
            add("|", i, 0, "eol")
            if i == 1:
                add("\n", 0, 0, "nl")
                add("| " + " | ".join("-" * x for x in w) + " |", 0, 0, "sep")
            if i < R:
                add("\n", i, 0, "nl")
        return "".join(out), lab
    for i, r in enumerate(rows, 1):
        prefix(i)
        for j, v in enumerate(r, 1):
            if j > 1:
                add(" ", i, j, "sp")
            for s, k in [("[", "open"), ("r", "r"), (str(i), "rdig"), ("|", "bar"), ("c", "c"), (str(j), "cdig"), ("=", "eq"), (v, "value"), ("]", "close")]:
                add(s, i, j, k)
        if i < R:
            add("\n", i, 0, "nl")
    return "".join(out), lab


def label_tokens(text, offs, rows, fmt):
    """Per token: row, column, kind ('' outside the table), and whether a value token is the last of its value."""
    t, lab = build_lines(rows, fmt)
    cs = text.rindex(t)
    T = len(offs)
    row, col, kind = np.zeros(T, int), np.zeros(T, int), [""] * T
    for i, (a, b) in enumerate(offs):
        if b <= cs or a >= cs + len(t) or b <= a:
            continue
        ch = [(lab[x - cs], text[x]) for x in range(max(a, cs), min(b, cs + len(t)))]
        v = [l for l, c in ch if l[2] == "value"]
        ns = [l for l, c in ch if not c.isspace()]
        l = v[0] if v else (ns[0] if ns else ch[0][0])
        row[i], col[i], kind[i] = l
    last = np.zeros(T, bool)
    for i in range(T):
        if kind[i] == "value" and not (i + 1 < T and kind[i + 1] == "value" and row[i + 1] == row[i] and col[i + 1] == col[i]):
            last[i] = True
    return row, col, np.array(kind), last


tok, model = load_hf_model()
ATT = {}


def ahook(L):
    def f(mod, inp, out):
        for (l, h) in HEADS:
            if l == L:
                ATT[(l, h)] = out[1][0, h].float().cpu().numpy().astype(np.float16)
    return f


for L in sorted(set(l for l, h in HEADS)):
    model.model.layers[L].self_attn.register_forward_hook(ahook(L))
meta, arrs = {}, {}
for fmt in ["markdown", "tagged"]:
    p, f = sz.build(ORDERS, TR, TC, fmt)
    text = chat(tok, p) + f
    ids, offs = token_ids(tok, text)
    row, col, kind, last = label_tokens(text, offs, ORDERS, fmt)
    with torch.no_grad():
        o = model(torch.tensor([ids], device="cuda"))
    for lh in HEADS:
        arrs[f"{fmt}_L{lh[0]}H{lh[1]}"] = ATT[lh]
    meta[fmt] = {"text": text, "offs": offs, "rows": ORDERS, "tr": TR, "tc": TC, "row": row.tolist(), "col": col.tolist(),
                 "kind": kind.tolist(), "last": last.tolist(), "pred": tok.decode([int(o.logits[0, -1].argmax())])}
    print(fmt, len(ids), "tokens, predicted first token", repr(meta[fmt]["pred"]), flush=True)
np.savez_compressed(OUT_DIR / "stages.npz", **arrs)
json.dump(meta, open(OUT_DIR / "stages.json", "w"))
print("done")
