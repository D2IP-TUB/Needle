#!/usr/bin/env python
"""Attention roles of every head (36 layers x 32 heads) under Markdown, JSON and CCS, on 16 synthetic tables
of 8 x 32 (seed 51, cell-locating query), and the head populations derived from them.

Roles (share of a query token's attention on a set of key tokens, averaged over cells and tables):
  count             the token that opens a cell -> the openers of the earlier cells of its line
                    (Markdown: the pipe; JSON: the quote that opens the key; CCS: the '[')
  bind_own          the last token of a value -> its own coordinates (CCS: the r..|c.. part of its tag; JSON: its key)
  bind_col_header   the last token of a value (rows >= 2) -> the header cell of its column
  bind_line_start   the last token of a value (rows >= 2) -> the first token of its line
  ans_value         the position that generates the answer -> the target value

Populations (``populations.json``): counting heads = Markdown ``count`` > 0.5; coordinate heads = CCS
``bind_own`` > 0.2 (the wider set > 0.15); column heads = the eight strongest Markdown ``bind_col_header``;
mover heads = the eight (four) strongest CCS ``ans_value``.

    python mechanistic/head_roles.py                 # -> results/mechanistic/heads/roles.json, populations.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as sz  # noqa: E402
from common import OUT, chat, json_prompt, json_units, load_hf_model, opener, seeded_tables, toks_in, token_ids, units  # noqa: E402

OUT_DIR = OUT / "heads"
OUT_DIR.mkdir(parents=True, exist_ok=True)
tok, model = load_hf_model()
NL, NH = model.config.num_hidden_layers, model.config.num_attention_heads
G, acc = {}, {}
ROLES = ["count", "bind_own", "bind_col_header", "bind_line_start"]


def mean_mass(A, pairs):
    vals = [A[:, q, ks].sum(1) for q, ks in pairs if ks]
    return torch.stack(vals).mean(0) if vals else torch.zeros(NH, device=A.device)


def hook(li):
    def f(mod, inp, out):
        A = out[1][0].float()
        for k in ROLES:
            if G.get(k):
                acc.setdefault(li, {}).setdefault(k, []).append(mean_mass(A, G[k]).cpu().numpy())
        acc.setdefault(li, {}).setdefault("ans_value", []).append(A[:, -1, G["ans_value"]].sum(1).cpu().numpy())
    return f


for li, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(hook(li))


def groups_table(text, offs, fmt, tr, tc):
    """The query -> keys pairs of every role for a Markdown or CCS prompt."""
    U = units(text, fmt)
    by_row = {}
    for row, lab, pos, sp in U:
        by_row.setdefault(row, []).append((lab, sp))
    blk = [m for m in re.finditer(r"(?:^\|.*\|\n?)+" if fmt == "markdown" else r"(?:^row \d+.*\n?)+", text, re.M)][-1]
    line_first, key = {}, 0
    for m in re.finditer(r"[^\n]+", blk.group(0)):
        if fmt == "markdown" and re.match(r"^\|?\s*:?-{2,}", m.group(0)):
            continue
        key = int(re.match(r"row (\d+)", m.group(0)).group(1)) if fmt == "tagged" else key + 1
        line_first[key] = toks_in(offs, (blk.start() + m.start(), blk.start() + m.start() + 1))
    header_val = {lab: toks_in(offs, sp["value"]) for lab, sp in by_row[1]}
    g = {k: [] for k in ROLES}
    for row, us in by_row.items():
        ds = [opener(offs, sp, fmt) for lab, sp in us]
        for i, (lab, sp) in enumerate(us):
            g["count"].append((ds[i], ds[:i]))
            tv = toks_in(offs, sp["value"])
            if not tv:
                continue
            if row > 1:
                g["bind_col_header"].append((tv[-1], header_val.get(lab, [])))
                g["bind_line_start"].append((tv[-1], line_first.get(row, [])))
            if "tag" in sp:
                g["bind_own"].append((tv[-1], toks_in(offs, sp["tag"])))
    tgt = [sp for row, lab, pos, sp in U if (row, lab) == (tr, tc)][0]
    g["ans_value"] = toks_in(offs, tgt["value"])
    return g


def groups_json(text, offs, R, C, tr, tc):
    U, line_first, _ = json_units(text)
    assert len(U) == (R - 1) * C
    T = {k: {n: toks_in(offs, s) for n, s in u.items()} for k, u in U.items()}
    g = {k: [] for k in ROLES}
    for row in range(2, R + 1):
        ds = [T[(row, c)]["open"][-1] for c in range(1, C + 1)]
        for c in range(1, C + 1):
            u = T[(row, c)]
            g["count"].append((ds[c - 1], ds[:c - 1]))
            if not u["value"]:
                continue
            q = u["value"][-1]
            g["bind_own"].append((q, u["key"]))
            g["bind_line_start"].append((q, toks_in(offs, line_first[row])))
            if row >= 3:   # the header cell of a column is the key in the first object, on another line
                g["bind_col_header"].append((q, T[(2, c)]["key"]))
    g["ans_value"] = T[(tr, tc)]["value"]
    return g


N, R, C = 16, 8, 32
tables = seeded_tables(51, N, R, C)
out = {}
for fmt in ["markdown", "tagged", "json"]:
    acc.clear()
    for n, (rows, tr, tc) in enumerate(tables):
        if fmt == "json":
            text = chat(tok, json_prompt(rows, tr, tc)) + '{"row_ID": "%d", "column_ID": "%d", "cell_value": "' % (tr, tc)
        else:
            p, f = sz.build(rows, tr, tc, fmt)
            text = chat(tok, p) + f
        ids, offs = token_ids(tok, text)
        G.clear()
        G.update(groups_json(text, offs, R, C, tr, tc) if fmt == "json" else groups_table(text, offs, fmt, tr, tc))
        with torch.no_grad():
            model(torch.tensor([ids], device="cuda"))
        print(fmt, n, len(ids), flush=True)
    res = {}
    for li, d in acc.items():
        for k, v in d.items():
            res.setdefault(k, np.zeros((NL, NH)))[li] = np.mean(v, 0)
    out[fmt] = {k: v.tolist() for k, v in res.items()}
    for k, a in res.items():
        print(f"{fmt:9s} {k:16s} max {a.max():.2f} at L{a.argmax() // NH}H{a.argmax() % NH}; heads > 0.5: {(a > 0.5).sum()}", flush=True)
json.dump(out, open(OUT_DIR / "roles.json", "w"))

A = lambda f, k: np.array(out[f][k])  # noqa: E731
top = lambda M, n: [(int(i // NH), int(i % NH)) for i in np.argsort(-M, axis=None)[:n]]  # noqa: E731
heads = lambda M, thr: [tuple(map(int, x)) for x in np.argwhere(M > thr)]  # noqa: E731
pop = {"counting": heads(A("markdown", "count"), 0.5), "coordinate": heads(A("tagged", "bind_own"), 0.2),
       "coordinate_wide": heads(A("tagged", "bind_own"), 0.15), "column": top(A("markdown", "bind_col_header"), 8),
       "movers": top(A("tagged", "ans_value"), 8), "movers4": top(A("tagged", "ans_value"), 4)}
json.dump(pop, open(OUT_DIR / "populations.json", "w"), indent=1)
print({k: len(v) for k, v in pop.items()})
print("done", flush=True)
