#!/usr/bin/env python
"""Attention on one real NIAT table (8 x 8 crop) under Markdown, JSON and CCS: how the table is read and where
the answer position looks (the attention figures of the paper).

For every cell-locating query of the table the prompt is run with the answer forced up to ``"cell_value": "``.
Attention is averaged over heads; the cell-to-cell map (every cell -> the earlier cells, from the first
query) and the answer map (the position that generates the value -> every cell, per query) are summed over
the tokens of each cell. Attention-sink tokens (a cell token that receives more than ten times the median
attention from the cell tokens) are excluded on the key side. Greedy generation gives the accuracy.

In JSON a cell of row r >= 2 is the '"key": "value"' pair of column c in object r-1; the header row has no
cell of its own (its cells are the keys), so its entries are NaN / 0.

    python mechanistic/attention_niat.py                          # markdown tagged json
    python mechanistic/attention_niat.py --formats markdown tagged --table TabFact_1-16457934-2.html.csv_32X32_8x8

Writes results/mechanistic/attention/attn_<fmt>.npz (cc: L x 64 x 64; ans: L x queries x 64; json also
ans_value), preds_<fmt>.json and attn_stats_<fmt>.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JSON_PAIR, OUT, chat, load_hf_model, toks_in  # noqa: E402  (puts the repository root on the path)
import benchmark  # noqa: E402
import solve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--formats", nargs="+", default=["markdown", "tagged", "json"])
ap.add_argument("--table", default="TabFact_1-16457934-2.html.csv_32X32_8x8")
args, _ = ap.parse_known_args()   # attention_crops.py imports this module with its own arguments
OUT_DIR = OUT / "attention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

recs = [r for r in benchmark.load(["niat_cropped"])["niat_cropped"] if r["meta"]["table_id"] == args.table]
recs.sort(key=lambda r: (r["meta"]["row"], r["meta"]["col"]))
R = max(r["meta"]["row"] for r in recs)
C = max(r["meta"]["col"] for r in recs)
print(f"table {args.table}: {len(recs)} queries, {R}x{C}", flush=True)
tok, model = load_hf_model()
L = model.config.num_hidden_layers
cells = [(i, j) for i in range(1, R + 1) for j in range(1, C + 1)]
N = len(cells)


def prompt_of(rec, fmt):
    if fmt == "markdown":
        return rec["prompt"]
    if fmt == "tagged":
        return solve.transform(rec["prompt"], "niat_cropped")
    with solve.configured(SERIALIZATION="json"):
        return solve.transform(rec["prompt"], "niat_cropped")


def cell_spans(text, fmt):
    """(row, col) -> {"pair": (a, b)} (Markdown / CCS: the cell unit; JSON also "value")."""
    spans = {}
    if fmt == "markdown":
        s = text.index("Given table:\n") + len("Given table:\n")
        e = text.index("\n\n", s)
        row = 0
        for m in re.finditer(r"[^\n]+", text[s:e]):
            line = m.group(0)
            if re.match(r"^\|?\s*:?-{2,}", line):
                continue
            row += 1
            col = 0
            for cm in re.finditer(r"\|([^|]*)", line):
                inner = cm.group(1)
                if cm.end() == len(line) and inner.strip() == "":
                    break
                col += 1
                st = cm.start(1) + (len(inner) - len(inner.lstrip()))
                en = cm.start(1) + len(inner.rstrip())
                if en <= st:
                    st, en = cm.start(1), cm.start(1) + max(1, len(inner))
                spans[(row, col)] = {"pair": (s + m.start() + st, s + m.start() + en)}
    elif fmt == "tagged":
        s = text.index("row 1 (header): ")
        e = text.index("\n\n", s)
        for m in re.finditer(r"[^\n]+", text[s:e]):
            line = m.group(0)
            row = int(re.match(r"row (\d+)", line).group(1))
            head = line.index(": ") + 2
            col = 0
            for cm in re.finditer(r"\[r\d+\|c\d+=.*?\](?= \[|$)", line[head:]):
                col += 1
                spans[(row, col)] = {"pair": (s + m.start() + head + cm.start(), s + m.start() + head + cm.end())}
    else:
        s = text.index("Given table:\n")
        s = text.index("[\n", s) + 2
        e = text.index("\n]", s)
        for li, m in enumerate(re.finditer(r"[^\n]+", text[s:e])):
            a = s + m.start()
            for col, pm in enumerate(JSON_PAIR.finditer(m.group(0)), 1):
                v0, v1 = pm.start(2), pm.end(2)
                if v1 == v0:
                    v0, v1 = v0 - 1, v1 + 1
                spans[(li + 2, col)] = {"pair": (a + pm.start(), a + pm.end()), "value": (a + v0, a + v1)}
    expected = (R - 1) * C if fmt == "json" else R * C
    assert len(spans) == expected, (fmt, len(spans))
    return spans


def run(fmt):
    cc = np.full((L, N, N), np.nan if fmt == "json" else 0.0, np.float32)
    ans = np.zeros((L, len(recs), N), np.float32)
    ans_value = np.zeros_like(ans)
    preds = []
    for qi, r in enumerate(recs):
        text = chat(tok, prompt_of(r, fmt))
        forced = '{"row_ID": "%d", "column_ID": "%d", "cell_value": "' % (r["meta"]["row"], r["meta"]["col"])
        full = text + forced
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        T = len(ids)
        sp = cell_spans(full, fmt)
        tp = {k: toks_in(offs, v["pair"]) for k, v in sp.items()}
        assert all(tp.values())
        live = [n for n, k in enumerate(cells) if k in tp]
        Pp = torch.zeros(N, T, device="cuda")
        Pv = torch.zeros(N, T, device="cuda")
        for n, k in enumerate(cells):
            if k in tp:
                Pp[n, tp[k]] = 1.0
                Pv[n, toks_in(offs, sp[k]["value"]) if "value" in sp[k] else tp[k]] = 1.0
        with torch.no_grad():
            out = model(torch.tensor([ids], device="cuda"), output_attentions=True)
            Am = torch.stack([a[0].float().mean(0) for a in out.attentions]).mean(0)   # layers and heads averaged
            cell_tok = sorted(set(t for v in tp.values() for t in v))
            recv = Am[cell_tok][:, cell_tok].sum(0)
            sinks = [cell_tok[i] for i in torch.nonzero(recv > 10 * recv.median()).flatten().tolist()]
            Pp[:, sinks] = 0.0
            Pv[:, sinks] = 0.0
            if qi == 0:
                print(fmt, "sink tokens excluded:", [(t, tok.decode([ids[t]])) for t in sinks], flush=True)
            Pq = Pp[live] / Pp[live].sum(1, keepdim=True)
            for l, A in enumerate(out.attentions):
                Ah = A[0].float().mean(0)                           # heads averaged, T x T
                if qi == 0:
                    cc[l][np.ix_(live, live)] = (Pq @ Ah @ Pp[live].T).cpu().numpy()
                ans[l, qi] = (Pp @ Ah[-1]).cpu().numpy()             # the position that generates the value -> each cell
                ans_value[l, qi] = (Pv @ Ah[-1]).cpu().numpy()
            del out
            pid = tok(text, add_special_tokens=False)["input_ids"]
            g = model.generate(torch.tensor([pid], device="cuda"), max_new_tokens=64, do_sample=False)
        txt = tok.decode(g[0][len(pid):], skip_special_tokens=True)
        m = re.search(r'"cell_value":\s*"(.*?)"', txt)
        preds.append({"row": r["meta"]["row"], "col": r["meta"]["col"], "gt": r["gt"], "pred": m.group(1) if m else txt.strip(),
                      "correct": bool(m and m.group(1) == r["gt"])})
        if qi % 16 == 0:
            print(fmt, qi, T, "tokens", preds[-1], flush=True)
    return cc, ans, ans_value, preds, T


for fmt in args.formats:
    cc, ans, ans_value, preds, T = run(fmt)
    arrays = {"cc": cc, "ans": ans, "cells": np.array(cells)}
    if fmt == "json":
        arrays["ans_value"] = ans_value
    np.savez_compressed(OUT_DIR / f"attn_{fmt}.npz", **arrays)
    json.dump(preds, open(OUT_DIR / f"preds_{fmt}.json", "w"), indent=1)
    tgt = np.array([cells.index((r["meta"]["row"], r["meta"]["col"])) for r in recs])
    live_q = np.array([r["meta"]["row"] >= 2 for r in recs]) if fmt == "json" else np.ones(len(recs), bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        an = ans / ans.sum(2, keepdims=True)
    st = {"table": args.table, "R": R, "C": C, "layers": L, "n_tokens_example": int(T), "acc": float(np.mean([p["correct"] for p in preds])),
          "target_share": [float(np.mean(an[l][live_q, tgt[live_q]])) for l in range(L)],
          "target_top1": [float(np.mean(an[l][live_q].argmax(1) == tgt[live_q])) for l in range(L)],
          "chance_target": 1 / int(live_q.sum())}
    json.dump(st, open(OUT_DIR / f"attn_stats_{fmt}.json", "w"), indent=1)
    print(fmt, "acc", st["acc"], "max target share", max(st["target_share"]), "max top-1 rate", max(st["target_top1"]), flush=True)
print("done", flush=True)
