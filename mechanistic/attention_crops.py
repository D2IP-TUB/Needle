#!/usr/bin/env python
"""Cell-to-cell attention on larger crops of the same NIAT table (16 x 16, 32 x 32; the appendix figures), and
the cell-locating accuracy on every query of each crop, under Markdown, JSON and CCS.

One forward pass per size and format from the (1,1) query with the forced answer prefix; the attention is
reduced to the cell tokens layer by layer inside a hook so that the 32 x 32 prompt fits in memory. Then greedy
generation on every cell-locating query of the crop (64 / 256 / 1,024).

    python mechanistic/attention_crops.py --sizes 16 32 --formats markdown tagged json

Writes results/mechanistic/attention/attn_cc_<size>_<fmt>.npz (cc: L x N x N) and merges the accuracies into
attn_cc_acc.json ({"<size>_<fmt>": {"n", "correct", "preds"}}).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, chat, toks_in  # noqa: E402  (puts the repository root on the path)
import benchmark  # noqa: E402
import attention_niat as A  # noqa: E402  (loads the model and the 8x8 table on import)
from attention_niat import cell_spans as _cell_spans  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
ap.add_argument("--formats", nargs="+", default=["markdown", "tagged", "json"])
ap.add_argument("--base", default="TabFact_1-16457934-2.html.csv_32X32")
args, _ = ap.parse_known_args()
OUT_DIR = OUT / "attention"
tok, model, L = A.tok, A.model, A.L
allrecs = benchmark.load(["niat_cropped"])["niat_cropped"]
state = {}


def hook(layer):
    def f(mod, inp, out):
        w = out[1][0]                                    # heads x T x T
        ct = state["cell_tok"]
        acc = torch.zeros(len(ct), len(ct), device=w.device)
        for h in range(w.shape[0]):
            acc += w[h][ct][:, ct].float()
        state["sub"][layer] = acc / w.shape[0]           # heads averaged, restricted to the cell tokens
        return (out[0], None) + tuple(out[2:])
    return f


def cc_map(size, fmt):
    A.R = A.C = size
    hooks = [lyr.self_attn.register_forward_hook(hook(l)) for l, lyr in enumerate(model.model.layers)]
    r = [r for r in allrecs if r["meta"]["table_id"] == f"{args.base}_{size}x{size}" and r["meta"]["row"] == 1 and r["meta"]["col"] == 1][0]
    text = chat(tok, A.prompt_of(r, fmt)) + '{"row_ID": "1", "column_ID": "1", "cell_value": "'
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    T = len(ids)
    tsets = {k: toks_in(offs, v["pair"]) for k, v in _cell_spans(text, fmt).items()}
    assert all(tsets.values())
    cells = [(i, j) for i in range(1, size + 1) for j in range(1, size + 1)]
    N = len(cells)
    live = [n for n, k in enumerate(cells) if k in tsets]
    cell_tok = sorted(set(t for v in tsets.values() for t in v))
    pos = {t: i for i, t in enumerate(cell_tok)}
    state["cell_tok"] = torch.tensor(cell_tok, device="cuda")
    state["sub"] = [None] * L
    with torch.no_grad():
        model(torch.tensor([ids], device="cuda"))
    Pk = torch.zeros(len(live), len(cell_tok), device="cuda")
    for n, li in enumerate(live):
        Pk[n, [pos[t] for t in tsets[cells[li]]]] = 1.0
    Pq = Pk / Pk.sum(1, keepdim=True)
    recv = torch.stack(state["sub"]).mean(0).sum(0)
    sinks = torch.nonzero(recv > 10 * recv.median()).flatten().tolist()   # attention-sink tokens
    Pk[:, sinks] = 0.0
    sub = torch.stack([Pq @ s @ Pk.T for s in state["sub"]]).cpu().numpy().astype(np.float32)
    cc = np.full((L, N, N), np.nan if fmt == "json" else 0.0, np.float32)
    cc[:, np.ix_(live, live)[0], np.ix_(live, live)[1]] = sub
    state["sub"] = None
    torch.cuda.empty_cache()
    for h in hooks:
        h.remove()
    print(f"{size}x{size} {fmt}: {T} tokens, {len(cell_tok)} cell tokens, sinks {[(cell_tok[i], tok.decode([ids[cell_tok[i]]])) for i in sinks]}", flush=True)
    np.savez_compressed(OUT_DIR / f"attn_cc_{size}_{fmt}.npz", cc=cc, cells=np.array(cells))
    return T


def accuracy(size, fmt):
    recs = [r for r in allrecs if r["meta"]["table_id"] == f"{args.base}_{size}x{size}"]
    recs.sort(key=lambda r: (r["meta"]["row"], r["meta"]["col"]))
    preds = []
    for qi, r in enumerate(recs):
        if qi % 128 == 0:
            print("acc", size, fmt, qi, sum(p["correct"] for p in preds), flush=True)
        text = chat(tok, A.prompt_of(r, fmt))
        enc = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=64, do_sample=False)
        txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r'"cell_value":\s*"(.*?)"', txt)
        preds.append({"row": r["meta"]["row"], "col": r["meta"]["col"], "gt": r["gt"], "pred": m.group(1) if m else txt.strip(),
                      "correct": bool(m and m.group(1) == r["gt"])})
    return {"n": len(preds), "correct": sum(p["correct"] for p in preds), "preds": preds}


if __name__ == "__main__":
    accf = OUT_DIR / "attn_cc_acc.json"
    res = json.load(open(accf)) if accf.exists() else {}
    for size in args.sizes:
        for fmt in args.formats:
            cc_map(size, fmt)
            res[f"{size}_{fmt}"] = accuracy(size, fmt)
            print(size, fmt, res[f"{size}_{fmt}"]["correct"], "/", res[f"{size}_{fmt}"]["n"], flush=True)
            json.dump(res, open(accf, "w"), indent=1)
    print("done", flush=True)
