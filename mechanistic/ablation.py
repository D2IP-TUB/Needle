#!/usr/bin/env python
"""Head knock-outs: does the answer depend on the count? (the causal numbers and the counting-head ablation
table of the paper)

A group of heads is removed by zeroing its output (the head's slice of the o_proj input) at every position.
On 40 tables of 8 x 16 (cell-locating query) three quantities are read out downstream: how much of the column
index a layer-18 ridge probe fitted on 24 unablated tables of the same serialization still reads from the
last token of each value (count code, R^2), how much attention the four strongest mover heads place on the
target value, and the answer accuracy (first token). Every ablation is compared with ten random sets of the
same number of heads drawn from the same layers.

Populations come from results/mechanistic/heads/populations.json (head_roles.py). Groups per serialization:
the counting heads everywhere; for CCS also the coordinate heads and the coordinate (wide) + column + mover
heads together.

    python mechanistic/ablation.py --formats rowpre_md coltag tagged rowpre_coltag   # the causal-path formats
    python mechanistic/ablation.py --formats plain valuefirst nametag jsonl           # the other serializations

Writes results/mechanistic/ablation/<format>.json.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as sz  # noqa: E402
from common import OUT, chat, load_hf_model, synth_rows, toks_in, token_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--formats", nargs="+", default=["rowpre_md", "coltag", "tagged", "rowpre_coltag"])
ap.add_argument("--tables", type=int, default=40)
ap.add_argument("--random-sets", type=int, default=10)
args = ap.parse_args()
OUT_DIR = OUT / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
tok, model = load_hf_model()
NL, NH, HD, D = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.head_dim, model.config.hidden_size

pop = json.load(open(OUT / "heads" / "populations.json"))
pop = {k: [tuple(h) for h in v] for k, v in pop.items()}
POP = {"counting": pop["counting"], "coordinate": pop["coordinate"],
       "coordinate_wide + column + movers": sorted(set(pop["coordinate_wide"]) | set(pop["column"]) | set(pop["movers"]))}
MOVERS4 = pop["movers4"]


def random_like(heads, seed):
    r = random.Random(seed)
    out = set()
    layers = [l for l, h in heads]
    ex = set(heads)
    while len(out) < len(heads):
        c = (r.choice(layers), r.randint(0, NH - 1))
        if c not in ex:
            out.add(c)
    return sorted(out)


def conditions(keep):
    return ([("none", [])] + [(k, POP[k]) for k in keep]
            + [(f"random like {k}, draw {d}", random_like(POP[k], 100 * d + len(POP[k]))) for k in keep for d in range(args.random_sets)])


# ---- value spans (row, col, (a, b)) of the last table block, per serialization
def values(text, fmt):
    out = []
    if fmt == "markdown":
        b = [m for m in re.finditer(r"(?:^\|.*\|\n?)+", text, re.M)][-1]
        row = 0
        for m in re.finditer(r"[^\n]+", b.group(0)):
            line = m.group(0)
            lb = b.start() + m.start()
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
                out.append((row, col, (lb + st, lb + en)))
        return out
    if fmt == "rowpre_md":
        b = [m for m in re.finditer(r"(?:^row \d+.*\n?)+", text, re.M)][-1]
        for m in re.finditer(r"[^\n]+", b.group(0)):
            line = m.group(0)
            lb = b.start() + m.start()
            pm = re.match(r"row (\d+)(?: \(header\))?: ", line)
            row = int(pm.group(1))
            col = 0
            rest = line[pm.end():]
            for cm in re.finditer(r"\|([^|]*)", rest):
                inner = cm.group(1)
                if cm.end() == len(rest) and inner.strip() == "":
                    break
                col += 1
                st = cm.start(1) + (len(inner) - len(inner.lstrip()))
                en = cm.start(1) + len(inner.rstrip())
                out.append((row, col, (lb + pm.end() + st, lb + pm.end() + en)))
        return out
    if fmt in ("tagged", "rowpre_coltag", "coltag", "nametag", "valuefirst", "plain"):
        pat = {"tagged": r"\[r(\d+)\|c(\d+)=(.*?)\](?= \[|$)", "rowpre_coltag": r"\[c(\d+)=(.*?)\](?= \[|$)", "coltag": r"\[c(\d+)=(.*?)\](?= \[|$)",
               "nametag": r"\[r(\d+)\|[^=\]]+?=(.*?)\](?= \[|$)", "valuefirst": r"\[(.*?) @ r(\d+)\|c(\d+)\](?= \[|$)", "plain": r"r(\d+)c(\d+):(\S+)"}[fmt]
        blk = r"(?:^row \d+.*\n?)+" if fmt != "coltag" else r"(?:^\[c1=.*\n?)+"
        b = [m for m in re.finditer(blk, text, re.M)][-1]
        li = 0
        for m in re.finditer(r"[^\n]+", b.group(0)):
            line = m.group(0)
            lb = b.start() + m.start()
            li += 1
            row = int(re.match(r"row (\d+)", line).group(1)) if fmt != "coltag" else li
            col = 0
            for cm in re.finditer(pat, line):
                col += 1
                if fmt == "valuefirst":
                    sp = (cm.start(1), cm.end(1))
                elif fmt in ("tagged", "nametag", "plain"):
                    g = 2 if fmt == "nametag" else 3
                    sp = (cm.start(g), cm.end(g))
                else:
                    sp = (cm.start(2), cm.end(2))
                out.append((row, col, (lb + sp[0], lb + sp[1])))
        return out
    if fmt == "jsonl":
        b = [m for m in re.finditer(r"(?:^\{.*\}\n?)+", text, re.M)][-1]
        row = 0
        for m in re.finditer(r"[^\n]+", b.group(0)):
            line = m.group(0)
            lb = b.start() + m.start()
            row += 1
            col = 0
            for cm in re.finditer(r'"[^"]*": "([^"]*)"', line):
                col += 1
                out.append((row, col, (lb + cm.start(1), lb + cm.end(1))))
        return out
    raise ValueError(fmt)


ABL, CAPA = {}, {}


def pre(li):
    def f(mod, inp):
        hs = ABL.get(li)
        if not hs:
            return None
        x = inp[0].clone()
        for h in hs:
            x[..., h * HD:(h + 1) * HD] = 0
        return (x,) + tuple(inp[1:])
    return f


def att(li):
    def f(mod, inp, out):
        if li in CAPA:
            CAPA[li] = out[1][0, :, -1, :].float().cpu()
    return f


for li, layer in enumerate(model.model.layers):
    layer.self_attn.o_proj.register_forward_pre_hook(pre(li))
    layer.self_attn.register_forward_hook(att(li))


def set_ablation(heads):
    ABL.clear()
    for l, h in heads:
        ABL.setdefault(l, []).append(h)


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))


R, C, N, NP = 8, 16, args.tables, 24
rng = random.Random(11)
tables = [(synth_rows(rng, R, C), rng.randint(2, R), rng.randint(2, C)) for _ in range(N)]
rng2 = random.Random(12)
ptables = [(synth_rows(rng2, R, C), rng2.randint(2, R), rng2.randint(2, C)) for _ in range(NP)]
for fmt in args.formats:
    keep = ["counting"] + (["coordinate", "coordinate_wide + column + movers"] if fmt == "tagged" else [])
    # the count probe of this serialization: layer 18, last token of each body value -> column index, unablated tables
    set_ablation([])
    X, Y = [], []
    for rows, tr, tc in ptables:
        p, f = sz.build(rows, tr, tc, fmt)
        text = chat(tok, p) + f
        ids, offs = token_ids(tok, text)
        with torch.no_grad():
            H = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states[18][0].float().cpu().numpy()
        for row, col, sp in values(text, fmt):
            t = toks_in(offs, sp)
            if row > 1 and t:
                X.append(H[t[-1]])
                Y.append(col)
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    Wc = np.linalg.solve(Xb.T @ Xb + 10 * np.eye(D + 1), Xb.T @ Y)
    print(fmt, "probe fitted on", len(Y), "values; train R2 %.2f" % r2(Xb @ Wc, Y), flush=True)
    res = []
    for cname, heads in conditions(keep):
        set_ablation(heads)
        recs = []
        for rows, tr, tc in tables:
            p, f = sz.build(rows, tr, tc, fmt)
            text = chat(tok, p) + f
            ids, offs = token_ids(tok, text)
            V = values(text, fmt)
            tgt = [sp for row, col, sp in V if (row, col) == (tr, tc)][0]
            for l, h in MOVERS4:
                CAPA[l] = None
            with torch.no_grad():
                out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
            H18 = out.hidden_states[18][0].float().cpu().numpy()
            vt = [(toks_in(offs, sp)[-1], col) for row, col, sp in V if row > 1 and toks_in(offs, sp)]
            Xv = np.concatenate([H18[[i for i, _ in vt]], np.ones((len(vt), 1))], 1)
            yv = np.array([c for _, c in vt], float)
            cnt = r2(Xv @ Wc, yv)
            mover = float(np.mean([CAPA[l][h, toks_in(offs, tgt)].sum() for l, h in MOVERS4]))
            logp = torch.log_softmax(out.logits[0, -1].float(), -1)
            g = tok(rows[tr - 1][tc - 1], add_special_tokens=False)["input_ids"][0]
            lp = logp.clone()
            lp[g] = -1e9
            recs.append({"count_r2": cnt, "mover": mover, "correct": int(logp.argmax()) == g, "margin": float(logp[g] - lp.max())})
        agg = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        agg["margin_median"] = float(np.median([r["margin"] for r in recs]))
        res.append({"fmt": fmt, "ablation": cname, "n_heads": len(heads), **agg})
        print(f"{fmt:14s} {cname:44s} | count R2 {agg['count_r2']:.2f} | mover attn {agg['mover']:.2f} | acc {agg['correct']:.2f} margin {agg['margin_median']:+.1f}", flush=True)
    json.dump(res, open(OUT_DIR / f"{fmt}.json", "w"), indent=1)
set_ablation([])
print("done")
