"""Synthetic tables in ten serializations and the behavioural runner (greedy answer, first-token margin).

Serializations (``table_text``): the four of the counted-axis analysis (``markdown`` writes neither
coordinate, ``rowpre_md`` the row once per line, ``coltag`` the column on every cell, ``tagged`` both) and
the six of the serialization table (``csv``, ``jsonl``, ``nametag``, ``rowpre_coltag``, ``valuefirst``,
``plain``). The query is NIAT's cell-locating query by row and column index.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, WORDS, chat, md_prompt, md_table, synth_rows, tagged_prompt, load_hf_model  # noqa: E402,F401

tok = model = None


def load():
    global tok, model
    if model is None:
        tok, model = load_hf_model(eager=False)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    return tok, model


# ----------------------------------------------------------------------------------------------- tables
def md_lines(rows):
    C = len(rows[0])
    w = [max(len(r[j]) for r in rows) for j in range(C)]
    return ["| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(r)) + " |" for r in rows]


def rowpre(i):
    return f"row {i} (header): " if i == 1 else f"row {i}: "


NOTES = {
    "tagged":        "every cell is written as [r<row_ID>|c<column_ID>=value], the header being row 1",
    "rowpre_md":     "every line starts with its row_ID, cells are separated by |, the header being row 1",
    "coltag":        "every cell is written as [c<column_ID>=value]; the lines are the rows in order, the header being row 1",
    "rowpre_coltag": "every line starts with its row_ID and every cell is written as [c<column_ID>=value], the header being row 1",
    "valuefirst":    "every cell is written as [value @ r<row_ID>|c<column_ID>], the header being row 1",
    "plain":         "every cell is written as r<row_ID>c<column_ID>:value, the header being row 1",
    "nametag":       "every cell is written as [r<row_ID>|<column name>=value], the header being row 1",
    "jsonl":         "every line is one row as a JSON object {\"<column name>\": value}, the header being row 1",
    "csv":           "comma-separated values, one row per line, the header being row 1",
}
FORMATS = ["markdown", "csv", "jsonl", "rowpre_md", "coltag", "nametag", "rowpre_coltag", "valuefirst", "plain", "tagged"]
LABELS = {"markdown": "Markdown", "csv": "CSV", "jsonl": "JSON rows", "rowpre_md": "row identifier per line",
          "coltag": "column tag", "nametag": "row identifier, column name", "rowpre_coltag": "row identifier per line + column tag",
          "valuefirst": "value first", "plain": "no brackets", "tagged": "CCS"}


def table_text(rows, fmt):
    """Table block (without the 'Given table:' line): markdown -> md_table; others -> note line + lines."""
    R = len(rows)
    if fmt == "markdown":
        return md_table(rows)
    if fmt == "tagged":
        lines = [rowpre(i + 1) + " ".join(f"[r{i + 1}|c{j + 1}={v}]" for j, v in enumerate(r)) for i, r in enumerate(rows)]
    elif fmt == "rowpre_md":
        ml = md_lines(rows)
        lines = [rowpre(i + 1) + ml[i] for i in range(R)]
    elif fmt == "coltag":
        lines = [" ".join(f"[c{j + 1}={v}]" for j, v in enumerate(r)) for r in rows]
    elif fmt == "rowpre_coltag":
        lines = [rowpre(i + 1) + " ".join(f"[c{j + 1}={v}]" for j, v in enumerate(r)) for i, r in enumerate(rows)]
    elif fmt == "valuefirst":
        lines = [rowpre(i + 1) + " ".join(f"[{v} @ r{i + 1}|c{j + 1}]" for j, v in enumerate(r)) for i, r in enumerate(rows)]
    elif fmt == "plain":
        lines = [rowpre(i + 1) + " ".join(f"r{i + 1}c{j + 1}:{v}" for j, v in enumerate(r)) for i, r in enumerate(rows)]
    elif fmt == "nametag":
        lines = [rowpre(i + 1) + " ".join(f"[r{i + 1}|{rows[0][j]}={v}]" for j, v in enumerate(r)) for i, r in enumerate(rows)]
    elif fmt == "jsonl":
        lines = [json.dumps({rows[0][j]: v for j, v in enumerate(r)}) for r in rows]
    elif fmt == "csv":
        lines = [",".join(r) for r in rows]
    else:
        raise ValueError(fmt)
    return f"(The table is written one row per line; {NOTES[fmt]}.)\n" + "\n".join(lines)


# ----------------------------------------------------------------------------------------------- prompts
def index_prompt(rows, tr, tc, fmt):
    """The cell-locating prompt by row and column index. markdown = NIAT's prompt; tagged = the exact CCS
    transform of solve.py (target repeated before the table); the other serializations get the same
    target-repeat line."""
    md = md_prompt(rows, tr, tc)
    if fmt == "markdown":
        return md
    if fmt == "tagged" and len(rows[0]) <= 128:
        return tagged_prompt(md)
    s = md.index("Given table:\n")
    e = md.index("\n\n", s + len("Given table:\n"))
    return md[:s] + f"Target cell (repeated after the table): row_ID {tr}, column_ID {tc}.\n\nGiven table:\n" + table_text(rows, fmt) + md[e:]


def build(rows, tr, tc, fmt):
    """(prompt, forced answer prefix): the model continues after ``"cell_value": "`` with the value."""
    return index_prompt(rows, tr, tc, fmt), '{"row_ID": "%d", "column_ID": "%d", "cell_value": "' % (tr, tc)


# ----------------------------------------------------------------------------------------------- running
def batches(lengths, order, budget):
    b = []
    for i in order:
        if b and (len(b) + 1) * max(lengths[j] for j in b + [i]) > budget:
            yield b
            b = []
        b.append(i)
    if b:
        yield b


@torch.no_grad()
def generate(texts, max_new=16, budget=int(os.environ.get("CCS_MECH_BUDGET", 90000)), margin_ids=None):
    """Greedy continuation of each text; with ``margin_ids`` (gold first token per text) also the log-prob
    margin of that token at the first generated position (logp(gold) - max logp(other)) and its rank."""
    tok, model = load()
    enc_all = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    print("  tokenised", len(texts), "prompts, max", max(map(len, enc_all)), "tokens", flush=True)
    lengths = [len(e) for e in enc_all]
    outs = [None] * len(texts)
    marg = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    done = 0
    t0 = time.time()
    for ix in batches(lengths, order, budget):
        ml = max(lengths[i] for i in ix)
        ids = torch.full((len(ix), ml), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ix), ml), dtype=torch.long)
        for k, i in enumerate(ix):
            ids[k, ml - lengths[i]:] = torch.tensor(enc_all[i])
            att[k, ml - lengths[i]:] = 1
        ids, att = ids.cuda(), att.cuda()
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
        first = torch.log_softmax(g.scores[0].float(), -1)
        for k, i in enumerate(ix):
            outs[i] = tok.decode(g.sequences[k, ml:], skip_special_tokens=True)
            if margin_ids is not None and margin_ids[i] is not None:
                lp = first[k]
                gold = margin_ids[i]
                lg = lp[gold].item()
                lp2 = lp.clone()
                lp2[gold] = -1e9
                marg[i] = {"logp": lg, "margin": lg - lp2.max().item(), "rank": int((lp > lg).sum().item()) + 1}
        done += len(ix)
        if done % 200 < len(ix):
            print(f"  gen {done}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    return outs, marg, lengths


def first_token(s):
    tok, _ = load()
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def parse(out):
    """The answered value: the text up to the closing quote."""
    return out.split('"')[0].strip()


def locate(rows, val):
    """(row, col) of a value in the table (None if absent or ambiguous)."""
    hits = [(i + 1, j + 1) for i, r in enumerate(rows) for j, v in enumerate(r) if v == val]
    return hits[0] if len(hits) == 1 else None


def sample_targets(rng, R, C):
    return rng.randint(2, R), rng.randint(2, C)


def run(name, configs, N=100, seed=11, max_new=16, out_dir=None):
    """configs: list of (fmt, R, C). Each size gets N tables (built in order of first appearance, from one
    seeded generator), shared by every serialization listed for it. Writes <out_dir>/sz_<name>.json."""
    out_dir = Path(out_dir or OUT / "sizes")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model = load()
    rng = random.Random(seed)
    tables = {}
    for fmt, R, C in configs:
        if (R, C) not in tables:
            tables[(R, C)] = []
            for _ in range(N):
                rows = synth_rows(rng, R, C)
                tr, tc = sample_targets(rng, R, C)
                tables[(R, C)].append((rows, tr, tc))
    print(name, "tables built", flush=True)
    jobs = []
    for fmt, R, C in configs:
        for n, (rows, tr, tc) in enumerate(tables[(R, C)]):
            p, forced = build(rows, tr, tc, fmt)
            jobs.append((fmt, R, C, n, rows, tr, tc, chat(tok, p) + forced))
    print(name, len(jobs), "generations", flush=True)
    gold_ids = [first_token(rows[tr - 1][tc - 1]) for fmt, R, C, n, rows, tr, tc, t in jobs]
    outs, marg, lengths = generate([j[-1] for j in jobs], max_new=max_new, margin_ids=gold_ids)
    rec = []
    for (fmt, R, C, n, rows, tr, tc, t), o, m, ln in zip(jobs, outs, marg, lengths):
        gold = rows[tr - 1][tc - 1]
        a = parse(o)
        ok = a == gold
        loc = locate(rows, a) if not ok else (tr, tc)
        rec.append({"fmt": fmt, "task": "index", "R": R, "C": C, "n": n, "tr": tr, "tc": tc, "gold": gold, "pred": a,
                    "correct": bool(ok), "prow": loc[0] if loc else None, "pcol": loc[1] if loc else None, "ntok": ln,
                    "margin": m["margin"] if m else None, "logp": m["logp"] if m else None, "rank": m["rank"] if m else None})
    json.dump(rec, open(out_dir / f"sz_{name}.json", "w"))
    summary(rec)
    return rec


def summary(rec, keys=("fmt", "R", "C")):
    agg = {}
    for r in rec:
        agg.setdefault(tuple(r[k] for k in keys), []).append(r)
    print("  ".join(f"{k:>12s}" for k in keys) + "      acc   same-row  same-col   med-margin  ntok")
    for k in sorted(agg, key=lambda k: [str(x) for x in k]):
        rs = agg[k]
        acc = np.mean([r["correct"] for r in rs])
        wrong = [r for r in rs if not r["correct"] and r["prow"] is not None]
        sr = np.mean([r["prow"] == r["tr"] for r in wrong]) if wrong else float("nan")
        sc = np.mean([r["pcol"] == r["tc"] for r in wrong]) if wrong else float("nan")
        mg = np.median([r["margin"] for r in rs if r["margin"] is not None]) if any(r["margin"] is not None for r in rs) else float("nan")
        print("  ".join(f"{str(x):>12s}" for x in k) + f"   {acc:.3f}   {sr:8.2f}  {sc:8.2f}   {mg:9.2f}  {int(np.mean([r['ntok'] for r in rs]))}", flush=True)
