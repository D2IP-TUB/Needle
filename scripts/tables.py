#!/usr/bin/env python
"""Print the tables of the paper from the result files in results/ (no GPU).

    python scripts/tables.py            # every table
    python scripts/tables.py main ablation transfer tokens table_qa fullset sets

Rows are given in the paper's order; a cell is "-" when its run is not in results/.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark  # noqa: E402
from ccs.models import MODELS, TRANSFER_MODELS  # noqa: E402

R = ROOT / "results"
SETS = ["mmtu_niht", "mmtu_table_locate", "niat_cell_locating", "niat_cropped", "niat_cell_lookup"]
HEAD = ["needle", "table-locate", "cell-locating", "cropped", "cell-lookup", "score"]
KEYS = {"mmtu_niht": "mmtu_niht_acc", "mmtu_table_locate": "mmtu_table_locate_acc", "niat_cell_locating": "niat_cell_locating_em",
        "niat_cropped": "niat_cropped_em", "niat_cell_lookup": "niat_cell_lookup_em"}


def subset(model, preset):
    f = R / "subset" / f"{model}__{preset}.json"
    return json.load(open(f)) if f.exists() else None


def fullset(preset, model="qwen3-4b-2507"):
    """Per-set accuracies of a full-set run directory, plus the aggregate when all five sets are there."""
    d = R / "fullset" / f"{model}__{preset}"
    acc = {}
    for s in SETS:
        f = d / f"metrics_{s}.json"
        if f.exists():
            acc[s] = json.load(open(f))["acc"]
    return acc


def fmt(v, digits=3):
    return "-" if v is None or v != v else f"{v:.{digits}f}"


def row(label, cells):
    print(f"{label:46s}" + "".join(f"{c:>14s}" for c in cells))


def header(title, cols=HEAD):
    print(f"\n== {title}")
    row("", cols)


def main_table():
    header("Table 1: serialization baselines (answer protocol) and CCS, Qwen3-4B-Instruct-2507")
    print("   serialization baselines on the evaluation subset (7,996 prompts); CCS and SpreadsheetLLM as in the paper")
    for label, preset in [("Markdown", "markdown_protocol"), ("HTML", "html_own_protocol"), ("CSV", "csv_protocol"), ("JSON", "jsonl_protocol")]:
        m = subset("qwen3-4b-2507", preset)
        row(label + " (subset)", [fmt(m and m[KEYS[s]]) for s in SETS] + [fmt(m and m["score"])])
    # CCS on the full sets: the needle task from the ccs_colword run, the other four sets from the ccs_valuefirst_needle
    # run (identical tag shape on those sets)
    a = fullset("ccs_valuefirst_needle")
    a_cw = fullset("ccs_colword")
    if a and a_cw:
        acc = dict(a, mmtu_niht=a_cw["mmtu_niht"])
        row("CCS (full sets; needle: colword shape)", [fmt(acc[s]) for s in SETS] + [fmt(benchmark.aggregate(acc)["score"])])
    for preset, label in [("ccs", "CCS [r|c=v] (full sets)"), ("ccs_colword", "CCS [row r, column c = v] (full sets)"),
                          ("ccs_valuefirst_needle", "CCS, value-first needle tag (full sets)"), ("full", "CCS, per-task shapes (full sets)")]:
        acc = fullset(preset)
        row(label, [fmt(acc.get(s)) for s in SETS] + [fmt(benchmark.aggregate(acc)["score"] if len(acc) == 5 else None)])
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from score_spreadsheetllm import mapped_score
        from ccs.utils import read_jsonl
        data = benchmark.load()
        for preset, label in [("spreadsheetllm", "SpreadsheetLLM (w/o inverted index), subset"), ("spreadsheetllm_inv", "SpreadsheetLLM, subset")]:
            f = R / "subset" / "outputs" / f"qwen3-4b-2507__{preset}.jsonl"
            if not f.exists():
                continue
            outs = {r["id"]: r["text"] for r in read_jsonl(f)}
            m = subset("qwen3-4b-2507", preset) or {}
            acc = {}
            for s, recs in data.items():
                if any(r["id"] in outs for r in recs):
                    acc[s] = sum(mapped_score(s, r, outs.get(r["id"], "")) for r in recs) / len(recs)
                elif KEYS[s] in m:
                    acc[s] = m[KEYS[s]]
            row(label + " (mapped)", [fmt(acc.get(s)) for s in SETS] + [fmt(benchmark.aggregate(acc)["score"] if len(acc) == 5 else None)])
    except FileNotFoundError as e:
        print("  (SpreadsheetLLM rows need the evaluation subset: python benchmark.py --build)", e)


def ablation():
    header("Table 2: ablation on the evaluation subset, Qwen3-4B-Instruct-2507 (per-task tag shapes)")
    for label, preset in [("CCS (w/o answer protocol)", "tags_only"), ("+ fixed answer format", "tags_protocol"), ("+ constrained decoding", "full")]:
        m = subset("qwen3-4b-2507", preset)
        row(label, [fmt(m and m[KEYS[s]]) for s in SETS] + [fmt(m and m["score"])])


def transfer():
    header("Table 3: aggregate score on the evaluation subset, each model at its own maximum context length", ["Markdown", "JSON", "CCS"])
    for model in TRANSFER_MODELS:
        ms = [subset(model, p) for p in ["markdown_protocol", "jsonl_protocol", "full"]]
        row(MODELS[model].label, [fmt(m and m["score"]) for m in ms])
    print("  (Mistral-7B's 32K window holds no MMTU prompt: its MMTU accuracy is 0 in every column)")


def tokens():
    header("Table 4: prompt length in Qwen3 tokens on the evaluation subset", ["published mean", "CCS mean", "CCS max"])
    for f in sorted(R.glob("tokens_*.json")):
        t = json.load(open(f))
        print(f"  configuration: {t['preset']}")
        for s in SETS:
            d = t["sets"][s]
            row("  " + s, [f"{d['orig_mean'] / 1000:.1f}k", f"{d['new_mean'] / 1000:.1f}k", f"{d['new_max'] / 1000:.1f}k"])


def table_qa():
    header("Table 5: MMTU Table-QA accuracy per source, brief reasoning", ["FinQA", "TableBench", "WikiQA", "task"])
    for f in sorted((R / "table_qa").glob("*_summary.json")):
        d = json.load(open(f))
        for label, v in [("Markdown", "reason_markdown"), ("JSON", "reason_json"), ("CCS", "reason_tagged")]:
            if v in d:
                p = d[v]["per_dataset"]
                row(label, [fmt(p.get("FinQA")), fmt(p.get("TableBench")), fmt(p.get("WikiQA")), fmt(d[v]["task_score"])])


def fullset_table():
    header("Every full-set run in results/fullset (Qwen3-4B-Instruct-2507)")
    for d in sorted((R / "fullset").iterdir()):
        preset = d.name.split("__", 1)[1]
        acc = fullset(preset)
        row(preset, [fmt(acc.get(s)) for s in SETS] + [fmt(benchmark.aggregate(acc)["score"] if len(acc) == 5 else None)])


def sets():
    header("Prompt sets", ["full", "subset"])
    ids = json.load(open(ROOT / "data" / "subset_ids.json"))
    full = {}
    for s in SETS:
        f = ROOT / "data" / "processed" / f"{s}.jsonl"
        full[s] = sum(1 for l in open(f) if l.strip()) if f.exists() else None
    for s in SETS:
        row(s, [fmt(full[s], 0) if full[s] is not None else "-", str(len(ids[s]))])
    row("total", [str(sum(v for v in full.values() if v)) if all(full.values()) else "-", str(sum(len(v) for v in ids.values()))])


if __name__ == "__main__":
    what = sys.argv[1:] or ["main", "ablation", "transfer", "tokens", "table_qa", "fullset", "sets"]
    for w in what:
        {"main": main_table, "ablation": ablation, "transfer": transfer, "tokens": tokens, "table_qa": table_qa,
         "fullset": fullset_table, "sets": sets}[w]()
