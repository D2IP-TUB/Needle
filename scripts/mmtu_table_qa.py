#!/usr/bin/env python
"""MMTU's Table-QA task (1,793 questions from FinQA, TableBench and WikiQA) with the table written as Markdown
(as MMTU renders it), as JSON rows, or as CCS tags; scored with a port of MMTU's TQAEvaluator (task score =
mean of the three source accuracies).

Variants (prefix = prompt protocol, suffix = table serialization):
  mmtu_markdown / mmtu_json / mmtu_tagged        MMTU's prompt verbatim ("no explanation, JSON answer")
  reason_markdown / reason_json / reason_tagged  the no-explanation instruction replaced by brief reasoning
                                                 before the JSON line (the setting of the paper's table)
Decoding is greedy; budgets 1,024 tokens for the verbatim prompt, 2,048 for the reasoning variants.

    CUDA_VISIBLE_DEVICES=0 python scripts/mmtu_table_qa.py
    CUDA_VISIBLE_DEVICES=0 python scripts/mmtu_table_qa.py --variants reason_markdown reason_json reason_tagged

Writes results/table_qa/mmtu_tqa_<model>_<variant>.jsonl (per question) and mmtu_tqa_<model>_summary.json.
"""

import argparse
import ast
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccs.engine import Engine  # noqa: E402
from ccs.table_qa import SERIALIZERS, find_markdown_table, generate  # noqa: E402

NO_EXPLAIN = ('4. Format the Answer: No explanation, return the answer in a structured JSON format: {"answer": <answer>}. '
              'If the question cannot be answered due to missing or ambiguous data, return a JSON response with {"answer": null}.')
REASON = ('4. Format the Answer: Reason briefly first, without using curly braces, then end with one line holding the '
          'answer in a structured JSON format: {"answer": <answer>}. If the question cannot be answered due to missing '
          'or ambiguous data, end with {"answer": null}.')


def load_mmtu_tqa(limit):
    from datasets import load_dataset
    ds = load_dataset("MMTU-benchmark/MMTU", split="train").filter(lambda x: x["task"] == "Table-QA")
    recs = []
    for ex in ds:
        m = json.loads(ex["metadata"])
        recs.append({"id": f"{ex['dataset']}/{m['test_case']}", "dataset": ex["dataset"], "prompt": ex["prompt"], "label": m["label"]})
    if limit:  # a per-dataset head, so that every source is represented in a pilot
        seen = defaultdict(int)
        recs = [r for r in recs if (seen.__setitem__(r["dataset"], seen[r["dataset"]] + 1) or seen[r["dataset"]] <= limit)]
    return recs


def build_prompt(rec, variant):
    p = rec["prompt"]
    assert NO_EXPLAIN in p, rec["id"]
    if variant.startswith("reason"):
        p = p.replace(NO_EXPLAIN, REASON, 1)
    ser = next((f for k, f in SERIALIZERS.items() if k in variant), None)
    if ser is not None:
        found = find_markdown_table(p)
        if found is not None and len(found[2]) >= 2:
            s, e, rows = found
            p = p[:s] + ser(rows) + p[e:]
    return p


# ---- MMTU's TQAEvaluator, ported verbatim (exceptions count as wrong, as in their evaluate.py) ----
def extract_json_answer(text, key="answer"):
    if text.startswith("<think>"):
        m = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        text = m.group(2) if m else text
    for match in re.findall(r"{[^}]*}", text):
        for loader in (json.loads, ast.literal_eval):
            try:
                result = loader(match)
                if isinstance(result, dict) and key in result:
                    return result[key]
            except Exception:
                continue
    return "JSONParsingError"


def mmtu_correct(y_true, y_pred):
    try:
        if str(y_true).lower() == str(y_pred).lower():
            return 1
        if isinstance(y_pred, str):
            if "%" in y_pred:
                y_pred = y_pred.replace("%", "")
                if round(float(y_true), 2) == round(float(y_pred), 2):
                    return 1
                y_pred = float(y_pred) / 100
                if round(float(y_true), 2) == round(float(y_pred), 2):
                    return 1
            elif "$" in y_pred:
                y_pred = y_pred.replace("$", "").replace(",", "")
                if round(float(y_true), 2) == round(float(y_pred), 2):
                    return 1
        else:
            if round(float(y_true), 2) == round(float(y_pred), 2):
                return 1
    except Exception:
        return 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["mmtu_markdown", "mmtu_json", "mmtu_tagged",
                                                      "reason_markdown", "reason_json", "reason_tagged"])
    ap.add_argument("--limit", type=int, default=None, help="questions per source dataset (pilot)")
    ap.add_argument("--model", default="qwen3-4b-2507")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "table_qa"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_mmtu_tqa(args.limit)
    print(f"[table-qa] {len(recs)} questions", flush=True)
    engine = Engine(args.model, long_context=True, max_model_len=131072)
    summary_path = out / f"mmtu_tqa_{args.model}_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for variant in args.variants:
        t0 = time.time()
        prompts = [build_prompt(r, variant) for r in recs]
        texts = generate(engine, prompts, 2048 if variant.startswith("reason") else 1024)
        rows, by_ds = [], defaultdict(list)
        for r, t in zip(recs, texts):
            pred = extract_json_answer(t)
            c = mmtu_correct(r["label"], pred)
            by_ds[r["dataset"]].append(c)
            rows.append({"id": r["id"], "dataset": r["dataset"], "label": r["label"], "pred": pred, "correct": c, "text": t})
        per = {d: sum(v) / len(v) for d, v in sorted(by_ds.items())}
        summary[variant] = {"per_dataset": per, "task_score": sum(per.values()) / len(per),
                            "json_error": sum(1 for x in rows if x["pred"] == "JSONParsingError"), "minutes": (time.time() - t0) / 60}
        print(f"[table-qa] {variant}: " + " ".join(f"{d}={a:.3f}" for d, a in per.items()) + f" task={summary[variant]['task_score']:.3f}"
              f" (json errors {summary[variant]['json_error']}, {summary[variant]['minutes']:.1f} min)", flush=True)
        with open(out / f"mmtu_tqa_{args.model}_{variant}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        json.dump(summary, open(summary_path, "w"), indent=2)


if __name__ == "__main__":
    main()
