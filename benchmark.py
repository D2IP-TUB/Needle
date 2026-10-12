"""The five cell-retrieval prompt sets, the fixed evaluation subset and the scoring.

Prompt sets (full sizes are built by ``scripts/build_data.py`` into ``data/processed``):

    mmtu_niht            MMTU needle, value -> position          999 prompts
    mmtu_table_locate    MMTU table-locate, position -> value   1,000
    niat_cell_locating   NIAT cell-locating, position -> value  14,396
    niat_cropped         NIAT fixed-size tables 8x8 .. 32x32    48,720
    niat_cell_lookup     NIAT cell-lookup, question -> value    14,054

The evaluation subset (7,996 prompts, ``data/benchmark``) is a seeded sample of the full sets: 150 prompts
per MMTU task, 600 per NIAT task, and every cell of two tables per size for the cropped set. It is used for
the ablation and the model-transfer runs. ``data/subset_ids.json`` pins its prompt ids; ``build`` checks the
rebuilt subset against it.

Metrics are the ones the benchmarks report: MMTU accuracy with the MMTU evaluators, NIAT exact match.
``score = (mmtu_acc + niat_em) / 2`` with ``mmtu_acc`` the mean of the two MMTU tasks and ``niat_em`` the mean
of the three NIAT tasks. Every unanswered, unparsable or context-overflow prompt counts as wrong.

    python benchmark.py --build                 # one-time data prep (downloads MMTU on first use)
    python benchmark.py --score outputs.jsonl   # score a file of {"id": ..., "text": ...} lines
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ccs.mmtu_eval import parse_niht, score_niht, score_table_locate  # noqa: E402
from ccs.niat_eval import ANSWER_KEY, score as niat_score  # noqa: E402
from ccs.utils import extract_json_answer, read_jsonl, write_jsonl  # noqa: E402

SEED = 0
BENCH_DIR = ROOT / "data" / "benchmark"
NIAT_SOURCE_DIR = ROOT / "data" / "niat"
SUBSET_IDS = ROOT / "data" / "subset_ids.json"

N_MMTU_PER_TASK = 150
N_NIAT_CELL_LOCATING = 600
N_NIAT_CELL_LOOKUP = 600
CROP_TABLES_PER_SIZE = 2       # x 7 sizes, every cell queried -> 6,496 prompts
CROP_SIZES = [8, 12, 16, 20, 24, 28, 32]

# generation budgets (tokens)
MAX_TOKENS = {
    "mmtu_niht": 1024,
    "mmtu_table_locate": 512,
    "niat_cell_locating": 2048,
    "niat_cell_lookup": 2048,
    "niat_cropped": 2048,
}
SETS = list(MAX_TOKENS)
METRIC_NAMES = ["mmtu_niht_acc", "mmtu_table_locate_acc", "mmtu_acc",
                "niat_cell_locating_em", "niat_cell_lookup_em", "niat_cropped_em", "niat_em", "score"]


# -----------------------------------------------------------------------------
# Building the evaluation subset
# -----------------------------------------------------------------------------
def build(force: bool = False) -> dict:
    """Materialise the evaluation subset into data/benchmark/<set>.jsonl."""
    from ccs.mmtu_data import TASKS as MMTU_TASKS, load_mmtu_frame
    from ccs.niat_data import (build_cell_locating, build_cell_lookup, build_cropped,
                               load_cropped_samples, load_niat_tables)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all((BENCH_DIR / f"{s}.jsonl").exists() for s in SETS):
        return {s: len(read_jsonl(BENCH_DIR / f"{s}.jsonl")) for s in SETS}

    rng = random.Random(SEED)
    counts = {}

    # MMTU: verbatim prompts from the HF dataset, seeded sample per task
    df = load_mmtu_frame()
    for key, task_name in MMTU_TASKS.items():
        sub = df[df.task == task_name].sample(n=N_MMTU_PER_TASK, random_state=SEED)
        rows = []
        for _, r in sub.iterrows():
            md = json.loads(r.metadata)
            gt = ({"row": int(md["needle_row_idx"]), "column": str(md["needle_col_name"])}
                  if key == "niht" else str(md["needle_value"]))
            rows.append({"id": f"mmtu/{key}/{md['test_case']}", "task": key, "prompt": r.prompt, "gt": gt,
                         "meta": {"needle_row_idx": int(md["needle_row_idx"]),
                                  "needle_col_idx": int(md["needle_col_idx"])}})
        write_jsonl(BENCH_DIR / f"mmtu_{key}.jsonl", rows)
        counts[f"mmtu_{key}"] = len(rows)

    # NIAT: regenerate the full sets with the same seed as the main experiments, then sample
    tables = load_niat_tables(NIAT_SOURCE_DIR)
    loc = build_cell_locating(tables, cells_per_table=20, seed=SEED)
    look = build_cell_lookup(tables, questions_per_table=20, seed=SEED)
    table_store = {}  # cropped tables are stored once; prompts are rebuilt on load
    crop = build_cropped(tables, load_cropped_samples(NIAT_SOURCE_DIR), CROP_TABLES_PER_SIZE, CROP_SIZES, SEED,
                         table_store=table_store)
    json.dump(table_store, open(BENCH_DIR / "niat_cropped_tables.json", "w"))
    for name, recs, n in [("niat_cell_locating", loc, N_NIAT_CELL_LOCATING),
                          ("niat_cell_lookup", look, N_NIAT_CELL_LOOKUP),
                          ("niat_cropped", crop, None)]:
        if n is not None and len(recs) > n:
            recs = rng.sample(recs, n)
            recs.sort(key=lambda r: r["meta"]["table_id"])  # group by table for prefix caching
        write_jsonl(BENCH_DIR / f"{name}.jsonl", recs)
        counts[name] = len(recs)
    json.dump(counts, open(BENCH_DIR / "counts.json", "w"), indent=2)
    check_ids()
    return counts


def check_ids():
    """The built subset must hold exactly the prompt ids of the paper's runs (data/subset_ids.json)."""
    if not SUBSET_IDS.exists():
        return
    expected = json.load(open(SUBSET_IDS))
    for s in SETS:
        got = [r["id"] for r in read_jsonl(BENCH_DIR / f"{s}.jsonl")]
        if sorted(got) != sorted(expected[s]):
            raise RuntimeError(f"{s}: the rebuilt subset differs from data/subset_ids.json "
                               f"({len(got)} vs {len(expected[s])} ids, {len(set(got) & set(expected[s]))} shared)")
    print("[benchmark] subset ids match data/subset_ids.json")


def load(sets=None) -> dict:
    """{set_name: [records]}; each record has id, prompt, gt, meta."""
    sets = sets or SETS
    missing = [s for s in sets if not (BENCH_DIR / f"{s}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"benchmark data missing for {missing}; run `python benchmark.py --build`")
    from ccs.niat_data import load_prompt_records
    return {s: load_prompt_records(BENCH_DIR / f"{s}.jsonl") for s in sets}


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def score_one(set_name: str, record: dict, text: str) -> int:
    """1 if the model output ``text`` is correct for ``record`` under the benchmark's metric."""
    text = text or ""
    if set_name == "mmtu_niht":
        return score_niht(record["gt"], parse_niht(text))["correct"]
    if set_name == "mmtu_table_locate":
        return score_table_locate(record["gt"], extract_json_answer(text, "cell", last=False))["correct"]
    task = set_name[len("niat_"):]
    pred = extract_json_answer(text, ANSWER_KEY[task], last=True)
    return niat_score(record["gt"], pred)["em"]


def aggregate(per_set: dict) -> dict:
    """The paper's metrics from per-set accuracies (missing sets give NaN)."""
    m = {
        "mmtu_niht_acc": per_set.get("mmtu_niht", float("nan")),
        "mmtu_table_locate_acc": per_set.get("mmtu_table_locate", float("nan")),
        "niat_cell_locating_em": per_set.get("niat_cell_locating", float("nan")),
        "niat_cell_lookup_em": per_set.get("niat_cell_lookup", float("nan")),
        "niat_cropped_em": per_set.get("niat_cropped", float("nan")),
    }
    m["mmtu_acc"] = (m["mmtu_niht_acc"] + m["mmtu_table_locate_acc"]) / 2
    m["niat_em"] = (m["niat_cell_locating_em"] + m["niat_cell_lookup_em"] + m["niat_cropped_em"]) / 3
    m["score"] = (m["mmtu_acc"] + m["niat_em"]) / 2
    return m


def score(data: dict, outputs: dict) -> dict:
    """``data``: {set: [records]}, ``outputs``: {id: text}. Missing ids count as wrong."""
    per_set = {}
    for s, recs in data.items():
        correct = [score_one(s, r, outputs.get(r["id"], "")) for r in recs]
        per_set[s] = sum(correct) / max(1, len(correct))
    m = aggregate(per_set)
    m["n_prompts"] = sum(len(v) for v in data.values())
    m["answered"] = sum(1 for v in data.values() for r in v if outputs.get(r["id"]))
    return m


def evaluate(predict, sets=None, save_outputs=None) -> dict:
    """Run ``predict(prompts: list[str], max_tokens: int, set_name: str) -> list[str]`` on every set of the
    evaluation subset and return the metrics. ``predict`` returns one completion string per prompt, in order;
    "" for anything it cannot answer (e.g. a prompt longer than the model's context)."""
    data = load(sets)
    outputs = {}
    t0 = time.time()
    for s, recs in data.items():
        texts = predict([r["prompt"] for r in recs], MAX_TOKENS[s], s)
        assert len(texts) == len(recs), f"predict returned {len(texts)} outputs for {len(recs)} prompts ({s})"
        outputs.update({r["id"]: (t or "") for r, t in zip(recs, texts)})
    metrics = score(data, outputs)
    metrics["eval_seconds"] = time.time() - t0
    if save_outputs:
        write_jsonl(save_outputs, [{"id": k, "text": v} for k, v in outputs.items()])
    return metrics


def print_summary(metrics: dict):
    print("---")
    for k in METRIC_NAMES:
        print(f"{k + ':':24s}{metrics[k]:.6f}")
    for k in ("n_prompts", "answered"):
        if k in metrics:
            print(f"{k + ':':24s}{metrics[k]}")
    if "eval_seconds" in metrics:
        print(f"{'eval_seconds:':24s}{metrics['eval_seconds']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the evaluation subset")
    ap.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    ap.add_argument("--score", type=str, help="JSONL of {id, text} to score against the evaluation subset")
    args = ap.parse_args()

    if args.build or not all((BENCH_DIR / f"{s}.jsonl").exists() for s in SETS):
        counts = build(force=args.force)
        print("benchmark sets:", json.dumps(counts))
    if args.score:
        outputs = {r["id"]: r.get("text", "") for r in read_jsonl(args.score)}
        print_summary(score(load(), outputs))
