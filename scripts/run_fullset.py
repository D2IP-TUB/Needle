#!/usr/bin/env python
"""Run one configuration of solve.py on the FULL prompt sets (data/processed).

    CUDA_VISIBLE_DEVICES=0 python scripts/run_fullset.py                          # CCS, all five sets
    CUDA_VISIBLE_DEVICES=1 python scripts/run_fullset.py mmtu_niht mmtu_table_locate
    CCS_PRESET=jsonl_protocol CUDA_VISIBLE_DEVICES=2 python scripts/run_fullset.py
    CCS_PRESET=ccs_colword CUDA_VISIBLE_DEVICES=0 python scripts/run_fullset.py mmtu_niht
    python scripts/run_fullset.py --limit 4                                       # smoke test (not a valid score)

Outputs go to results/fullset/<model>__<preset>/ (override with CCS_FULL_OUT): outputs_<set>.jsonl and
metrics_<set>.json per set, with a checkpoint after every batch so that a killed job resumes. Aggregate the
five sets the paper's way with scripts/score_fullset.py.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import benchmark  # noqa: E402
import solve  # noqa: E402
from ccs.niat_data import load_prompt_records  # noqa: E402
from ccs.utils import write_jsonl  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sets", nargs="*", default=benchmark.SETS, help="prompt sets (default: all five)")
ap.add_argument("--limit", type=int, default=0, help="smoke test: first N prompts per set (not a valid score)")
cli = ap.parse_args()
sets, LIMIT = cli.sets or benchmark.SETS, cli.limit
OUT = Path(os.environ.get("CCS_FULL_OUT") or ROOT / "results" / "fullset" / f"{solve.MODEL}__{solve.PRESET}{'_limit' if LIMIT else ''}")
OUT.mkdir(parents=True, exist_ok=True)
print("[full] config:", solve.describe(), "->", OUT, flush=True)

t0 = time.time()
for s in sets:
    recs = load_prompt_records(ROOT / "data" / "processed" / f"{s}.jsonl")
    if LIMIT:
        recs = recs[:LIMIT]
    part = OUT / f"partial_{s}.jsonl"
    texts = [json.loads(l)["text"] for l in part.read_text().splitlines()] if part.exists() else []
    if texts:
        print(f"[full] {s}: resuming from {len(texts)}/{len(recs)}", flush=True)
    for k in range(len(texts), len(recs), 4096):
        batch = solve.predict([r["prompt"] for r in recs[k:k + 4096]], benchmark.MAX_TOKENS[s], s)
        texts.extend(batch)
        with part.open("a") as f:
            f.writelines(json.dumps({"id": r["id"], "text": t or ""}, ensure_ascii=False) + "\n" for r, t in zip(recs[k:k + 4096], batch))
        print(f"[full] {s}: {min(k + 4096, len(recs))}/{len(recs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    write_jsonl(OUT / f"outputs_{s}.jsonl", [{"id": r["id"], "text": t or ""} for r, t in zip(recs, texts)])
    part.unlink(missing_ok=True)
    correct = sum(benchmark.score_one(s, r, t) for r, t in zip(recs, texts))
    m = {"set": s, "n": len(recs), "correct": correct, "acc": correct / len(recs),
         "answered": sum(1 for t in texts if t), "minutes": round((time.time() - t0) / 60, 1),
         "model": solve.MODEL, "preset": solve.PRESET, "config": solve.describe()}
    (OUT / f"metrics_{s}.json").write_text(json.dumps(m, indent=1))
    print(f"[full] {s}: n={len(recs)} acc={m['acc']:.4f} answered={m['answered']}", flush=True)
print("[full] done", flush=True)
