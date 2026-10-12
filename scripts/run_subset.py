#!/usr/bin/env python
"""Run configurations of solve.py on the evaluation subset, several (model, preset) pairs over several GPUs.

Every (model, preset) pair launches ``solve.py`` in a subprocess with the preset's knobs in the environment,
on the first free GPU, and collects the metrics.

    python scripts/run_subset.py --gpus 0 1 2                                   # nine models x four arms
    python scripts/run_subset.py --gpus 0 --models qwen3-4b-2507 --presets ccs   # the paper's system
    python scripts/run_subset.py --gpus 0 --presets markdown_protocol jsonl_protocol html_own_protocol csv_protocol
    python scripts/run_subset.py --gpus 0 --models phi-4-mini --presets tags_only --limit 4   # smoke test

Arms of the ablation / transfer tables (cumulative; all greedy, no fine-tuning, every call sees every cell):
  baseline        the prompt exactly as published by MMTU / NIAT
  tags_only       only the table changes: every cell re-emitted as a tag
  tags_protocol   + the answer protocol (column codes, repeated query, answer format, hints, lookup retry)
  full            + grammar-constrained decoding of the fields copied out of the table

Results: results/subset/<model>__<preset>.json, logs in results/subset/logs/, raw outputs in
results/subset/outputs/, table in results/subset/summary.tsv.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ccs.models import TRANSFER_MODELS, get_model  # noqa: E402
from ccs.presets import ARMS, PRESETS  # noqa: E402

OUT_DIR = Path(os.environ.get("CCS_SUBSET_OUT") or ROOT / "results" / "subset")
LOG_DIR = OUT_DIR / "logs"


def job_env(model, name, gpus, limit, sets):
    spec = get_model(model)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["CCS_MODEL"] = model
    env["CCS_PRESET"] = name
    env["CCS_MAX_MODEL_LEN"] = str(spec.long_max_len)
    env["CCS_OUT"] = str(OUT_DIR / "outputs" / f"{model}__{name}.jsonl")
    if limit:
        env["CCS_LIMIT"] = str(limit)
    if sets:
        env["CCS_SETS"] = " ".join(sets)
    return env


def busy_gpus(threshold_mib=8000):
    """GPUs on which another process already holds memory (waited for with --wait-busy)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(i) for i, m in (l.split(",") for l in out.strip().splitlines()) if int(m) > threshold_mib}
    except Exception:
        return set()


def parse_metrics(log_path):
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        if line.startswith("metrics_json:"):
            return json.loads(line[len("metrics_json:"):])
    return None


def write_summary():
    rows = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("*__*.json"))]
    if not rows:
        return
    order = {m: i for i, m in enumerate(TRANSFER_MODELS)}
    rows.sort(key=lambda r: (order.get(r.get("model"), 99), r.get("preset", "")))
    cols = ["model", "preset", "score", "mmtu_acc", "mmtu_niht_acc", "mmtu_table_locate_acc", "niat_em",
            "niat_cell_locating_em", "niat_cell_lookup_em", "niat_cropped_em", "n_prompts", "answered", "minutes"]
    with open(OUT_DIR / "summary.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + "\n")
    print(f"[subset] wrote {OUT_DIR / 'summary.tsv'} ({len(rows)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=TRANSFER_MODELS)
    ap.add_argument("--presets", nargs="+", default=ARMS, help=f"any of {sorted(PRESETS)}")
    ap.add_argument("--sets", nargs="+", default=None, help="prompt sets (default: all five)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N prompts per set")
    ap.add_argument("--force", action="store_true", help="re-run pairs that already have a result")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--wait-busy", action="store_true", help="start on a GPU only once no other process holds memory on it")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        return write_summary()

    queue = []
    for name in args.presets:
        for model in args.models:
            tag = f"{model}__{name}"
            if not args.force and not args.limit and (OUT_DIR / f"{tag}.json").exists():
                print(f"[subset] skip {tag} (done)")
                continue
            queue.append((model, name, get_model(model).tensor_parallel))
    queue.sort(key=lambda j: -j[2])   # multi-GPU jobs first
    print(f"[subset] {len(queue)} runs on GPUs {args.gpus}")

    free = list(args.gpus)
    running = []
    while queue or running:
        progress = True
        while progress:
            progress = False
            busy = busy_gpus() if args.wait_busy else set()
            idle = [g for g in free if g not in busy]
            for i, (model, name, tp) in enumerate(queue):
                if len(idle) >= tp:
                    gpus = idle[:tp]
                    for g in gpus:
                        free.remove(g)
                    queue.pop(i)
                    log = LOG_DIR / f"{model}__{name}.log"
                    fh = open(log, "w")
                    proc = subprocess.Popen([sys.executable, "solve.py"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                            env=job_env(model, name, gpus, args.limit, args.sets))
                    print(f"[subset] start {model}__{name} on GPU {gpus} (pid {proc.pid})", flush=True)
                    running.append([proc, model, name, gpus, log, time.time(), fh])
                    progress = True
                    break
        time.sleep(20)
        for job in list(running):
            proc, model, name, gpus, log, t0, fh = job
            if proc.poll() is None:
                continue
            running.remove(job)
            fh.close()
            free.extend(gpus)
            mins = (time.time() - t0) / 60
            m = parse_metrics(log)
            if proc.returncode != 0 or m is None:
                print(f"[subset] FAIL {model}__{name} rc={proc.returncode} after {mins:.1f} min -> {log}", flush=True)
                continue
            m.update(model=model, preset=name, minutes=round(mins, 1), limit=args.limit)
            if not args.limit:
                (OUT_DIR / f"{model}__{name}.json").write_text(json.dumps(m, indent=2))
            print(f"[subset] done  {model}__{name}  score={m['score']:.4f} mmtu={m['mmtu_acc']:.4f} niat={m['niat_em']:.4f} ({mins:.1f} min)", flush=True)
            write_summary()
    write_summary()


if __name__ == "__main__":
    main()
