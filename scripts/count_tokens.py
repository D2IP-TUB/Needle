#!/usr/bin/env python
"""Prompt length of the published prompts and of their CCS form, in Qwen3 tokens, on the evaluation subset
(the prompt-length table of the paper).

    python scripts/count_tokens.py                      # the default configuration
    CCS_PRESET=full python scripts/count_tokens.py      # per-task tag shapes

Writes results/tokens_<preset>.json and prints the table rows (mean published, mean CCS, max CCS).
"""

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark  # noqa: E402
import solve  # noqa: E402
from ccs.models import get_model  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(get_model(solve.MODEL).hf_id)


def n_tokens(p):
    return len(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True))


LABEL = {"mmtu_niht": "MMTU value -> position", "mmtu_table_locate": "MMTU position -> value",
         "niat_cell_locating": "NIAT position -> value", "niat_cropped": "NIAT position -> value (crop)",
         "niat_cell_lookup": "NIAT question -> value"}
out = {}
for s, recs in benchmark.load().items():
    a = [n_tokens(r["prompt"]) for r in recs]
    b = [n_tokens(solve.transform(r["prompt"], s)) for r in recs]
    out[s] = {"n": len(recs), "orig_mean": st.mean(a), "new_mean": st.mean(b), "ratio_mean": st.mean(y / x for x, y in zip(a, b)),
              "orig_max": max(a), "new_max": max(b)}
    print(f"{LABEL[s]:34s} {out[s]['orig_mean'] / 1000:6.1f}k {out[s]['new_mean'] / 1000:6.1f}k {out[s]['new_max'] / 1000:7.1f}k   (ratio {out[s]['ratio_mean']:.2f})", flush=True)
json.dump({"model": solve.MODEL, "preset": solve.PRESET, "config": solve.describe(), "sets": out},
          open(ROOT / "results" / f"tokens_{solve.PRESET}.json", "w"), indent=2)
