"""MMTU needle-in-a-table tasks.

The prompts are taken verbatim from the MMTU Hugging Face dataset (``MMTU-benchmark/MMTU``, split
``train``):

* ``Table-needle-in-a-haystack`` (value -> position: "given a unique value, return its row index and
  column name") -> task key ``niht``
* ``Table-Locate-by-Row-Col`` (position -> value: "given row index and column name, return the cell
  value") -> task key ``table_locate``

All tables are 50 rows x 50 columns (plus a ``row_idx`` column); the needle position is uniform over
the grid.
"""

import json
from typing import Optional

import pandas as pd

from .utils import write_jsonl

HF_DATASET = "MMTU-benchmark/MMTU"

TASKS = {
    "niht": "Table-needle-in-a-haystack",
    "table_locate": "Table-Locate-by-Row-Col",
}


def load_mmtu_frame() -> pd.DataFrame:
    """The MMTU dataset as a DataFrame (downloaded to the Hugging Face cache on first use)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    return ds.to_pandas()


def build_mmtu(out_dir, limit: Optional[int] = None, seed: int = 0) -> dict:
    """Materialise ``{out_dir}/mmtu_{task}.jsonl``; each record has ``id, task, prompt, gt, meta``."""
    df = load_mmtu_frame()
    summary = {}
    for key, task_name in TASKS.items():
        sub = df[df.task == task_name].reset_index(drop=True)
        if limit:
            sub = sub.sample(n=min(limit, len(sub)), random_state=seed).reset_index(drop=True)
        rows = []
        for _, r in sub.iterrows():
            md = json.loads(r.metadata)
            if key == "niht":
                gt = {"row": int(md["needle_row_idx"]), "column": str(md["needle_col_name"])}
            else:
                gt = str(md["needle_value"])
            rows.append({
                "id": f"mmtu/{key}/{md['test_case']}",
                "task": key,
                "prompt": r.prompt,
                "gt": gt,
                "meta": {
                    "test_case": md["test_case"],
                    "needle_row_idx": int(md["needle_row_idx"]),
                    "needle_col_idx": int(md["needle_col_idx"]),
                    "needle_col_name": str(md["needle_col_name"]),
                    "needle_value": str(md["needle_value"]),
                },
            })
        path = f"{out_dir}/mmtu_{key}.jsonl"
        write_jsonl(path, rows)
        summary[key] = len(rows)
        print(f"[mmtu] {task_name}: {len(rows)} prompts -> {path}")
    return summary
