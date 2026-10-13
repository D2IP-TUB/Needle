"""Evaluation of the two MMTU needle tasks.

Faithful re-implementation of ``evaluators/tniah_evaluator.py`` and ``evaluators/tablelocate_evaluator.py``
of the MMTU repository (https://github.com/MMTU-Benchmark/MMTU):

* needle (value -> position, MMTU task ``Table-needle-in-a-haystack``): answer JSON ``{"row": r, "column": c}``;
  ``row`` correct if the ``int`` values match, ``column`` correct if the case-insensitive stripped strings
  match, ``correct`` = both.
* table-locate (position -> value, MMTU task ``Table-Locate-by-Row-Col``): answer JSON ``{"cell": v}``;
  correct if the strings match or both parse to equal floats.

Position bins follow MMTU's ``round_by_interval(x, 5)``: index 1..5 -> 5, 6..10 -> 10, ...
"""

from .utils import extract_json_answer


def bin5(x: int) -> int:
    return (int((x - 1) / 5) + 1) * 5


def score_niht(gt: dict, pred) -> dict:
    if not isinstance(pred, dict):
        return {"correct": 0, "row_correct": 0, "col_correct": 0}
    try:
        row_correct = int(int(gt["row"]) == int(pred["row"]))
    except Exception:
        row_correct = 0
    col_correct = int(str(gt["column"]).strip().lower() == str(pred.get("column", "")).strip().lower())
    return {"correct": int(row_correct and col_correct), "row_correct": row_correct, "col_correct": col_correct}


def score_table_locate(gt, pred) -> dict:
    correct = int(str(gt) == str(pred))
    if not correct:
        try:
            correct = int(float(gt) == float(pred))
        except Exception:
            pass
    return {"correct": correct}


def parse_niht(text: str):
    """The ``{"row": ..., "column": ...}`` answer of the needle task (MMTU takes the first JSON object)."""
    pred = {}
    for k in ("row", "column"):
        v = extract_json_answer(text, k, last=False)
        if v == "JSONParsingError":
            return "JSONParsingError"
        pred[k] = v
    return pred
