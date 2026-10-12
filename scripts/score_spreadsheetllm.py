#!/usr/bin/env python
"""Score SpreadsheetLLM runs with their answers mapped to the task's vocabulary.

SpreadsheetLLM answers in sheet coordinates (column letters, sheet rows). Before scoring, a column letter is
translated to the header name of that column and a sheet row to the table's row_idx (needle task); on the
table-locate task the requested row_idx is read as a sheet row (the cell two rows above). The raw score, the
mapped score and their union are printed; the paper reports the mapped score.

    python scripts/score_spreadsheetllm.py results/subset/outputs/qwen3-4b-2507__spreadsheetllm.jsonl
    python scripts/score_spreadsheetllm.py results/subset/outputs/qwen3-4b-2507__spreadsheetllm_inv.jsonl
    python scripts/score_spreadsheetllm.py --full results/fullset/qwen3-4b-2507__spreadsheetllm
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark  # noqa: E402
from ccs.niat_data import load_prompt_records  # noqa: E402
from ccs.utils import read_jsonl  # noqa: E402
from solve import find_markdown_table  # noqa: E402

_LOCQ = re.compile(r'row_idx: (\d+)\n\ncolumn name: "(.*?)"\n\nTable:', re.S)


def _letter_to_idx(s):
    n = 0
    for ch in s.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def mapped_score(set_name, rec, text, union=False):
    """The answer with sheet coordinates translated to the task's vocabulary, then scored. With ``union`` an
    answer also counts when it is correct as written (before the translation)."""
    if not set_name.startswith("mmtu"):
        return benchmark.score_one(set_name, rec, text)   # NIAT answers are cell values: no coordinates to map
    if union and benchmark.score_one(set_name, rec, text):
        return 1
    found = find_markdown_table(rec["prompt"])
    if not found:
        return 0
    rows = found[2]
    header, body = rows[0], rows[1:]
    if set_name == "mmtu_niht":
        m = re.search(r"\{[^{}]*\}", text or "")
        try:
            ans = json.loads(m.group(0))
        except Exception:
            return 0
        col, row = str(ans.get("column", "")), ans.get("row", None)
        if re.fullmatch(r"[A-Za-z]{1,3}", col) and col.lower() not in [h.lower() for h in header]:
            j = _letter_to_idx(col)
            col = header[j] if 0 <= j < len(header) else col
        try:
            k = int(row)
            if 2 <= k <= len(body) + 1 and header[0] == "row_idx":
                row = int(body[k - 2][0])
        except Exception:
            pass
        return benchmark.score_one(set_name, rec, json.dumps({"row": row, "column": col}))
    if set_name == "mmtu_table_locate":
        m = _LOCQ.search(rec["prompt"])
        if not m or m.group(2) not in header:
            return 0
        j = header.index(m.group(2))
        pos = next((i for i, b in enumerate(body) if b[0] == m.group(1)), None)
        if pos is None or pos - 2 < 0:
            return 0
        return benchmark.score_one(set_name, dict(rec, gt=body[pos - 2][j]), text)
    return 0


def main():
    full = "--full" in sys.argv
    path = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
    if full:
        data = {s: load_prompt_records(ROOT / "data" / "processed" / f"{s}.jsonl") for s in benchmark.SETS}
        outs = {}
        for s in benchmark.SETS:
            f = path / f"outputs_{s}.jsonl"
            if f.exists():
                outs.update({r["id"]: r["text"] for r in read_jsonl(f)})
    else:
        data = benchmark.load()
        outs = {r["id"]: r["text"] for r in read_jsonl(path)}
    raw, mapped, union = {}, {}, {}
    for s, recs in data.items():
        if not any(r["id"] in outs for r in recs):
            continue
        raw[s] = sum(benchmark.score_one(s, r, outs.get(r["id"], "")) for r in recs) / len(recs)
        mapped[s] = sum(mapped_score(s, r, outs.get(r["id"], "")) for r in recs) / len(recs)
        union[s] = sum(mapped_score(s, r, outs.get(r["id"], ""), union=True) for r in recs) / len(recs)
    print(f"{'set':22s}{'raw':>8s}{'mapped':>8s}{'union':>8s}")
    for s in raw:
        print(f"{s:22s}{100 * raw[s]:8.1f}{100 * mapped[s]:8.1f}{100 * union[s]:8.1f}")
    if len(raw) == 5:
        print(f"score: raw {benchmark.aggregate(raw)['score']:.3f}  mapped {benchmark.aggregate(mapped)['score']:.3f}  union {benchmark.aggregate(union)['score']:.3f}")


if __name__ == "__main__":
    main()
