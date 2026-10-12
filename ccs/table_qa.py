"""Table serializations and generation helper for the MMTU Table-QA experiment.

The Table-QA prompts carry a Markdown table; the experiment swaps it for JSON rows or for CCS tags. The
tag used here is the column-name-carrying form ``[r<row>|c<n>|<column name>=<value>]`` with a
``columns (name -> code)`` list, rows numbered from 1 for the first data row.
"""

from __future__ import annotations

import json
import re
from typing import Iterable


def _clean(v) -> str:
    return re.sub(r"\s+", " ", str(v if v is not None else "")).strip()


TAG_NOTE = ("(The table is written one row per line as key=value pairs; the 'columns:' list maps each column "
            "code c<n> to the column name, and every cell is written as [r<row>|c<n>|column name=value], "
            "i.e. the row number, the column code and the column name of a value are inside the same "
            "brackets as the value. Row numbers start at 1 for the first data row.)")


def to_tagged(g) -> str:
    header, body = g[0], g[1:]
    out = [TAG_NOTE, "columns (name -> code):"]
    out += [f"  {h} -> c{j + 1}" for j, h in enumerate(header)]
    out.append("rows:")
    for i, r in enumerate(body):
        out.append(f"row {i + 1}: " + " ".join(f"[r{i + 1}|c{j + 1}|{header[j]}={r[j]}]" for j in range(len(header))))
    return "\n".join(out)


JSON_NOTE = "(The table is written as JSON: a list with one object per row below the header row, the keys being the header cells.)"


def to_json(g) -> str:
    """One object per data row, keys = header cells (duplicate names stay duplicate keys, in column order)."""
    d = lambda v: json.dumps(v.replace("\\|", "|"), ensure_ascii=False)  # noqa: E731
    header, body = g[0], g[1:]
    return JSON_NOTE + "\n[\n" + ",\n".join("{" + ", ".join(f"{d(k)}: {d(v)}" for k, v in zip(header, r)) + "}" for r in body) + "\n]"


SERIALIZERS = {"tagged": to_tagged, "json": to_json}

# --------------------------------------------------------------------------
# The Markdown table of a prompt
# --------------------------------------------------------------------------
_PIPE = re.compile(r"(?<!\\)\|")
_SEP_CELL = re.compile(r"^:?-{2,}:?$")


def find_markdown_table(prompt):
    """(start_char, end_char, rows) of the largest Markdown table in the prompt, separator row removed, or None."""
    lines = prompt.split("\n")
    best, cur, starts, pos = None, [], [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("|"):
            cur.append(i)
        else:
            if len(cur) >= 2 and (best is None or len(cur) > len(best)):
                best = cur
            cur = []
    if len(cur) >= 2 and (best is None or len(cur) > len(best)):
        best = cur
    if best is None:
        return None
    rows = []
    for i in best:
        cells = _PIPE.split(lines[i].strip())
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        cells = [c.strip() for c in cells]
        if cells and all(_SEP_CELL.match(c) for c in cells):
            continue  # separator row
        rows.append(cells)
    return starts[best[0]], starts[best[-1]] + len(lines[best[-1]]), rows


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generate(engine, prompts: Iterable[str], max_tokens: int, regexes: list | None = None) -> list[str]:
    """Greedy completions for ``prompts`` (one string each, "" on context overflow)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompts = list(prompts)
    ids = [engine.encode(p) for p in prompts]
    budget = engine.max_model_len - max_tokens
    keep = [i for i, x in enumerate(ids) if len(x) <= budget]
    texts = [""] * len(prompts)
    if not keep:
        return texts
    base = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    if regexes is None:
        params = base
    else:
        params = [SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0,
                                 structured_outputs=StructuredOutputsParams(regex=regexes[i])) if regexes[i] else base
                  for i in keep]
    outs = engine.llm.generate([{"prompt_token_ids": ids[i]} for i in keep], params, use_tqdm=True)
    for i, o in zip(keep, outs):
        texts[i] = o.outputs[0].text
    return texts
