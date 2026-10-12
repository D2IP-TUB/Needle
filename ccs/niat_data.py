"""NeedleInATable (NIAT) query construction.

The public repository (``wlr737/NeedleInATable``) ships the 360 source tables
with their GPT-4o-generated lookup questions, plus a *sample* of 50 prompts per
setting. The full query sets are re-generated here from the source tables with
the exact prompt templates found in the released samples.

Settings (as in Table 1 of the paper):
    structure in {flat, horizontal, hierarchical}   (flat = WTQ "vertical")
    format    in {markdown, HTML}
    task      in {cell_locating, cell_lookup}

Fixed-size ("cropped") tables (Figure 4 / Table 7): N x N Markdown tables for
N in {8,12,16,20,24,28,32}, 15 tables per size, one query per cell
(-> 15 * sum(N^2) = 48,720 queries, matching the paper's "48K").
Sources for cropped tables are the NIAT tables that are big enough
(cropped top-left); tables that are too narrow/short are extended by cycling
columns/rows, which is what the released samples exhibit (duplicated header
names).
"""

import json
import random
from pathlib import Path
from typing import Optional

from .utils import write_jsonl

STRUCTURE_MAP = {"vertical": "flat", "horizontal": "horizontal", "hierarchical": "hierarchical"}
STRUCTURES = ["flat", "horizontal", "hierarchical"]
FORMATS = ["markdown", "HTML"]
CROP_SIZES = [8, 12, 16, 20, 24, 28, 32]

MAX_TOKENS = {"cell_locating": 2048, "cell_lookup": 2048, "cropped": 2048}

# --------------------------------------------------------------------------
# Prompt templates (verbatim from NIAT_LLM_test_data samples)
# --------------------------------------------------------------------------
# NOTE: the released HTML samples use the same wording ("formatted in
# Markdown ... delimited by the | symbol") even though the table is HTML. We
# keep it verbatim for faithfulness.
CELL_LOCATING_TEMPLATE = (
    "\nYou are an expert in analyzing and extracting information from tabular data. "
    "You will be provided with a table formatted in Markdown, where each row is separated "
    "by a newline character and each column within a row is delimited by the “|” symbol. "
    "Your task is to carefully read the table and retrieve the value from a specified cell, "
    "based on a given row and column ID.\n\n"
    "Output the result as a JSON object in the following format:\n"
    "{\"row_ID\": \"m\", \"column_ID\": \"n\", \"cell_value\": \"<Corresponding Cell Value>\"}\n\n"
    "Please note:\n\n"
    "•\tRow and column IDs start from 1.\n"
    "•\tThe header row is considered the first row, with a row_ID of 1, so ensure you include it when locating the target row.\n"
    "•\tIf the target cell is empty, the cell_value should be an empty string (\"\").\n\n"
    "Given table:\n<<TABLE>>\n\n\n"
    "Now, retrieve the cell value at the position where the row_ID is <<ROW>> and the column_ID is <<COL>>.\n\n"
    "The answer is:\n"
)

# The sample prompts contain literal doubled braces "{{" / "}}" (an f-string
# artefact of the authors' generator); reproduced verbatim.
CELL_LOOKUP_TEMPLATE = (
    "Read the table carefully and answer the following question.\n\n"
    "Table:\n<<TABLE>>\n\n\n"
    "Let's first think step by step and then output a final answer in the JSON format:\n"
    "```json\n{{\n    \"answer\": \"<the final answer to the question>\"\n}}\n```\n"
    "## Question:\n<<QUESTION>>\n\n## Your Response:\n"
)


def locating_prompt(table_str: str, row: int, col: int) -> str:
    return CELL_LOCATING_TEMPLATE.replace("<<TABLE>>", table_str).replace("<<ROW>>", str(row)).replace("<<COL>>", str(col))


def lookup_prompt(table_str: str, question: str) -> str:
    return CELL_LOOKUP_TEMPLATE.replace("<<TABLE>>", table_str).replace("<<QUESTION>>", question)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------
def _clean(v) -> str:
    """Cell text as serialised inside a table (pipes escaped for Markdown)."""
    return "" if v is None else str(v).replace("\n", " ").replace("|", "\\|").strip()


def _gt(v) -> str:
    """Ground-truth cell text (no Markdown escaping)."""
    return "" if v is None else str(v).replace("\n", " ").strip()


def to_markdown_compact(rows) -> str:
    """``| a | b |`` / ``| --- | --- |`` style used by the arbitrary-size samples."""
    ncol = max(len(r) for r in rows)
    lines = ["| " + " | ".join(_clean(v) for v in list(r) + [""] * (ncol - len(r))) + " |" for r in rows]
    lines.insert(1, "| " + " | ".join(["---"] * ncol) + " |")
    return "\n".join(lines)


def to_markdown_aligned(rows) -> str:
    """Column-aligned Markdown (tabulate/pandas ``to_markdown`` style) used by
    the released cropped-table samples."""
    ncol = max(len(r) for r in rows)
    cells = [[_clean(v) for v in list(r) + [""] * (ncol - len(r))] for r in rows]
    widths = [max(4, max(len(c[j]) for c in cells)) for j in range(ncol)]
    fmt = lambda r: "| " + " | ".join(r[j].ljust(widths[j]) for j in range(ncol)) + " |"
    lines = [fmt(cells[0]), "| " + " | ".join("-" * widths[j] for j in range(ncol)) + " |"]
    lines += [fmt(c) for c in cells[1:]]
    return "\n".join(lines)


def to_html_simple(rows) -> str:
    ncol = max(len(r) for r in rows)
    out = ["<table border=\"1\" cellspacing=\"0\">"]
    for r in rows:
        r = list(r) + [""] * (ncol - len(r))
        out.append("<tr> " + " ".join(f"<td> {_clean(v)} </td>" for v in r) + " </tr>")
    out.append("</table>")
    return "\n".join(out)


def serialize(table: dict, fmt: str) -> str:
    if fmt == "markdown":
        return to_markdown_compact(table["table_rows"])
    if fmt == "HTML":
        # the source HTML keeps rowspan/colspan for hierarchical tables and the
        # original wikitable markup for WTQ, exactly as in the released samples
        return table.get("html_table") or to_html_simple(table["table_rows"])
    raise ValueError(fmt)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_niat_tables(data_dir) -> list:
    tables = json.load(open(Path(data_dir) / "360_NIAT_tables.json"))
    for t in tables:
        t["structure"] = STRUCTURE_MAP[t["table_structure"]]
        t["n_rows"] = len(t["table_rows"])
        t["n_cols"] = max(len(r) for r in t["table_rows"])
    return tables


def load_cropped_samples(data_dir) -> list:
    """Unique fixed-size tables from the released ``NIAT_cropped_tables.json``."""
    path = Path(data_dir) / "NIAT_cropped_tables.json"
    if not path.exists():
        return []
    seen, out = set(), []
    for r in json.load(open(path)):
        key = (r["table_id"], r["table_size"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"table_id": f"{r['table_id']}_{r['table_size'].replace(' ', '')}",
                    "table_rows": r["table_rows"], "source": "niat_cropped_sample"})
    return out


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------
def cell_value(rows, r1: int, c1: int) -> str:
    """1-based (row, col) lookup; header row is row 1. Missing -> ''."""
    row = rows[r1 - 1]
    return _gt(row[c1 - 1]) if c1 - 1 < len(row) else ""


def build_cell_locating(tables, cells_per_table: int, seed: int) -> list:
    rng = random.Random(seed)
    records = []
    for t in tables:
        rows = t["table_rows"]
        R, C = t["n_rows"], t["n_cols"]
        all_cells = [(r, c) for r in range(1, R + 1) for c in range(1, C + 1)]
        k = min(cells_per_table, len(all_cells)) if cells_per_table > 0 else len(all_cells)
        picked = rng.sample(all_cells, k)
        for fmt in FORMATS:
            table_str = serialize(t, fmt)
            for (r, c) in picked:
                records.append({
                    "id": f"niat/cell_locating/{fmt}/{t['table_id']}/{r}_{c}",
                    "task": "cell_locating",
                    "prompt": locating_prompt(table_str, r, c),
                    "gt": cell_value(rows, r, c),
                    "meta": {"table_id": t["table_id"], "structure": t["structure"], "format": fmt,
                             "row": r, "col": c, "n_rows": R, "n_cols": C,
                             "dataset": t["dataset_name"]},
                })
    return records


def build_cell_lookup(tables, questions_per_table: int, seed: int) -> list:
    rng = random.Random(seed)
    records = []
    for t in tables:
        qs = list(t["query_based_NIAT_questions"])
        if questions_per_table > 0 and len(qs) > questions_per_table:
            qs = rng.sample(qs, questions_per_table)
        for fmt in FORMATS:
            table_str = serialize(t, fmt)
            for qi, q in enumerate(qs):
                records.append({
                    "id": f"niat/cell_lookup/{fmt}/{t['table_id']}/{qi}",
                    "task": "cell_lookup",
                    "prompt": lookup_prompt(table_str, q["question"]),
                    "gt": _gt(q["answer"]),
                    "meta": {"table_id": t["table_id"], "structure": t["structure"], "format": fmt,
                             "question": q["question"], "n_rows": t["n_rows"], "n_cols": t["n_cols"],
                             "dataset": t["dataset_name"]},
                })
    return records


def make_fixed_size(rows, n: int, rng: random.Random) -> list:
    """Return an n x n grid: crop the top-left corner; if the source is too
    small, extend by cycling data rows / columns (as in the released samples)."""
    R = len(rows)
    C = max(len(r) for r in rows)
    src = [list(r) + [""] * (C - len(r)) for r in rows]
    col_idx = list(range(min(C, n)))
    while len(col_idx) < n:
        col_idx.append(rng.randrange(C))
    row_idx = list(range(min(R, n)))
    while len(row_idx) < n:
        row_idx.append(1 + rng.randrange(max(1, R - 1)))  # never repeat the header row
    return [[src[i][j] for j in col_idx] for i in row_idx]


def build_cropped(tables, cropped_samples, tables_per_size: int, sizes, seed: int,
                  cells_per_table: int = 0, table_store: Optional[dict] = None) -> list:
    """Fixed-size Markdown tables with one query per cell (or a sample of cells).

    When ``table_store`` is given, the serialised tables are stored there (one
    entry per table id) and the records carry no ``prompt``; the prompt is
    rebuilt on load by :func:`expand_records`. This keeps the 48K-prompt file
    small (each 32x32 table would otherwise be repeated 1024 times)."""
    rng = random.Random(seed)
    # candidate pool: released fixed-size samples first (already N x N crops
    # from the authors' sources), then the flat NIAT tables, biggest first
    flat = [t for t in tables if t["structure"] == "flat"]
    pool = list(cropped_samples) + sorted(flat, key=lambda t: -(min(t["n_rows"], t["n_cols"]) * 1000 + t["n_rows"] * t["n_cols"]))
    records = []
    for n in sizes:
        # prefer sources that need no extension
        ranked = sorted(pool, key=lambda t: -min(len(t["table_rows"]), max(len(r) for r in t["table_rows"])))
        chosen = ranked[:tables_per_size]
        for t in chosen:
            grid = make_fixed_size(t["table_rows"], n, rng)
            table_str = to_markdown_aligned(grid)
            tid = f"{t['table_id']}_{n}x{n}"
            if table_store is not None:
                table_store[tid] = table_str
            cells = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1)]
            if cells_per_table > 0 and cells_per_table < len(cells):
                cells = rng.sample(cells, cells_per_table)
            for (r, c) in cells:
                rec = {
                    "id": f"niat/cropped/{n}/{tid}/{r}_{c}",
                    "task": "cropped",
                    "gt": cell_value(grid, r, c),
                    "meta": {"table_id": tid, "size": n, "row": r, "col": c, "format": "markdown",
                             "source": t.get("source", t.get("dataset_name", "niat"))},
                }
                if table_store is None:
                    rec["prompt"] = locating_prompt(table_str, r, c)
                records.append(rec)
    return records


def expand_records(records: list, tables_path) -> list:
    """Rebuild ``prompt`` for compact cropped records from the table sidecar."""
    tables = json.load(open(tables_path))
    for r in records:
        if "prompt" not in r:
            m = r["meta"]
            r["prompt"] = locating_prompt(tables[m["table_id"]], m["row"], m["col"])
    return records


def load_prompt_records(path) -> list:
    """Read a prompt-set JSONL, expanding compact (cropped) records if needed."""
    from .utils import read_jsonl
    path = Path(path)
    records = read_jsonl(path)
    sidecar = path.with_name(path.stem + "_tables.json")
    if records and "prompt" not in records[0] and sidecar.exists():
        expand_records(records, sidecar)
    return records


def build_niat(data_dir, out_dir, cells_per_table: int = 20, questions_per_table: int = 20,
               tables_per_size: int = 15, sizes=CROP_SIZES, crop_cells_per_table: int = 0,
               seed: int = 0, limit_tables: Optional[int] = None) -> dict:
    tables = load_niat_tables(data_dir)
    if limit_tables:
        rng = random.Random(seed)
        by_s = {s: [t for t in tables if t["structure"] == s] for s in STRUCTURES}
        tables = [t for s in STRUCTURES for t in rng.sample(by_s[s], min(limit_tables, len(by_s[s])))]
    cropped_samples = load_cropped_samples(data_dir)
    table_store = {}
    summary = {}
    for name, recs in [
        ("cell_locating", build_cell_locating(tables, cells_per_table, seed)),
        ("cell_lookup", build_cell_lookup(tables, questions_per_table, seed)),
        ("cropped", build_cropped(tables, cropped_samples, tables_per_size, sizes, seed, crop_cells_per_table,
                                  table_store=table_store)),
    ]:
        if name == "cropped":
            json.dump(table_store, open(f"{out_dir}/niat_cropped_tables.json", "w"))
        path = f"{out_dir}/niat_{name}.jsonl"
        write_jsonl(path, recs)
        summary[name] = len(recs)
        print(f"[niat] {name}: {len(recs)} prompts -> {path}")
    return summary
