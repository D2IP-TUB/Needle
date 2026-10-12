"""Coordinate-carrying serialization (CCS): the method, and its evaluation on the benchmark subset.

A cell-retrieval prompt passes through three steps:

1. the table inside the published prompt is rewritten so that every cell carries its own coordinates,
   ``[r<row>|c<n>=value]``, with a ``columns (name -> code)`` list above the table (``reserialize``);
2. the prompt is given the answer protocol: the query repeated on the far side of the table, a fixed JSON
   answer format whose fields follow the steps of the lookup, and short hints (``add_protocol``);
3. the copied fields of the answer are decoded under a regular expression built from the table, so that the
   model can only report strings that exist in it (``predict``).

Every knob below can be set through the environment as ``CCS_<KNOB>``; ``CCS_PRESET=<name>`` selects one of
the named configurations in ``ccs/presets.py`` (the serialization baselines, the ablation arms, the tag shapes).

    CUDA_VISIBLE_DEVICES=0 python solve.py                      # CCS on the evaluation subset
    CCS_PRESET=markdown_protocol python solve.py                # a serialization baseline
    CCS_MODEL=llama-3.1-8b CCS_PRESET=full python solve.py       # another model, per-task tag shapes

The script prints the metrics and writes the raw outputs to ``CCS_OUT`` (default ``run_outputs.jsonl``).
Importing the module does not load a model: ``transform(prompt, set_name)`` gives the rewritten prompt and
the ``*_regex`` functions the grammars, which the analysis scripts use without a GPU.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
from ccs.presets import preset  # noqa: E402
from ccs.utils import extract_json_answer  # noqa: E402

# --------------------------------------------------------------------------
# Knobs: environment > preset > default
# --------------------------------------------------------------------------
PRESET = os.environ.get("CCS_PRESET", "ccs")
_PRESET = preset(PRESET)


def _raw(name, default):
    v = os.environ.get("CCS_" + name)
    if v is None:
        v = _PRESET.get(name)
    return default if v is None else str(v)


def _flag(name, default):
    v = _raw(name, None)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _set(name, default):
    v = _raw(name, None)
    return default if v is None else {s for s in v.replace(",", " ").split() if s}


MODEL = _raw("MODEL", "qwen3-4b-2507")                       # key in ccs/models.py
GPU_MEMORY_UTILIZATION = float(_raw("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN = int(_raw("MAX_MODEL_LEN", "262144"))        # Qwen3-4B-Instruct-2507's native window holds every prompt
SETS = [s for s in benchmark.SETS if s in _set("SETS", set(benchmark.SETS))]
LIMIT = int(_raw("LIMIT", "0"))                              # >0: first N prompts per set (smoke test, not a valid score)
OUT_PATH = _raw("OUT", "run_outputs.jsonl")

# How the table inside the prompt is written. Every variant keeps every cell.
#   "ccs"                 coordinate-carrying serialization (the tag shape below)
#   "baseline"            the prompt as published (Markdown; NIAT also HTML)
#   "markdown" / "html" / "csv" / "json"   the table in one of the common formats (a table already in that
#                         format is kept as published); "jsonl" / "html_df" are MMTU's own JSON and HTML
#                         serializers (pandas), "html_own" = MMTU's HTML on the MMTU sets, NIAT's on the NIAT sets
#   "spreadsheetllm"      SpreadsheetLLM (Dong et al. 2024): "|A1,Value|B1,Value|..." per row
#   "spreadsheetllm_inv"  SpreadsheetLLM with its inverted index: {Value: Address or Address-range}
SERIALIZATION = _raw("SERIALIZATION", "ccs")

# Tag shape of the CCS serialization.
#   "short"     [r<row>|c<n>=value]                      one tag for every task (the paper's method)
#   "colword"   [row <row>, column c<n> = value]        the same coordinates spelled out
#   "per_task"  one shape per task direction: [value @ row <row>, c<n>] on the needle task,
#               [r<row>|c<n>|column name=value] on table-locate, [r<row>|c<n>=value] on the NIAT sets
#               (the arms of the ablation and transfer runs; governed by the per-task knobs below)
TAG_SHAPE = _raw("TAG_SHAPE", "short")
UNIFORM_TAGS = TAG_SHAPE != "per_task"
TAG_TEMPLATES = {"short": "[r<R>|<C>=<V>]", "colword": "[row <R>, column <C> = <V>]"}
if UNIFORM_TAGS and TAG_SHAPE not in TAG_TEMPLATES:
    raise ValueError(f"CCS_TAG_SHAPE must be one of {sorted(TAG_TEMPLATES)} or per_task, not {TAG_SHAPE!r}")
# with a uniform shape: write the needle task value-first, [value @ row <row>, c<n>] (""/"valuefirst")
NEEDLE_TAG = _raw("NEEDLE_TAG", "")
# with a uniform shape: position -> value sets whose answer names the row and column codes, then copies the
# cell's tag (grammar-limited to the table's real tags), then the value, which is read off the copied tag
TAG_ANSWER_SETS = _set("TAG_ANSWER_SETS", {"niat_cell_locating", "niat_cropped"})

# per-task shapes (TAG_SHAPE = per_task) and the column codes
CELL_TAGS = _flag("CELL_TAGS", True)               # MMTU: tags on every cell
NIAT_CELL_TAGS = _flag("NIAT_CELL_TAGS", True)     # NIAT: tags on every cell
VALUE_FIRST_SETS = _set("VALUE_FIRST_SETS", {"mmtu_niht"})   # value -> position sets written value-first
COLUMNS_ONE_PER_LINE = _flag("COLUMNS_ONE_PER_LINE", True)   # the column list above the table, one column per line
CODE_COLUMNS = _flag("CODE_COLUMNS", True)         # needle task: columns carry a code c<n>; the answer gives the code
LOCATE_CODES = _flag("LOCATE_CODES", True)         # table-locate: codes in the tags, the answer copies the cell's tag

# answer protocol
JSON_ONLY_HINT = _flag("JSON_ONLY_HINT", True)     # needle task: "answer with the JSON object only"
QUERY_LAST_MMTU = _flag("QUERY_LAST_MMTU", True)   # table-locate: repeat row_idx and column name after the table
QUERY_FIRST = _flag("QUERY_FIRST", True)           # NIAT: repeat the question / target coordinates before the table
COPY_EXACT_HINT = _flag("COPY_EXACT_HINT", True)   # cell-lookup: copy the cell exactly as it appears
RETRY_NO_JSON = _set("RETRY_NO_JSON", {"niat_cell_lookup"})  # one direct-answer retry when no JSON object came back

# constrained decoding (XGrammar regular expressions built from the table)
CONSTRAIN_CELL_LOCATING = _flag("CONSTRAIN_CELL_LOCATING", True)  # cell-locating: cell_value is one of the table's cells
CONSTRAIN_LOCATE_TAG = _flag("CONSTRAIN_LOCATE_TAG", True)        # table-locate: the copied tag is one of the table's tags
CONSTRAIN_CROPPED = _flag("CONSTRAIN_CROPPED", True)              # cropped: as cell-locating


def fill_tag(r, c, v):
    return TAG_TEMPLATES[TAG_SHAPE].replace("<R>", r).replace("<C>", c).replace("<V>", v)


# --------------------------------------------------------------------------
# Table parsing
# --------------------------------------------------------------------------
_PIPE = re.compile(r"(?<!\\)\|")
_SEP_CELL = re.compile(r"^:?-{2,}:?$")


def find_markdown_table(prompt):
    """(start_char, end_char, rows) of the largest Markdown table in the prompt, separator row removed, or
    None. ``rows`` is a list of lists of cell strings."""
    lines = prompt.split("\n")
    best, cur = None, []
    pos = 0
    starts = []
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
    start = starts[best[0]]
    end = starts[best[-1]] + len(lines[best[-1]])
    return start, end, rows


from html.parser import HTMLParser  # noqa: E402


class _TableParser(HTMLParser):
    """Parse the first <table> into a rectangular grid, expanding rowspan/colspan by duplicating the cell
    text (as NIAT's flattened table_rows do)."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.cur = None
        self.cell = None
        self.span = None
        self.depth = 0
        self.pending = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.depth += 1
        if self.depth != 1:
            return
        if tag == "tr":
            self.cur = []
            self.col = 0
            for c in sorted(self.pending):  # NIAT convention: spanned-in cells first
                text, rows_left = self.pending[c]
                self.cur.append(text)
                if rows_left > 1:
                    self.pending[c] = (text, rows_left - 1)
                else:
                    del self.pending[c]
            self.col = len(self.cur)
        elif tag in ("td", "th") and self.cur is not None:
            self.cell = []
            self.span = (int(a.get("rowspan", 1) or 1), int(a.get("colspan", 1) or 1))
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        if self.depth < 1 and tag == "table":
            return
        if self.depth != 1:
            return
        if tag in ("td", "th") and self.cell is not None:
            text = re.sub(r"\s+", " ", "".join(self.cell)).strip()
            rs, cs = self.span
            for _ in range(cs):
                self.cur.append(text)
                if rs > 1:
                    self.pending[self.col] = (text, rs - 1)
                self.col += 1
            self.cell = None
        elif tag == "tr" and self.cur is not None:
            self.rows.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def parse_html_table(html):
    p = _TableParser()
    p.feed(html)
    return p.rows


def find_html_table(prompt):
    m = re.search(r"<table\b.*?</table>", prompt, flags=re.S | re.I)
    if not m:
        return None
    return m.start(), m.end(), parse_html_table(m.group(0))


def find_table(prompt):
    """The table of a published prompt: Markdown first, HTML otherwise (NIAT's hierarchical tables)."""
    return find_markdown_table(prompt) or find_html_table(prompt)


# --------------------------------------------------------------------------
# Serialization baselines
# --------------------------------------------------------------------------
NATIVE_FORMATS = ("markdown", "html", "csv", "json", "jsonl", "html_df", "html_own")
NATIVE_NOTES = {
    "markdown": "",   # the published prompts already announce a Markdown table
    "html": "(The table is written in HTML.)\n",
    "csv": "(The table is written as CSV: one line per row, cells separated by commas, the first line being the header row.)\n",
    "json": "(The table is written as JSON: a list with one object per row below the header row, the keys being the header cells.)\n",
    "jsonl": "(The table is written as JSON lines: one object per row below the header row, the keys being the header cells.)\n",
    "html_df": "(The table is written in HTML.)\n",
}


def native_table(rows, fmt, source):
    """The table in one of the common formats, every row padded to the widest one. ``source`` is the format
    the rows were parsed from (Markdown cells carry escaped pipes)."""
    ncol = max(len(r) for r in rows)
    grid = [[(c.replace("\\|", "|") if source == "markdown" else c) for c in r] + [""] * (ncol - len(r)) for r in rows]
    if fmt in ("jsonl", "html_df"):   # MMTU's JsonSerializer / HTMLSerializer (utils/table_serializer.py), cells as strings
        import pandas as _pd
        df = _pd.DataFrame(grid[1:], columns=grid[0], dtype=str)
        if fmt == "html_df":
            return df.to_html(index=False)
        seen, names = {}, []
        for c in df.columns:   # duplicate column headers get a _<n> suffix, as upstream
            seen[c] = seen.get(c, 0) + 1
            names.append(c if seen[c] == 1 else c + f"_{seen[c]}")
        df.columns = names
        return df.to_json(orient="records", lines=True).rstrip("\n")
    if fmt == "markdown":   # NIAT's compact style
        lines = ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in grid]
        lines.insert(1, "| " + " | ".join(["---"] * ncol) + " |")
        return "\n".join(lines)
    if fmt == "html":       # NIAT's one-row-per-line style
        import html as _html
        return "\n".join(['<table border="1" cellspacing="0">']
                         + ["<tr> " + " ".join(f"<td> {_html.escape(c, quote=False)} </td>" for c in r) + " </tr>" for r in grid]
                         + ["</table>"])
    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(grid)
        return buf.getvalue().rstrip("\n")
    d = lambda v: json.dumps(v, ensure_ascii=False)  # noqa: E731  json: duplicate column names stay duplicate keys
    return "[\n" + ",\n".join("{" + ", ".join(f"{d(k)}: {d(v)}" for k, v in zip(grid[0], r)) + "}" for r in grid[1:]) + "\n]"


def spreadsheetllm_table(rows, inverted):
    """SpreadsheetLLM encodings (Dong et al. 2024). Vanilla: "|A1,Value|B1,Value|..." per row, row 1 = header.
    Inverted index: {Value: Address or Address-range, ...}, equal values merged, empty cells dropped."""
    ncol = max(len(r) for r in rows)

    def col_letter(j):
        sname = ""
        j += 1
        while j:
            j, r = divmod(j - 1, 26)
            sname = chr(65 + r) + sname
        return sname

    grid = [list(r) + [""] * (ncol - len(r)) for r in rows]
    if not inverted:
        return "\n".join("".join(f"|{col_letter(j)}{i + 1},{grid[i][j]}" for j in range(ncol)) + "|" for i in range(len(grid)))
    index = {}
    for i in range(len(grid)):
        for j in range(ncol):
            v = grid[i][j]
            if v != "":
                index.setdefault(v, []).append((i, j))

    def ranges(cells):  # merge vertical runs in the same column into A2:A5
        out, cells = [], sorted(cells, key=lambda c: (c[1], c[0]))
        k = 0
        while k < len(cells):
            i0, j0 = cells[k]
            i1 = i0
            while k + 1 < len(cells) and cells[k + 1][1] == j0 and cells[k + 1][0] == i1 + 1:
                k += 1
                i1 = cells[k][0]
            out.append(f"{col_letter(j0)}{i0 + 1}" if i1 == i0 else f"{col_letter(j0)}{i0 + 1}:{col_letter(j0)}{i1 + 1}")
            k += 1
        return ",".join(out)

    return "{\n" + ",\n".join(f"  {json.dumps(v, ensure_ascii=False)}: {json.dumps(ranges(c), ensure_ascii=False)}" for v, c in index.items()) + "\n}"


SPREADSHEETLLM_NOTES = {
    "spreadsheetllm": "(The table is encoded spreadsheet-style, one row per line, each cell as |Address,Value|; row 1 is the header row and column A is the first column.)\n",
    "spreadsheetllm_inv": "(The table is encoded spreadsheet-style as an inverted index {Value: Address or Address-range}; A1 is the top-left cell, row 1 is the header row, column A is the first column; empty cells are omitted.)\n",
}


# --------------------------------------------------------------------------
# Coordinate-carrying serialization
# --------------------------------------------------------------------------
def uniform_tags(rows, set_name):
    """Tags of one shape for every task: ([(code, column name)], [(line label, [(tag, value)])]).
    MMTU rows are keyed by the table's own row_idx column and codes start at c1 after it; NIAT rows by
    1-based position, header = row 1."""
    ncol = max(len(r) for r in rows)
    grid = [r + [""] * (ncol - len(r)) for r in rows]
    header = grid[0]
    if set_name.startswith("mmtu"):
        cols = range(1, ncol) if header[0] == "row_idx" else range(ncol)
        keyed = [(r[0] if header[0] == "row_idx" else str(i), r) for i, r in enumerate(grid[1:])]
        keyed = [(f"row_idx={rid}" if header[0] == "row_idx" else f"row {rid}", rid, r) for rid, r in keyed]
        code = lambda j: f"c{j}"  # noqa: E731
    else:
        cols = range(ncol)
        keyed = [("row 1 (header)" if i == 0 else f"row {i + 1}", str(i + 1), r) for i, r in enumerate(grid)]
        code = lambda j: f"c{j + 1}"  # noqa: E731
    if set_name == "mmtu_niht" and NEEDLE_TAG == "valuefirst":
        tag = lambda rid, j, v: f"[{v} @ row {rid}, {code(j)}]"  # noqa: E731
    else:
        tag = lambda rid, j, v: fill_tag(rid, code(j), v)  # noqa: E731
    return ([(code(j), header[j]) for j in cols],
            [(label, [(tag(rid, j, r[j]), r[j]) for j in cols]) for label, rid, r in keyed])


def ccs_table(rows, set_name):
    """The table as tagged cells, one line per row. MMTU tasks address columns by *name* and rows by the
    table's own ``row_idx`` column; NIAT tasks address both by 1-based *position* (header = row 1)."""
    header = rows[0]
    ncol = max(len(r) for r in rows)
    header = header + [""] * (ncol - len(header))
    body = [r + [""] * (ncol - len(r)) for r in rows[1:]]
    named = set_name.startswith("mmtu")
    if named:
        colkeys = header
        rowkey = lambda i, r: f"row_idx={r[0]}" if header[0] == "row_idx" else f"row {i + 1}"  # noqa: E731
        cell_cols = range(1, ncol) if header[0] == "row_idx" else range(ncol)
    else:
        colkeys = [f"c{j + 1}" for j in range(ncol)]
        cell_cols = range(ncol)
    out = []
    if UNIFORM_TAGS:
        names, lines = uniform_tags(rows, set_name)
        out.append("columns (name -> code):")
        out.extend(f"  {nm} -> {code}" for code, nm in names)
        out.append("rows:")
        out.extend(label + ": " + " ".join(t for t, _ in tags) for label, tags in lines)
        return "\n".join(out)
    # per-task shapes
    if named:
        if (set_name in VALUE_FIRST_SETS and CODE_COLUMNS) or (LOCATE_CODES and set_name == "mmtu_table_locate"):
            if COLUMNS_ONE_PER_LINE and set_name == "mmtu_table_locate":
                out.append("columns (name -> code):")
                out.extend(f"  {header[j]} -> c{j}" for j in cell_cols)
                out.append("rows:")
            elif COLUMNS_ONE_PER_LINE:
                out.append("columns:")
                out.extend(f"  c{j} = {header[j]}" for j in cell_cols)
                out.append("rows:")
            else:
                out.append("columns: " + " | ".join(f"c{j}={header[j]}" for j in cell_cols))
        else:
            out.append("columns: " + " | ".join(header))
        for i, r in enumerate(body):
            rid = r[0] if header[0] == "row_idx" else str(i + 1)
            if CELL_TAGS and set_name in VALUE_FIRST_SETS and CODE_COLUMNS:
                out.append(rowkey(i, r) + ": " + " ".join(f"[{r[j]} @ row {rid}, c{j}]" for j in cell_cols))
            elif CELL_TAGS and set_name in VALUE_FIRST_SETS:
                out.append(rowkey(i, r) + ": " + " ".join(f"[{r[j]} @ r{rid}, {colkeys[j]}]" for j in cell_cols))
            elif CELL_TAGS and LOCATE_CODES and set_name == "mmtu_table_locate":
                out.append(rowkey(i, r) + ": " + " ".join(f"[r{rid}|c{j}|{colkeys[j]}={r[j]}]" for j in cell_cols))
            elif CELL_TAGS:
                out.append(rowkey(i, r) + ": " + " ".join(f"[r{rid}|{colkeys[j]}={r[j]}]" for j in cell_cols))
            else:
                out.append(rowkey(i, r) + ": " + " | ".join(f"{colkeys[j]}={r[j]}" for j in cell_cols))
    else:
        if NIAT_CELL_TAGS:
            out.append("row 1 (header): " + " ".join(f"[r1|{colkeys[j]}={header[j]}]" for j in cell_cols))
            for i, r in enumerate(body):
                out.append(f"row {i + 2}: " + " ".join(f"[r{i + 2}|{colkeys[j]}={r[j]}]" for j in cell_cols))
        else:
            out.append("row 1 (header): " + " | ".join(f"{colkeys[j]}={header[j]}" for j in cell_cols))
            for i, r in enumerate(body):
                out.append(f"row {i + 2}: " + " | ".join(f"{colkeys[j]}={r[j]}" for j in cell_cols))
    return "\n".join(out)


def ccs_note(set_name):
    """The one-sentence note that precedes the tagged table and states its convention."""
    mmtu = set_name.startswith("mmtu")
    if UNIFORM_TAGS:
        if set_name == "mmtu_niht" and NEEDLE_TAG == "valuefirst":
            body = ("the 'columns:' line maps each column code c<n> to the column name, and every cell is "
                    "written as [value @ row <row_idx>, c<n>]")
        elif mmtu:
            body = ("the 'columns:' line maps each column code c<n> to the column name, and every cell is "
                    "written as " + fill_tag("<row_idx>", "c<n>", "value"))
        else:
            body = ("the 'columns:' line maps each column code c<column_ID> to the column name, and every cell is "
                    "written as " + fill_tag("<row_ID>", "c<column_ID>", "value") + ", the header being row 1")
    elif set_name in VALUE_FIRST_SETS and CELL_TAGS and CODE_COLUMNS:
        body = ("every cell is written as [value @ row <row_idx>, c<n>], i.e. each value is directly followed "
                "by its row index and its column code; the 'columns:' line maps each code c<n> to the column name")
    elif set_name in VALUE_FIRST_SETS and CELL_TAGS:
        body = ("every cell is written as [value @ r<row_idx>, column name], i.e. each value is directly "
                "followed by its row index and its column name inside the same brackets")
    elif LOCATE_CODES and set_name == "mmtu_table_locate" and CELL_TAGS:
        body = ("the 'columns:' line maps each column code c<n> to the column name, and every cell is "
                "written as [r<row_idx>|c<n>|column name=value]")
    elif mmtu and CELL_TAGS:
        body = ("every cell is written as [r<row_idx>|column name=value], i.e. the row index and the "
                "column name of a value are inside the same brackets as the value")
    elif mmtu:
        body = "columns are addressed by their name"
    elif NIAT_CELL_TAGS:
        body = "every cell is written as [r<row_ID>|c<column_ID>=value], the header being row 1"
    else:
        body = "c<n> is the 1-based column_ID and 'row <n>' the 1-based row_ID, the header being row 1"
    return "(The table is written one row per line as key=value pairs; " + body + ".)\n"


def reserialize(prompt, set_name):
    """Replace the table block of a published prompt by its serialization under SERIALIZATION."""
    if SERIALIZATION == "baseline":
        return prompt
    md = find_markdown_table(prompt)
    found = md or find_html_table(prompt)
    if found is None or len(found[2]) < 2:
        return prompt  # no table found: unchanged
    start, end, rows = found
    if SERIALIZATION in NATIVE_FORMATS:
        source = "markdown" if md else "html"
        fmt = SERIALIZATION
        if fmt == "html_own":   # each benchmark in its own HTML: MMTU's serializer, NIAT's published tables
            fmt = "html_df" if set_name.startswith("mmtu") else "html"
        table = prompt[start:end] if source == fmt.split("_")[0] else native_table(rows, fmt, source)
        return prompt[:start] + NATIVE_NOTES[fmt] + table + prompt[end:]
    if SERIALIZATION in SPREADSHEETLLM_NOTES:
        table = spreadsheetllm_table(rows, inverted=SERIALIZATION == "spreadsheetllm_inv")
        return prompt[:start] + SPREADSHEETLLM_NOTES[SERIALIZATION] + table + prompt[end:]
    if SERIALIZATION != "ccs":
        raise ValueError(SERIALIZATION)
    return prompt[:start] + ccs_note(set_name) + ccs_table(rows, set_name) + prompt[end:]


# --------------------------------------------------------------------------
# Answer protocol
# --------------------------------------------------------------------------
ANSWER_KEYS = {"mmtu_niht": ["row", "column"], "mmtu_table_locate": ["cell"],
               "niat_cell_locating": ["cell_value"], "niat_cropped": ["cell_value"],
               "niat_cell_lookup": ["answer"]}

LOOKUP_HINT = ("Important: the final answer must be the content of a single table cell, copied exactly as it "
               "appears in the table - keep or omit currency symbols, thousands separators, parentheses and units "
               "exactly as the cell has them (do not add symbols that are not in the cell, do not reformat numbers, "
               "do not add words). Note that some tables have more than one header row: a column's full label may be "
               "split across the first two or three rows, and a row's label may be split across the first columns.\n")

# MMTU's output-format paragraph of the table-locate prompt, and its replacement: row code, column code, the cell's
# complete tag copied from the table, then the value
LOCATE_OLD_FORMAT = ('Return the exact cell value in JSON format: {"cell": <cell>}. For string values, return the string as it is. For numerical values, return the number as it is, do not round or format the number.\nDo not explain, return the JSON object directly. ')
LOCATE_NEW_FORMAT = ('First look up the code c<n> of the requested column name in the "columns:" line of the table, then find the cell '
                     'tagged [r<row_idx>|c<n>|...] in the requested row. Return the JSON object '
                     '{"row_code": "r<row_idx>", "column_code": "c<n>", "tag": "[r<row_idx>|c<n>|<column name>=<cell>]", "cell": <cell>} '
                     '(the row code and the column code first, then the complete cell tag copied character by character from the table, '
                     'then the cell value copied from that tag). For string values, return the string as it is. '
                     'For numerical values, return the number as it is, do not round or format the number.\\nDo not explain, return the JSON object directly. ')

# NIAT's answer object, and its replacement with the copied tag
NIAT_OLD_FORMAT = '{"row_ID": "m", "column_ID": "n", "cell_value": "<Corresponding Cell Value>"}'
NIAT_TAG_FORMAT = ('{"row_ID": "m", "column_ID": "n", "tag": "<<TAG>>", "cell_value": "<Corresponding Cell Value>"}\n'
                   '(the row_ID and the column_ID first, then the complete tag of the cell [r<m>|c<n>...] copied character by '
                   'character from the table, then the cell value copied from that tag)')

_Q_RE = re.compile(r"## Question:\n(.*?)\n\n## Your Response:", re.S)
_LOC_RE = re.compile(r"row_ID is (\d+) and the column_ID is (\d+)")
_LOCQ_RE = re.compile(r'row_idx: (\d+)\n\ncolumn name: "(.*?)"\n\nTable:', re.S)


def query_first(prompt, set_name):
    """Put the question (lookup) or the target coordinates (cell-locating) before the table too."""
    if set_name == "niat_cell_lookup":
        m = _Q_RE.search(prompt)
        if m and "\nTable:\n" in prompt:
            return prompt.replace("\nTable:\n", f"\nThe question to answer (repeated after the table): {m.group(1).strip()}\n\nTable:\n", 1)
    elif set_name in ("niat_cell_locating", "niat_cropped"):
        m = _LOC_RE.search(prompt)
        if m and "Given table:\n" in prompt:
            return prompt.replace("Given table:\n", f"Target cell (repeated after the table): row_ID {m.group(1)}, column_ID {m.group(2)}.\n\nGiven table:\n", 1)
    return prompt


def query_last_mmtu(prompt, set_name):
    """Table-locate: repeat the requested row_idx and column name after the table."""
    if set_name == "mmtu_table_locate":
        m = _LOCQ_RE.search(prompt)
        if m and "\nOutput:\n" in prompt:
            return prompt.replace("\nOutput:\n", f'\nReminder: row_idx {m.group(1)}, column name "{m.group(2)}".\n\nOutput:\n', 1)
    return prompt


def add_protocol(prompt, set_name):
    """The answer protocol: repeated query, answer format that follows the lookup, hints."""
    if QUERY_LAST_MMTU:
        prompt = query_last_mmtu(prompt, set_name)
    if QUERY_FIRST:
        prompt = query_first(prompt, set_name)
    if LOCATE_CODES and set_name == "mmtu_table_locate":
        new = LOCATE_NEW_FORMAT
        if UNIFORM_TAGS:
            new = (new.replace("[r<row_idx>|c<n>|...]", fill_tag("<row_idx>", "c<n>", "..."))
                      .replace("[r<row_idx>|c<n>|<column name>=<cell>]", fill_tag("<row_idx>", "c<n>", "<cell>")))
        prompt = prompt.replace(LOCATE_OLD_FORMAT, new, 1)
    if UNIFORM_TAGS and set_name in TAG_ANSWER_SETS and set_name.startswith("niat") and NIAT_OLD_FORMAT in prompt:
        fmt = NIAT_TAG_FORMAT.replace("<<TAG>>", fill_tag("<m>", "c<n>", "<cell>"))
        if TAG_SHAPE != "short":
            fmt = fmt.replace("[r<m>|c<n>...]", fill_tag("<m>", "c<n>", "..."))
        prompt = prompt.replace(NIAT_OLD_FORMAT, fmt, 1)
    if COPY_EXACT_HINT and set_name == "niat_cell_lookup" and "## Question:" in prompt:
        prompt = prompt.replace("## Question:", LOOKUP_HINT + "## Question:", 1)
    if JSON_ONLY_HINT and set_name == "mmtu_niht":
        extra = (' Give the column as its code from the brackets next to the value, e.g. {"row": 12, "column": "c7"}.'
                 if CODE_COLUMNS else "")
        prompt = prompt.rstrip("\n") + "\nAnswer with the JSON object only, without any explanation or search log." + extra + "\n"
    return prompt


import contextlib  # noqa: E402


@contextlib.contextmanager
def configured(**knobs):
    """Temporarily override knobs of this module (e.g. ``with solve.configured(SERIALIZATION="json"): ...``),
    for analyses that need several serializations of the same prompt in one process."""
    g = globals()
    old = {k: g[k] for k in knobs}
    g.update(knobs)
    try:
        yield
    finally:
        g.update(old)


def transform(prompt, set_name):
    """The prompt the model sees: serialization and answer protocol."""
    return add_protocol(reserialize(prompt, set_name), set_name)


# --------------------------------------------------------------------------
# Constrained decoding and answer post-processing
# --------------------------------------------------------------------------
_RX_META = set('\\.^$|?*+()[]{}')


def _rx_escape(v):
    return "".join("\\" + ch if ch in _RX_META else ch for ch in v)


def _json_inner(v):
    return json.dumps(v, ensure_ascii=False)[1:-1]


_CODE_RE = re.compile(r'"column"\s*:\s*"?c(\d+)"?')


def decode_column_codes(text, header):
    """Replace {"row": r, "column": "c7"} by the column's name (from the parsed header)."""
    m = _CODE_RE.search(text or "")
    if not m:
        return text
    j = int(m.group(1))
    if j >= len(header):
        return text
    name = header[j].replace("\\", "\\\\").replace('"', '\\"')
    return text[:m.start()] + f'"column": "{name}"' + text[m.end():]


def locate_tag_regex(prompt):
    """Per-task shapes, table-locate: {"row_code": ..., "column_code": ..., "tag": <one of the real tags>, "cell": ...}."""
    found = find_markdown_table(prompt)
    if not found:
        return None
    rows = found[2]
    header = rows[0]
    tags = []
    for i, r in enumerate(rows[1:]):
        rid = r[0] if header[0] == "row_idx" else str(i + 1)
        tags.extend(f"[r{rid}|c{j}|{header[j]}={r[j]}]" for j in range(1, len(header)) if j < len(r))
    alts = "|".join(_rx_escape(_json_inner(t)) for t in tags)
    return (r'\{"row_code": "r[0-9]+", "column_code": "c[0-9]+", "tag": "(' + alts + r')", "cell": "?[^"}]*"?\}')


_TAG_VALUE_RE = re.compile(r'"tag":\s*"\[r[0-9]+\|c[0-9]+\|.*=(.*?)\]"\s*,\s*"cell"', re.S)


def cell_from_tag(text):
    """Per-task shapes, table-locate: make the cell field agree with the tag the model chose."""
    m = _TAG_VALUE_RE.search(text or "")
    if not m:
        return text
    try:
        val = json.loads('"' + m.group(1) + '"')
    except Exception:
        return text
    return text[:m.end()] + ": " + json.dumps(val, ensure_ascii=False) + "}"


TAG_ANSWER_KEYS = {"mmtu_table_locate": ("row_code", "r", "column_code", "c", "cell"),
                   "niat_cell_locating": ("row_ID", "", "column_ID", "", "cell_value"),
                   "niat_cropped": ("row_ID", "", "column_ID", "", "cell_value")}


def tag_values(prompt, set_name):
    """{tag: value} of every cell as the uniform serialization writes it (None if no table)."""
    found = find_table(prompt)
    if not found or len(found[2]) < 2:
        return None
    return {t: v for _, tags in uniform_tags(found[2], set_name)[1] for t, v in tags}


def tag_answer_regex(tags, set_name, max_cells=4000):
    """Uniform shape: the answer names the row and column codes, copies one of the table's real tags, then the value."""
    if not tags or len(tags) > max_cells:
        return None
    rk, rp, ck, cp, vk = TAG_ANSWER_KEYS[set_name]
    alts = "|".join(_rx_escape(_json_inner(t)) for t in tags)
    return (r'\{"' + rk + '": "' + rp + r'[0-9]+", "' + ck + '": "' + cp + r'[0-9]+", "tag": "(' + alts + r')", "'
            + vk + r'": "?[^"}]*"?\}')


_TAG_FIELD_RE = re.compile(r'"tag":\s*"((?:[^"\\]|\\.)*)"')


def value_from_tag(text, tags, set_name):
    """Overwrite the value field with the value of the tag the model copied (looked up, not re-parsed)."""
    m = _TAG_FIELD_RE.search(text or "")
    if not m or not tags:
        return text
    try:
        v = tags.get(json.loads('"' + m.group(1) + '"'))
    except Exception:
        return text
    if v is None:
        return text
    return text[:m.end()] + f', "{TAG_ANSWER_KEYS[set_name][4]}": ' + json.dumps(v, ensure_ascii=False) + "}"


def cell_value_regex(prompt, max_cells=4000):
    """{"row_ID": "m", "column_ID": "n", "cell_value": <one of the table's cell strings>}."""
    found = find_table(prompt)
    if not found:
        return None
    vals = sorted({c.strip() for row in found[2] for c in row} - {""})
    if len(vals) > max_cells:
        return None
    alts = "|".join(_rx_escape(_json_inner(v)) for v in vals)
    # the group is optional so that an empty cell ("") is also allowed
    return r'\{"row_ID": "?[0-9]+"?, "column_ID": "?[0-9]+"?, "cell_value": "(' + alts + r')?"\}'


def grammars(prompts, set_name):
    """Per prompt: the regular expression that constrains the answer (None = unconstrained), and the
    {tag: value} map of the uniform shape where the value is read off the copied tag."""
    tag_maps = None
    if UNIFORM_TAGS and (set_name in TAG_ANSWER_SETS or (LOCATE_CODES and set_name == "mmtu_table_locate")):
        tag_maps = [tag_values(p, set_name) for p in prompts]
        return [tag_answer_regex(tm, set_name) for tm in tag_maps], tag_maps
    if (CONSTRAIN_CELL_LOCATING and set_name == "niat_cell_locating") or (CONSTRAIN_CROPPED and set_name == "niat_cropped"):
        return [cell_value_regex(p) for p in prompts], None
    if CONSTRAIN_LOCATE_TAG and set_name == "mmtu_table_locate":
        return [locate_tag_regex(p) for p in prompts], None
    return None, None


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
engine = None


def load_engine():
    global engine
    if engine is None:
        from ccs.engine import Engine
        engine = Engine(MODEL, long_context=True, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_model_len=MAX_MODEL_LEN)
    return engine


def predict(prompts, max_tokens, set_name):
    """One completion per published prompt ("" for prompts that do not fit the context window)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    load_engine()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    regexes, tag_maps = grammars(prompts, set_name)
    sps = None
    if regexes is not None:
        sps = [SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0, structured_outputs=StructuredOutputsParams(regex=rx))
               if rx else sp for rx in regexes]
    headers = None
    if CODE_COLUMNS and set_name in VALUE_FIRST_SETS:
        headers = [((find_table(p) or (0, 0, [[]]))[2][0]) for p in prompts]
    prompts = [transform(p, set_name) for p in prompts]
    ids = [engine.encode(p) for p in prompts]
    keep = [i for i, x in enumerate(ids) if len(x) <= engine.max_model_len - max_tokens]
    texts = [""] * len(prompts)
    if keep:
        params = [sps[i] for i in keep] if sps else sp
        outs = engine.llm.generate([{"prompt_token_ids": ids[i]} for i in keep], params, use_tqdm=True)
        for i, o in zip(keep, outs):
            texts[i] = o.outputs[0].text
            if tag_maps is not None:
                texts[i] = value_from_tag(texts[i], tag_maps[i], set_name)
            elif sps is not None and set_name == "mmtu_table_locate":
                texts[i] = cell_from_tag(texts[i])
            if headers is not None:
                texts[i] = decode_column_codes(texts[i], headers[i])
        if set_name in RETRY_NO_JSON:
            key = ANSWER_KEYS[set_name][0]
            redo = [i for i in keep if extract_json_answer(texts[i], key) == "JSONParsingError"]
            if redo:
                retry_prompts = [prompts[i].rstrip("\n") + "\nDo not reason at length: give at most three short sentences "
                                 "and then the JSON answer.\n" for i in redo]
                r_ids = [engine.encode(p) for p in retry_prompts]
                r_outs = engine.llm.generate([{"prompt_token_ids": x} for x in r_ids], sp, use_tqdm=True)
                for i, o in zip(redo, r_outs):
                    cand = o.outputs[0].text
                    if extract_json_answer(cand, key) != "JSONParsingError":
                        texts[i] = cand
    return texts


def describe():
    """One line with the configuration (for logs and result files)."""
    return (f"preset={PRESET} serialization={SERIALIZATION} tag_shape={TAG_SHAPE} needle_tag={NEEDLE_TAG or '-'} "
            f"tag_answer_sets={sorted(TAG_ANSWER_SETS)} tags={int(CELL_TAGS)}/{int(NIAT_CELL_TAGS)} "
            f"codes={int(CODE_COLUMNS)}/{int(LOCATE_CODES)} value_first={sorted(VALUE_FIRST_SETS)} "
            f"hints={int(JSON_ONLY_HINT)}/{int(COPY_EXACT_HINT)} query={int(QUERY_FIRST)}/{int(QUERY_LAST_MMTU)} "
            f"grammar={int(CONSTRAIN_CELL_LOCATING)}/{int(CONSTRAIN_LOCATE_TAG)}/{int(CONSTRAIN_CROPPED)} "
            f"retry={sorted(RETRY_NO_JSON)}")


def main():
    t_start = time.time()
    load_engine()
    if LIMIT:   # smoke test: first LIMIT prompts of each set (the printed score is not comparable)
        data = {s: recs[:LIMIT] for s, recs in benchmark.load(SETS).items()}
        outputs = {}
        for s, recs in data.items():
            texts = predict([r["prompt"] for r in recs], benchmark.MAX_TOKENS[s], s)
            outputs.update({r["id"]: (t or "") for r, t in zip(recs, texts)})
        metrics = benchmark.score(data, outputs)
    else:
        metrics = benchmark.evaluate(predict, sets=SETS, save_outputs=OUT_PATH)
    metrics["total_seconds"] = time.time() - t_start
    benchmark.print_summary(metrics)
    print(f"{'total_seconds:':24s}{metrics['total_seconds']:.1f}")
    print(f"{'model:':24s}{MODEL}")
    print(f"{'max_model_len:':24s}{engine.max_model_len}")
    print(f"{'config:':24s}{describe()}  limit={LIMIT}")
    print("metrics_json:" + json.dumps({k: metrics.get(k) for k in benchmark.METRIC_NAMES + ["n_prompts", "answered", "total_seconds"]}))


if __name__ == "__main__":
    main()
