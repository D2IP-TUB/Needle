"""Shared pieces of the mechanistic analysis: synthetic NIAT-style cell-locating prompts in Markdown and in
the CCS serialization, character spans of the table tokens, the model, and the output directory.

Synthetic tables hold one distinct word or number per cell. The prompt is NIAT's cell-locating prompt with
the table swapped; the CCS form is produced by ``solve.transform`` under the per-task tag shapes
(``[r<row_ID>|c<column_ID>=value]`` on NIAT prompts), which is the serialization the paper analyses.
"""

import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CCS_PRESET", "full")   # per-task tag shapes (see solve.py); set before solve is imported
import benchmark  # noqa: E402
import solve  # noqa: E402

MODEL = os.environ.get("CCS_MECH_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
OUT = Path(os.environ.get("CCS_MECH_OUT", ROOT / "results" / "mechanistic"))

WORDS = """apple river stone cloud forest garden window bridge silver copper marble velvet candle basket meadow harbor
castle lantern pepper ginger walnut maple cedar willow falcon sparrow heron otter badger beaver salmon trout
pearl amber coral ivory ruby topaz jade onyx quartz cobalt indigo crimson scarlet violet olive saffron
london paris berlin madrid vienna prague lisbon dublin oslo tokyo delhi cairo lima quito denver austin boston
anchor compass rudder cabin harvest orchard pasture valley canyon glacier desert island lagoon summit ridge
piano violin cello flute trumpet drum guitar banjo harp organ tango waltz polka samba blues jazz opera
baker tailor miller carter mason cooper fisher hunter porter weaver potter smith brewer farmer sailor pilot
monday tuesday friday sunday april june august october winter spring summer autumn dawn dusk noon midnight
oxygen carbon helium sodium iron zinc nickel argon neon radon cobalt lithium boron silicon sulfur
hammer chisel wrench pliers ladder bucket shovel needle thread button ribbon fabric leather cotton linen
tiger zebra camel panda koala lemur bison moose cobra gecko iguana parrot pelican puffin raven robin wren
alpha bravo delta echo foxtrot hotel india juliet kilo lima mike oscar papa romeo sierra tango victor zulu
""".split()
WORDS = sorted(set(WORDS))

# a table that reads like a real one, for the single-head figure
ORDERS = [["order", "customer", "city", "product", "price"], ["1041", "Meyer", "Berlin", "laptop", "899"],
          ["1042", "Tanaka", "Osaka", "monitor", "249"], ["1043", "Haddad", "Beirut", "printer", "179"],
          ["1044", "Silva", "Lisbon", "tablet", "329"]]

_TEMPLATE = None


def template():
    """One NIAT cell-locating prompt of the cropped set, the template for the synthetic tables."""
    global _TEMPLATE
    if _TEMPLATE is None:
        d = benchmark.load(["niat_cropped"])["niat_cropped"]
        _TEMPLATE = [r for r in d if r["meta"]["size"] == 8][0]["prompt"]
    return _TEMPLATE


def synth_rows(rng, R, C):
    """R rows (row 1 = header) x C columns; every cell string unique within the table."""
    used = set()
    rows = []
    for i in range(R):
        row = []
        for j in range(C):
            while True:
                v = rng.choice(WORDS) if (i == 0 or rng.random() < 0.5) else str(rng.randint(0, 9999))
                if i == 0 and len(used) >= len(WORDS) - 2:   # header wider than the word list
                    v = rng.choice(WORDS) + str(rng.randint(2, 99))
                if v not in used:
                    used.add(v)
                    row.append(v)
                    break
        rows.append(row)
    return rows


def md_table(rows):
    C = len(rows[0])
    w = [max(len(r[j]) for r in rows) for j in range(C)]
    lines = ["| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(r)) + " |" for r in rows]
    sep = "| " + " | ".join("-" * w[j] for j in range(C)) + " |"
    return "\n".join([lines[0], sep] + lines[1:])


_LOC = re.compile(r"row_ID is (\d+) and the column_ID is (\d+)")


def md_prompt(rows, tr, tc):
    """NIAT's cell-locating prompt with the synthetic table and the target (tr, tc)."""
    t = template()
    s = t.index("Given table:\n") + len("Given table:\n")
    e = t.index("\n\n", s)
    p = t[:s] + md_table(rows) + t[e:]
    return _LOC.sub(f"row_ID is {tr} and the column_ID is {tc}", p)


def tagged_prompt(md):
    """The CCS form of a Markdown prompt, exactly as solve.py writes it (per-task tag shapes)."""
    return solve.transform(md, "niat_cropped")


def chat(tok, prompt):
    kw = {"enable_thinking": False} if "Qwen3-8B" in MODEL else {}
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, **kw)


def token_ids(tok, text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]


def toks_in(offs, span):
    a, b = span
    return [i for i, (x, y) in enumerate(offs) if y > a and x < b and y > x]


def load_hf_model(eager=True):
    """Tokenizer and model (bfloat16, on the visible GPU); eager attention exposes the attention weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    kw = {"attn_implementation": "eager"} if eager else {}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, **kw).cuda().eval()
    return tok, model


# --------------------------------------------------------------------------
# Character spans of the table tokens
# --------------------------------------------------------------------------
_UNIT = re.compile(r"\[r(\d+)\|c(\d+)=(.*?)\](?= \[|$)")


def units(text, fmt):
    """Cells of the last table in ``text``: a list of (row, column, ordinal column, spans) with spans
    {"value": (a, b), "delim": (a, b)} and, for the tagged form, "tag" = the ``r<i>|c<j>`` part.
    Markdown: delim = the '|' before the cell. Tagged: delim = the closing ']'."""
    res = []
    if fmt == "markdown":
        b = [m for m in re.finditer(r"(?:^\|.*\|\n?)+", text, re.M)][-1]
        row = 0
        for m in re.finditer(r"[^\n]+", b.group(0)):
            line = m.group(0)
            lb = b.start() + m.start()
            if re.match(r"^\|?\s*:?-{2,}", line):
                continue
            row += 1
            col = 0
            for cm in re.finditer(r"\|([^|]*)", line):
                inner = cm.group(1)
                if cm.end() == len(line) and inner.strip() == "":
                    break
                col += 1
                st = cm.start(1) + (len(inner) - len(inner.lstrip()))
                en = cm.start(1) + len(inner.rstrip())
                res.append((row, col, col, {"value": (lb + st, lb + en), "delim": (lb + cm.start(), lb + cm.start() + 1)}))
    else:
        b = [m for m in re.finditer(r"(?:^row \d+.*\n?)+", text, re.M)][-1]
        for m in re.finditer(r"[^\n]+", b.group(0)):
            line = m.group(0)
            lb = b.start() + m.start()
            row = int(re.match(r"row (\d+)", line).group(1))
            for pos, cm in enumerate(_UNIT.finditer(line), 1):
                res.append((row, int(cm.group(2)), pos, {"value": (lb + cm.start(3), lb + cm.end(3)),
                                                         "tag": (lb + cm.start() + 1, lb + cm.start(3) - 1),
                                                         "delim": (lb + cm.end() - 1, lb + cm.end())}))
    return res


def opener(offs, sp, fmt):
    """Token index of the token that opens a cell: the pipe before it (Markdown) or the '[' (tagged)."""
    return toks_in(offs, sp["delim"] if fmt == "markdown" else (sp["tag"][0] - 1, sp["tag"][0]))[-1]


def json_table(rows):
    """The JSON rows of the serialization baseline (one object per data row, header cells as keys)."""
    import json as _json
    d = lambda v: _json.dumps(v, ensure_ascii=False)  # noqa: E731
    return "[\n" + ",\n".join("{" + ", ".join(f"{d(k)}: {d(v)}" for k, v in zip(rows[0], r)) + "}" for r in rows[1:]) + "\n]"


JSON_NOTE = "(The table is written as JSON: a list with one object per row below the header row, the keys being the header cells.)\n"
JSON_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)":\s*"((?:[^"\\]|\\.)*)"')


def json_prompt(rows, tr, tc):
    """The synthetic prompt with the table as JSON rows and the target repeated before the table."""
    md = md_prompt(rows, tr, tc)
    s = md.index("Given table:\n")
    e = md.index("\n\n", s + len("Given table:\n"))
    return md[:s] + f"Target cell (repeated after the table): row_ID {tr}, column_ID {tc}.\n\nGiven table:\n" + JSON_NOTE + json_table(rows) + md[e:]


def json_units(text):
    """Cells of a JSON table: (row, col) -> token spans {"open", "key", "value", "pair"} for the data rows 2..R
    (the header row has no cell of its own; its cells are the keys), plus {row: first token of the line}."""
    s = text.index("[\n", text.index("Given table:\n")) + 2
    e = text.index("\n]", s)
    U, line_first = {}, {}
    for li, m in enumerate(re.finditer(r"[^\n]+", text[s:e])):
        row = li + 2
        a = s + m.start()
        line_first[row] = (a, a + 1)
        for col, pm in enumerate(JSON_PAIR.finditer(m.group(0)), 1):
            v0, v1 = pm.start(2), pm.end(2)
            if v1 == v0:   # empty value: the two quotes
                v0, v1 = v0 - 1, v1 + 1
            U[(row, col)] = {"open": (a + pm.start(), a + pm.start() + 1), "key": (a + pm.start(1), a + pm.end(1)),
                             "value": (a + v0, a + v1), "pair": (a + pm.start(), a + pm.end())}
    return U, line_first, (s, e)


def seeded_tables(seed, n, R, C):
    """The n synthetic tables of a seed with their targets (rows 2.., columns 2..), as the head analyses use them."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        rows = synth_rows(rng, R, C)
        out.append((rows, rng.randint(2, R), rng.randint(2, C)))
    return out
