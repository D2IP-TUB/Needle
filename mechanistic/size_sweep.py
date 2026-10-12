#!/usr/bin/env python
"""The size x serialization table of the paper: accuracy of the cell-locating query on synthetic tables, ten
serializations, 100 tables per size, the tables of a size shared by all serializations.

The sweep is split into parts (one GPU each; the largest prompts reach 59K tokens):
    python mechanistic/size_sweep.py a|b|c|d|e|f
    python mechanistic/size_sweep.py table         # LaTeX rows of the table from results/mechanistic/sizes
    python mechanistic/size_sweep.py errors        # which coordinate the wrong answers miss

Within a part the tables of a size are generated in the order the sizes first appear, from one seeded
generator (seed 11), so a part gives the same tables whatever the list of serializations.
"""

import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as sz  # noqa: E402
from common import OUT  # noqa: E402

F = sz.FORMATS
S = lambda sizes: [(f, R, C) for R, C in sizes for f in F]  # noqa: E731
PARTS = {"a": S([(512, 8)]), "b": S([(64, 64)]), "c": S([(256, 8), (8, 192), (8, 128)]),
         "d": S([(128, 8), (8, 64), (8, 8), (64, 8), (32, 8), (8, 32), (8, 16)]),
         "e": S([(8, 256)]), "f": S([(16, 8)])}

# columns of the paper's table: 8 rows with growing columns, 64 x 64, 8 columns with growing rows
SIZES = [(8, 8), (8, 16), (8, 64), (8, 256), (64, 64), (64, 8), (128, 8), (512, 8)]
ROWS = ["markdown", "rowpre_md", "coltag", "tagged", "rowpre_coltag", "valuefirst", "plain", "csv", "jsonl", "nametag"]


def pooled():
    agg = collections.defaultdict(list)
    for f in sorted(glob.glob(str(OUT / "sizes" / "sz_*.json"))):
        for r in json.load(open(f)):
            agg[(r["fmt"], r["R"], r["C"])].append(r)
    return agg


def table():
    agg = pooled()

    def cell(fmt, R, C):
        rs = agg.get((fmt, R, C))
        if not rs:
            return "--"
        s = f"{np.mean([r['correct'] for r in rs]):.2f}"
        return r"\textbf{1.00}" if s == "1.00" else s

    print("serialization & " + " & ".join(f"{R}x{C}" for R, C in SIZES) + r" \\")
    for fmt in ROWS:
        print(f"{sz.LABELS[fmt]} & " + " & ".join(cell(fmt, R, C) for R, C in SIZES) + r" \\")
        print("%   n: " + " ".join(f"{len(agg.get((fmt, R, C), [])):3d}" for R, C in SIZES))


def errors():
    """Of the wrong answers that name a cell of the table: share in the target row / target column."""
    agg = pooled()
    for fmt in ROWS:
        w = [r for k, rs in agg.items() if k[0] == fmt for r in rs if not r["correct"]]
        loc = [r for r in w if r["prow"] is not None]
        if w:
            print(f"{sz.LABELS[fmt]}: {len(w)} wrong, {len(loc)} located in the table; same row "
                  f"{np.mean([r['prow'] == r['tr'] for r in loc]) if loc else float('nan'):.2f}, same column "
                  f"{np.mean([r['pcol'] == r['tc'] for r in loc]) if loc else float('nan'):.2f}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what in PARTS:
        sz.run("sizes_" + what, PARTS[what], N=100)
    else:
        globals()[what]()
