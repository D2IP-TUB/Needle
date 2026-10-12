#!/usr/bin/env python
"""Aggregate the per-set metrics of a scripts/run_fullset.py directory the paper's way.

    python scripts/score_fullset.py results/fullset/qwen3-4b-2507__ccs [reference_dir]

MMTU accuracy = mean(needle, table-locate); NIAT exact match = mean(cell-locating, cropped, cell-lookup);
score = mean(MMTU accuracy, NIAT exact match). A second directory is compared set by set.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark  # noqa: E402

SETS = ["mmtu_niht", "mmtu_table_locate", "niat_cell_locating", "niat_cropped", "niat_cell_lookup"]


def load(run):
    acc, n = {}, {}
    for s in SETS:
        p = Path(run) / f"metrics_{s}.json"
        if p.exists():
            m = json.loads(p.read_text())
            acc[s], n[s] = m["acc"], m["n"]
    return acc, n


def main():
    run = Path(sys.argv[1])
    acc, n = load(run)
    ref = load(sys.argv[2])[0] if len(sys.argv) > 2 else {}
    print(f"{'set':22s}{'n':>8s}{'run':>8s}{'ref':>8s}{'diff':>8s}")
    for s in SETS:
        a, b = acc.get(s), ref.get(s)
        f = lambda x: f"{100 * x:8.1f}" if x is not None else f"{'-':>8s}"  # noqa: E731
        print(f"{s:22s}{n.get(s, 0):8d}{f(a)}{f(b)}{f(a - b) if a is not None and b is not None else '':>8s}")
    missing = [s for s in SETS if s not in acc]
    if missing:
        print("missing:", " ".join(missing))
        return
    m = benchmark.aggregate(acc)
    line = f"MMTU acc {100 * m['mmtu_acc']:.1f}  NIAT em {100 * m['niat_em']:.1f}  score {m['score']:.3f}"
    if all(s in ref for s in SETS):
        r = benchmark.aggregate(ref)
        line += f"   (ref: MMTU {100 * r['mmtu_acc']:.1f}  NIAT {100 * r['niat_em']:.1f}  score {r['score']:.3f}; diff {m['score'] - r['score']:+.3f})"
    print(line)


if __name__ == "__main__":
    main()
