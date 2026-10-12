#!/usr/bin/env python
"""Materialise the full prompt sets under data/processed/.

    python scripts/build_data.py            # full sizes (default)
    python scripts/build_data.py --quick    # tiny subsets for a smoke test

Full sizes
    mmtu_niht            999 prompts (all of MMTU's needle task)
    mmtu_table_locate   1000 prompts (all of MMTU's table-locate task)
    niat_cell_locating   360 tables x 2 formats x 20 cells      = 14,396
    niat_cell_lookup     360 tables x 2 formats x <=20 questions = 14,054
    niat_cropped         7 sizes x 15 tables x N*N cells        = 48,720

MMTU is downloaded from the Hugging Face hub on first use; the NIAT source tables are in data/niat.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ccs import DATA_DIR, PROCESSED_DIR  # noqa: E402
from ccs.mmtu_data import build_mmtu  # noqa: E402
from ccs.niat_data import CROP_SIZES, build_niat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--quick", action="store_true", help="small subsets for smoke testing")
    ap.add_argument("--benchmarks", nargs="+", default=["mmtu", "niat"], choices=["mmtu", "niat"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells-per-table", type=int, default=20)
    ap.add_argument("--questions-per-table", type=int, default=20)
    ap.add_argument("--tables-per-size", type=int, default=15)
    ap.add_argument("--crop-sizes", type=int, nargs="+", default=CROP_SIZES)
    ap.add_argument("--crop-cells-per-table", type=int, default=0, help="0 = every cell")
    ap.add_argument("--mmtu-limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.cells_per_table, args.questions_per_table = 3, 3
        args.tables_per_size, args.crop_cells_per_table = 2, 12
        args.mmtu_limit = 24
        limit_tables = 4
    else:
        limit_tables = None

    summary = {}
    if "mmtu" in args.benchmarks:
        summary["mmtu"] = build_mmtu(out, limit=args.mmtu_limit, seed=args.seed)
    if "niat" in args.benchmarks:
        summary["niat"] = build_niat(
            DATA_DIR / "niat", out,
            cells_per_table=args.cells_per_table, questions_per_table=args.questions_per_table,
            tables_per_size=args.tables_per_size, sizes=args.crop_sizes,
            crop_cells_per_table=args.crop_cells_per_table, seed=args.seed, limit_tables=limit_tables,
        )
    json.dump(summary, open(out / "build_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
