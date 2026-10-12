#!/usr/bin/env python
"""Accuracy by needle position on the MMTU needle task (value -> position), one panel per serialization, from
the full-set outputs in results/fullset (the position heat-map figure of the paper).

Bins of five rows / five columns (MMTU's ``round_by_interval``). Panels: the serialization baselines under
the answer protocol and CCS (per-task tag shapes, the full-set run of the paper).

    python figures/fig_position_heatmaps.py [--acc]      # --acc writes the accuracy under each title

Writes figures/out/fig_position_heatmaps.{pdf,png}.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccs.mmtu_eval import parse_niht, score_niht  # noqa: E402
from ccs.utils import read_jsonl  # noqa: E402

OUT = ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)
FULL = ROOT / "results" / "fullset"
PANELS = [("Markdown", "markdown_protocol"), ("HTML", "html_own_protocol"), ("CSV", "csv_protocol"),
          ("JSON", "jsonl_protocol"), ("CCS", "full")]
CMAP = LinearSegmentedColormap.from_list("mmtu", ["#F0496E", "#EBB839", "#0CD79F"])   # MMTU's own ramp
SHOW_ACC = "--acc" in sys.argv
DATA = {r["id"]: r for r in read_jsonl(ROOT / "data" / "processed" / "mmtu_niht.jsonl")}


def grid(path, width=5):
    out = {r["id"]: r for r in read_jsonl(path)}
    rows = []
    b = lambda x: (int((x - 1) / width) + 1) * width  # noqa: E731
    for rid, rec in DATA.items():
        r = out.get(rid)
        pred = parse_niht(r["text"]) if r is not None else "JSONParsingError"
        rows.append(dict(row_bin=b(rec["meta"]["needle_row_idx"]), col_bin=b(rec["meta"]["needle_col_idx"]),
                         correct=score_niht(rec["gt"], pred)["correct"]))
    df = pd.DataFrame(rows)
    piv = df.groupby(["row_bin", "col_bin"])["correct"].mean().unstack("col_bin").sort_index().sort_index(axis=1)
    return piv, df["correct"].mean(), len(df)


def main():
    panels = []
    for title, preset in PANELS:
        f = FULL / f"qwen3-4b-2507__{preset}" / "outputs_mmtu_niht.jsonl"
        if f.exists():
            panels.append((title, *grid(f)))
        else:
            print("missing", f)
    ncol = len(panels)
    width = 8.0
    fig, axes = plt.subplots(1, ncol, figsize=(width, width / ncol * 1.15 + 0.4), squeeze=False)
    cbar_ax = fig.add_axes([0.915, 0.25, 0.012, 0.5])
    for i, (ax, (title, piv, acc, n)) in enumerate(zip(axes.flat, panels)):
        sns.heatmap(piv, ax=ax, cmap=CMAP, vmin=0, vmax=1, square=True, cbar=(i == 0), cbar_ax=cbar_ax,
                    linewidths=0.4, linecolor="white", cbar_kws={"label": "accuracy"})
        ax.set_title(f"{title}\naccuracy {acc:.2f}" if SHOW_ACC else title, fontsize=9)
        labels = [f"{b - 4}–{b}" for b in piv.columns]
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels if i == 0 else [], rotation=0, fontsize=6)
        ax.set_xlabel("needle column", fontsize=8)
        ax.set_ylabel("needle row" if i == 0 else "", fontsize=8)
        ax.tick_params(length=0)
        print(f"{title:12s} n={n} accuracy={acc:.3f}")
    cbar_ax.tick_params(labelsize=7)
    cbar_ax.yaxis.label.set_size(8)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.86, bottom=0.14, wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_position_heatmaps.{ext}", dpi=200, bbox_inches="tight")
    print("saved", OUT / "fig_position_heatmaps.pdf")


if __name__ == "__main__":
    main()
