#!/usr/bin/env python
"""The attention figures of the paper, from results/mechanistic/attention (mechanistic/attention_niat.py and
attention_crops.py):

  cells8   cell-to-cell attention on the 8 x 8 table, Markdown | JSON | CCS (JSON shows rows 2-8: it has no header line)
  cells16  the same on the 16 x 16 crop;  cells32  on the 32 x 32 crop (needs attn_cc_32_*.npz, 66 MB each)
  answer   the answer position -> cells at layer 24 for four queries, one row per serialization
  layers   per layer: rate at which the target is the most attended cell, and attention share on the target

    python figures/fig_attention.py [cells8 cells16 cells32 answer layers]

Writes figures/out/fig_attention_*.{pdf,png} and prints the numbers quoted in the text.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "mechanistic" / "attention"
OUT = ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)
KINDS = ["markdown", "json", "tagged"]
NAMES = {"markdown": "Markdown", "json": "JSON", "tagged": "CCS"}
COLORS = {"markdown": "tab:red", "json": "tab:green", "tagged": "tab:blue"}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})


def save(fig, name, dpi=300):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


def cc_mean(size, k):
    """Cell-to-cell attention averaged over layers, each layer normalised over the attended cells."""
    f = A / f"attn_cc_{size}_{k}.npz"
    if not f.exists() and size == 8:
        f = A / f"attn_{k}.npz"     # the 8 x 8 map of attention_niat.py
    cc = np.load(f)["cc"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cc / np.nansum(cc, 2, keepdims=True)).mean(0)


def cells(size):
    acc = json.load(open(A / "attn_cc_acc.json"))
    N = size * size
    idx = np.arange(N)
    low = idx[:, None] > idx[None, :]
    CC = {k: cc_mean(size, k) for k in KINDS}
    shown = {"markdown": idx, "tagged": idx, "json": np.arange(size, N)}
    vmax = float(np.ceil(np.nanpercentile(np.concatenate([np.nan_to_num(CC[k])[np.ix_(shown[k], shown[k])][low[np.ix_(shown[k], shown[k])]] for k in KINDS]), 99) * 200) / 200)
    step = 1 if size <= 8 else 2 if size <= 16 else 4
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    for ax, k in zip(axes, KINDS):
        Kk = shown[k]
        D = np.where(low, np.nan_to_num(CC[k]), 0.0)[np.ix_(Kk, Kk)]
        np.fill_diagonal(D, np.nan)
        im = ax.imshow(D, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        first = Kk[0] // size + 1
        rows = list(range(first, size + 1, step))
        tk = [(r - first) * size for r in rows]
        ax.set_xticks(tk)
        ax.set_xticklabels([f"r{r}" for r in rows], fontsize=7, rotation=90)
        ax.set_yticks(tk)
        ax.set_yticklabels([f"r{r}" for r in rows], fontsize=7)
        ax.tick_params(length=2, pad=1.5)
        if size <= 16:
            for b in range(size, len(Kk), size):
                ax.axhline(b - 0.5, color="w", lw=0.2, alpha=0.3)
                ax.axvline(b - 0.5, color="w", lw=0.2, alpha=0.3)
        a = acc.get(f"{size}_{k}")
        ax.set_title(f"{NAMES[k]}\n{a['correct']}/{a['n']} queries correct" if a else NAMES[k], fontsize=8, linespacing=1.15)
        if k == "markdown":
            ax.set_ylabel("cell that attends", fontsize=8)
    fig.supxlabel("cell attended to (row by row, tick = first cell of a table row)", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.035, aspect=30)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.locator = matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 5, 10])
    cb.formatter = matplotlib.ticker.FormatStrFormatter("%g")
    cb.update_ticks()
    cb.set_label("share of the cell's attention on cells", fontsize=8)
    # reading shares (same column / same row) on the data-row cells, quoted in the appendix
    rows_of, cols_of = idx // size, idx % size
    same_col = (cols_of[:, None] == cols_of[None, :]) & low
    same_row = (rows_of[:, None] == rows_of[None, :]) & low
    live = idx // size >= 1
    sub = np.ix_(live, live)
    for k in KINDS:
        X = np.where(low, np.nan_to_num(CC[k]), 0.0)[sub]
        tot = X[low[sub]].sum()
        print(f"{size}x{size} {NAMES[k]:8s} same column {X[same_col[sub]].sum() / tot:.2f}  same row {X[same_row[sub]].sum() / tot:.2f}")
    save(fig, f"fig_attention_cells_{size}x{size}", dpi=300 if size < 32 else 450)


def _answer_data():
    P = {k: json.load(open(A / f"preds_{k}.json")) for k in KINDS}
    ANS = {k: np.load(A / f"attn_{k}.npz")["ans"].astype(np.float64) for k in KINDS}   # L x 64 queries x 64 cells
    return P, ANS


def answer(layer=24):
    R = C = 8
    P, ANS = _answer_data()
    gt = {(p["row"], p["col"]): p["gt"] for p in P["markdown"]}
    EX = [(2, 3), (3, 6), (6, 4), (7, 2)]
    fig, axes = plt.subplots(len(KINDS), len(EX), figsize=(3.35, 2.55))
    for r, k in enumerate(KINDS):
        a = ANS[k][layer]
        an = a / a.sum(1, keepdims=True)
        for c, (m, n) in enumerate(EX):
            ax = axes[r, c]
            q = (m - 1) * C + n - 1
            p = P[k][q]
            im = ax.imshow(an[q].reshape(R, C), cmap="magma", vmin=0, vmax=0.5, interpolation="nearest")
            ans_cells = [cell for cell, v in gt.items() if v == p["pred"]]
            if len(ans_cells) == 1:
                ax.add_patch(Rectangle((ans_cells[0][1] - 1.5, ans_cells[0][0] - 1.5), 1, 1, fill=False, ec="w", lw=1.0, ls=(0, (2, 1.2))))
            ax.add_patch(Rectangle((n - 1.5, m - 1.5), 1, 1, fill=False, ec="cyan", lw=1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color("0.5")
            if r == 0:
                ax.set_title(f"query ({m},{n})", fontsize=6.5, pad=2.5)
            if c == 0:
                ax.set_ylabel(f"{NAMES[k]}\nL{layer}", fontsize=6)
    fig.subplots_adjust(left=0.1, right=0.88, top=0.93, bottom=0.02, wspace=0.06, hspace=0.06)
    cax = fig.add_axes([0.90, 0.03, 0.022, 0.89])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label("share of the attention on cells", fontsize=5.5)
    save(fig, "fig_attention_answer")
    # where the most attended cell lies at this layer, for the queries of rows 2-8
    tgt = np.array([(p["row"] - 1) * C + p["col"] - 1 for p in P["markdown"]])
    live_q = tgt // C >= 1
    for k in KINDS:
        a = ANS[k][layer]
        an = a / a.sum(1, keepdims=True)
        top = an[live_q].argmax(1)
        tq = tgt[live_q]
        top_all = an.argmax(1)
        below = lambda t, g: ((t == g + C) | (t == g + C + 1)).sum()  # noqa: E731  the cell below the target, or below and to its right
        print(f"L{layer} {NAMES[k]:8s}: target most attended {int((top == tq).sum())}/{int(live_q.sum())}; most attended cell below / "
              f"below-right of the target {int(below(top_all, tgt))} of {len(tgt)} queries ({int(below(top, tq))} of the {int(live_q.sum())} with a "
              f"data-row target); queries correct {sum(p['correct'] for p in P[k])}/{len(P[k])}")


def layers():
    P, ANS = _answer_data()
    C = 8
    L = ANS["markdown"].shape[0]
    tgt = np.array([(p["row"] - 1) * C + p["col"] - 1 for p in P["markdown"]])
    live_q = tgt // C >= 1          # queries whose target is a data-row cell (JSON has no header cells)
    CUR = {}
    for k in KINDS:
        an = ANS[k] / ANS[k].sum(2, keepdims=True)
        CUR[k] = ([float(np.mean(an[l][live_q, tgt[live_q]])) for l in range(L)],
                  [float(np.mean(an[l][live_q].argmax(1) == tgt[live_q])) for l in range(L)])
        print(f"{NAMES[k]:8s} max attention share on the target {max(CUR[k][0]):.3f} at layer {int(np.argmax(CUR[k][0]))}; "
              f"max rate target most attended {max(CUR[k][1]):.2f}")
    CH = 1 / int(live_q.sum())
    X = np.arange(L)
    RC = {"font.size": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}

    def panel(ax, m, ylab, log=False):
        for k in KINDS:
            sns.lineplot(x=X, y=CUR[k][m], ax=ax, color=COLORS[k], lw=1.4, label=NAMES[k], estimator=None, legend=False)
        ax.axhline(CH, color="gray", lw=0.8, ls=":", label=f"chance (1/{int(live_q.sum())})")
        ax.set_xlim(0, L - 1)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.tick_params(length=2.5, bottom=True, left=True)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.008, 1)
            T = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
            ax.set_yticks(T)
            ax.set_yticklabels([f"{t:.2f}" for t in T])
            ax.minorticks_off()
        else:
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

    with sns.axes_style("white"), sns.plotting_context("paper", rc=RC):
        fig, ax = plt.subplots(2, 1, figsize=(3.35, 3.1), sharex=True)
        panel(ax[0], 1, "target most attended")
        panel(ax[1], 0, "attention on target", log=True)
        ax[1].set_xlabel("layer")
        sns.despine(fig)
        fig.tight_layout(h_pad=0.8)
        fig.align_ylabels(ax)
        fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.8)
        save(fig, "fig_attention_layers", dpi=250)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["cells8", "cells16", "cells32", "answer", "layers"]:
        if w.startswith("cells"):
            size = int(w[5:])
            if all(os.path.exists(A / f"attn_cc_{size}_{k}.npz") or (size == 8 and os.path.exists(A / f"attn_{k}.npz")) for k in KINDS):
                cells(size)
            else:
                print(f"cells{size}: attn_cc_{size}_*.npz missing, run mechanistic/attention_crops.py --sizes {size}")
        else:
            globals()[w]()
