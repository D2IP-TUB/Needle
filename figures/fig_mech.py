#!/usr/bin/env python
"""The figures of the mechanistic section, from results/mechanistic:

  probe    the absolute-index probe read-out, normalized by the table size (probe/)
  roles    layer x head maps of four attention roles, Markdown | JSON | CCS (heads/roles.json)
  heads    the eight strongest counting heads and mover heads under the three serializations
  stages   single heads reading the 5 x 5 orders table, Markdown and CCS side by side (heads/stages.*)
  numbers  the numbers quoted in the text: probe hit rates, head populations, knock-out results

    python figures/fig_mech.py [probe roles heads stages numbers]

Writes figures/out/fig_mech_*.{pdf,png}.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "mechanistic"
OUT = ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)
KINDS = ["markdown", "json", "tagged"]
NAMES = {"markdown": "Markdown", "json": "JSON", "tagged": "CCS"}
COLORS = {"markdown": "tab:red", "json": "tab:green", "tagged": "tab:blue"}
NH = 32
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name, dpi=200):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


# --------------------------------------------------------------------------------------------- probe
def probe_series(N, g, tagged=False):
    """(true index 1..N, mean read-out per index, twelve-table share within half a cell) for the counted axis
    g (1 = column, 0 = row): the first held-out table, averaged over its 8 lines."""
    R, C = (8, N) if g else (N, 8)
    name = "p16" if N == 16 else (f"col{N}" if g else f"row{N}")
    suf = "_tagged" if tagged else ""
    d = np.load(M / "probe" / f"probe_{R}x{C}{suf}.npy", allow_pickle=True).item()
    t, p = d["rc"][:, g], d[name][:, g]
    idx = np.arange(1, N + 1)
    m = np.array([p[t == i].mean() for i in idx])
    s = json.load(open(M / "probe" / f"probe_sizes{suf}.json"))[f"{R}x{C}_{name}_{'col' if g else 'row'}"]["half"]
    return idx, m, s


def probe():
    SIZES = [16, 32, 64, 128]
    COL = dict(zip(SIZES, plt.cm.viridis(np.linspace(0.0, 0.85, len(SIZES)))))
    RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}
    with sns.axes_style("white"), sns.plotting_context("paper", rc=RC):
        fig, ax = plt.subplots(2, 1, figsize=(3.3, 5.4))
        for k, g in enumerate([1, 0]):
            a = ax[k]
            axis = "column" if g else "row"
            for N in SIZES:
                idx, m, s = probe_series(N, g)
                sns.lineplot(x=(idx - 1) / (N - 1), y=(m - 1) / (N - 1), ax=a, color=COL[N], lw=1, estimator=None, sort=False,
                             label=f"{N} {axis}s ({s:.0%} of cells correct)")
            sns.lineplot(x=[0, 1], y=[0, 1], ax=a, color="k", lw=0.7, linestyle="--", label="perfect prediction")
            a.set_title(f"{axis} index, tables of 8 {'rows' if g else 'columns'}")
            a.set_xlim(0, 1.02)
            a.set_ylim(-0.04, 1.04)
            a.set_xlabel(f"true {axis} index, normalized (0 = first, 1 = last)")
            a.set_ylabel("predicted index, normalized")
            a.legend(frameon=False, fontsize=6.3, loc="upper left")
        sns.despine(fig)
        fig.tight_layout(h_pad=1.2)
        save(fig, "fig_mech_probe")


# --------------------------------------------------------------------------------------------- roles, heads
def roles():
    o = json.load(open(M / "heads" / "roles.json"))
    ROWS = [("delimiter $\\to$ earlier delimiters of its line", "count"), ("value $\\to$ header cell of its column", "bind_col_header"),
            ("value $\\to$ start of its line", "bind_line_start"), ("answer position $\\to$ target value", "ans_value")]
    fig, ax = plt.subplots(len(ROWS), 3, figsize=(3.6, 5.2), sharex=True, sharey=True, constrained_layout=True)
    for i, (title, key) in enumerate(ROWS):
        X = {f: np.array(o[f][key]).T for f in KINDS}
        vmax = max(x.max() for x in X.values())
        for j, f in enumerate(KINDS):
            im = ax[i, j].imshow(X[f], aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax[i, j].set_title(f"{title}\n{NAMES[f]}" if j == 1 else f"\n{NAMES[f]}", fontsize=5.8, loc="center")
            ax[i, j].tick_params(labelsize=5.5, length=2)
            for sp in ax[i, j].spines.values():
                sp.set_visible(True)
        ax[i, 0].set_ylabel("head", fontsize=7)
        cb = fig.colorbar(im, ax=ax[i, :], fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6)
    for a in ax[-1]:
        a.set_xlabel("layer", fontsize=6.5)
        a.set_xticks(range(0, 36, 10))
    save(fig, "fig_mech_roles", dpi=300)


def heads():
    o = json.load(open(M / "heads" / "roles.json"))
    pop = json.load(open(M / "heads" / "populations.json"))
    A = lambda f, k: np.array(o[f][k])  # noqa: E731
    top = lambda X, k=8: [(int(i // NH), int(i % NH)) for i in np.argsort(-np.array(X), axis=None)[:k]]  # noqa: E731
    P = [("counting heads\n(delimiter $\\to$ earlier delimiters of its line)", top(o["markdown"]["count"]), "count"),
         ("mover heads\n(answer position $\\to$ target value)", [tuple(h) for h in pop["movers"]], "ans_value")]
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.3))
    w = 0.26
    for a, (title, hs, k) in zip(ax, P):
        x = np.arange(len(hs))
        for off, f in [(-w, "markdown"), (0, "json"), (w, "tagged")]:
            a.bar(x + off, [A(f, k)[l, h] for l, h in hs], w, color=COLORS[f], label=NAMES[f])
        a.set_xticks(x)
        a.set_xticklabels([f"L{l}H{h}" for l, h in hs], rotation=40, fontsize=6.5, ha="right", rotation_mode="anchor")
        a.set_title(title)
        a.set_ylim(0, 1)
        a.grid(axis="y", alpha=0.25)
    ax[0].set_ylabel("share of attention")
    ax[1].set_ylabel("share of attention")
    h, l = ax[1].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3)
    save(fig, "fig_mech_heads")


# --------------------------------------------------------------------------------------------- stages
# (title, query-token description, head, kind of query token, cell, role) of the paper's figure
STAGE_PANELS = [("Row counter", "the delimiter that opens cell (5,4)", (1, 31), "open", (5, 4), "rows"),
                ("Column counter", "the delimiter that opens cell (5,4)", (9, 6), "open", (5, 4), "line"),
                ("Coordinate head", "the value of cell (5,4)", (18, 3), "value", (5, 4), "coord"),
                ("Mover head", "the position that generates the answer, target cell (4,4)", (29, 11), "answer", (4, 4), "target")]


def stages(PANELS=STAGE_PANELS, out="fig_mech_stages"):
    meta = json.load(open(M / "heads" / "stages.json"))
    Z = np.load(M / "heads" / "stages.npz")
    panels = {}
    for fmt in ["markdown", "tagged"]:
        m = meta[fmt]
        text, offs = m["text"], m["offs"]
        row, col, kind, last = (np.array(m[k]) for k in ["row", "col", "kind", "last"])
        T = len(offs)
        tab = np.where(kind != "")[0]
        t0, t1 = int(tab[0]), int(tab[-1])
        cs = offs[t0][0]
        ce = cs + text[cs:].index("\n\n")
        dl = "|" if fmt == "markdown" else "["
        opens = {(row[i], col[i]): i for i in range(T) if kind[i] == "open"}
        vlast = {(row[i], col[i]): i for i in range(T) if last[i]}
        for k, (st, qd, (l, h), where, (r, c), role) in enumerate(PANELS):
            qt = {"open": lambda: opens[(r, c)], "value": lambda: vlast[(r, c)], "answer": lambda: T - 1}[where]()
            w = Z[f"{fmt}_L{l}H{h}"][qt].astype(np.float32)
            lastt = min(qt, t1)
            mx = max(w[t0:lastt + 1].max(), 1e-9)
            lines = [[]]
            for i in range(t0, lastt + 1):
                a, b = max(offs[i][0], cs), min(offs[i][1], ce)
                for j, part in enumerate(text[a:b].split("\n")):
                    if j:
                        lines.append([])
                    if part:
                        lines[-1].append((part, float(w[i] / mx), i == qt))
            own = (row == r) & (col == c) & (np.arange(T) < qt)
            if role == "line":
                s = f"{w[[opens[(r, cc)] for cc in range(1, c)]].sum():.2f} on the earlier {dl} of its line"
            elif role == "rows":
                s = f"{w[[t for (rr, cc), t in opens.items() if rr < r]].sum():.2f} on the {dl} of the earlier lines"
            elif role == "coord":
                s = (f"{w[own & np.isin(kind, ['c', 'cdig'])].sum():.2f} on its column tag c{c}" if fmt == "tagged"
                     else f"{w[opens[(r, c)]]:.2f} on the | that opens its cell")
            else:
                s = f"{w[(row == r) & (col == c) & (kind == 'value')].sum():.2f} on the target value"
            panels[(k, fmt)] = (lines, f"L{l}H{h}: {s}")
            print(fmt, st, f"L{l}H{h}", s)
    _draw_stages(panels, [(st, qd) for st, qd, *_ in PANELS], out)


def _draw_stages(panels, heads, name):
    """panels[(k, fmt)] = (lines of (text, weight, is query token), subtitle); heads[k] = (title, query description)."""
    width = lambda f: max(sum(len(p) for p, _, _ in ln) for (k, ff), (L, _) in panels.items() if ff == f for ln in L)  # noqa: E731
    Wm, Wt = max(width("markdown"), 44), max(width("tagged"), 92)
    GAP = 4
    NC = Wm + GAP + Wt + 1
    FW = 6.9
    fs = FW * 72 / NC / 0.6021
    lh = 1.75
    ys, y = {}, 0.0
    for k in range(len(heads)):
        ys[k] = y
        y += 2.6 * lh + max(len(panels[(k, f)][0]) for f in ["markdown", "tagged"]) * lh + 0.8 * lh
    H = y - 0.6 * lh
    fig = plt.figure(figsize=(FW, FW * H / NC))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, NC)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for k, (st, qd) in enumerate(heads):
        ax.text(0, ys[k] + 0.6 * lh, f"{st} — query token: {qd}", fontsize=fs * 1.05, fontweight="bold", va="center")
        for fmt, x0 in [("markdown", 0), ("tagged", Wm + GAP)]:
            lines, title = panels[(k, fmt)]
            ax.text(x0, ys[k] + 1.7 * lh, f"{NAMES[fmt]}, {title}", fontsize=fs * 0.98, va="center")
            for li, ln in enumerate(lines):
                x = x0
                yy = ys[k] + 2.6 * lh + li * lh
                for part, v, isq in ln:
                    v = float(np.sqrt(min(max(v, 0), 1)))
                    ax.add_patch(Rectangle((x, yy), len(part), lh * 0.94, fc=(1, 1 - 0.78 * v, 1 - 0.92 * v), ec="none"))
                    if isq:
                        ax.add_patch(Rectangle((x, yy), len(part), lh * 0.94, fc="none", ec=(0, 0.6, 0.78), lw=1.0))
                    ax.text(x, yy + lh * 0.5, part, fontsize=fs, family="DejaVu Sans Mono", va="center", ha="left")
                    x += len(part)
    save(fig, name, dpi=250)


# --------------------------------------------------------------------------------------------- numbers
def numbers():
    pj = json.load(open(M / "probe" / "probe_sizes.json"))
    print("probe (Markdown, layer 17), twelve held-out tables: mean |error| / share within half a cell")
    for N in [16, 32, 64, 128]:
        for g, nm, key in [(1, "columns", f"8x{N}_{'p16' if N == 16 else 'col%d' % N}_col"), (0, "rows", f"{N}x8_{'p16' if N == 16 else 'row%d' % N}_row")]:
            key = key if N != 16 else ("16x16_p16_col" if g else "16x16_p16_row")
            if key in pj:
                print(f"  {N:4d} {nm:8s} {pj[key]['mae']:.2f} / {pj[key]['half']:.2f}")
    pt = M / "probe" / "probe_sizes_tagged.json"
    if pt.exists():
        pj = json.load(open(pt))
        print("probe on CCS tables (read at the opening bracket):")
        for N in [16, 64, 128]:
            kc = "16x16_p16_col" if N == 16 else f"8x{N}_col{N}_col"
            kr = "16x16_p16_row" if N == 16 else f"{N}x8_row{N}_row"
            print(f"  {N:4d} columns {pj[kc]['mae']:.2f} rows {pj[kr]['mae']:.2f}")
    cc = M / "heads" / "count_code.json"
    if cc.exists():
        st = json.load(open(cc))["stats"]
        print("count code fits per layer (R^2 of a+b/j vs a+b log j):", {L: (round(v["r2_inv"], 2), round(v["r2_log"], 2)) for L, v in st.items()})
    pop = json.load(open(M / "heads" / "populations.json"))
    print("head populations:", {k: len(v) for k, v in pop.items()})
    o = json.load(open(M / "heads" / "roles.json"))
    cnt = np.array(o["markdown"]["count"]) > 0.5
    print(f"counting heads (Markdown > 0.5): {cnt.sum()}, of them > 0.5 on the CCS bracket: {(cnt & (np.array(o['tagged']['count']) > 0.5)).sum()}, "
          f"JSON heads > 0.5: {(np.array(o['json']['count']) > 0.5).sum()}")
    for f in KINDS:
        av = np.array(o[f]["ans_value"])
        print(f"  {NAMES[f]:8s} movers: max share on the target {av.max():.2f}, heads above 0.5: {(av > 0.5).sum()}")
    print("knock-outs (accuracy / count R^2 / mover attention; random sets of the same size in parentheses):")
    for f in sorted(os.listdir(M / "ablation")):
        res = json.load(open(M / "ablation" / f))
        g = lambda ab, k: [r for r in res if r["ablation"] == ab][0][k]  # noqa: E731
        rnd = lambda ab, k: np.mean([r[k] for r in res if r["ablation"].startswith(f"random like {ab},")])  # noqa: E731
        print(f"  {f[:-5]:14s} none: acc {g('none', 'correct'):.2f} R2 {g('none', 'count_r2'):.2f} mover {g('none', 'mover'):.2f}")
        for ab in sorted({r["ablation"] for r in res if r["ablation"] != "none" and not r["ablation"].startswith("random")}):
            print(f"    - {ab:36s} acc {g(ab, 'correct'):.2f} ({rnd(ab, 'correct'):.2f})  R2 {g(ab, 'count_r2'):.2f} ({rnd(ab, 'count_r2'):.2f})  mover {g(ab, 'mover'):.2f} ({rnd(ab, 'mover'):.2f})")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["probe", "roles", "heads", "stages", "numbers"]:
        globals()[w]()
