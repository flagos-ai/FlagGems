#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC-GEM W8A16 Benchmark — Plotting Script
=========================================
Reads qcgem_w8a16_data.csv and generates PNG + PDF charts (dark theme).

Usage:
    python plot_results_w8a16.py

Output:
    qcgem_w8a16_benchmark.png
    qcgem_w8a16_benchmark.pdf
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_W8A16 = os.path.join(SCRIPT_DIR, "qcgem_w8a16_data.csv")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0b0c10",
    "axes.facecolor":   "#13141c",
    "axes.edgecolor":   "#1e2030",
    "axes.labelcolor":  "#8888aa",
    "xtick.color":       "#55556a",
    "ytick.color":       "#55556a",
    "text.color":        "#c8c8d0",
    "grid.color":        "#1a1b28",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans, sans-serif",
})

CAT_COLORS = {
    "Down projection":  "#5b8df0",
    "Up projection":    "#62d17a",
    "Gate projection":  "#f09b5b",
    "Router":           "#c97df0",
}

PALETTE = {
    "W8A16": "#5b8df0",
    "FP16":  "#f09b5b",
}


def load_data(path):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "idx":         int(row["idx"]),
                "category":    row["category"],
                "shape":       row["shape"],
                "m":           int(row["m"]),
                "n":           int(row["n"]),
                "k":           int(row["k"]),
                "qc_ms":       float(row["qc_ms"]),
                "fp16_ms":     float(row["fp16_ms"]),
                "qc_tflops":   float(row["qc_tflops"]),
                "fp16_tflops": float(row["fp16_tflops"]),
                "speedup":     float(row["speedup"]),
            })
    return rows


def shade_cats(ax, cats, x, w):
    for i, cat in enumerate(cats):
        color = CAT_COLORS.get(cat, "#888888")
        ax.axvspan(x[i] - w/2, x[i] + w/2, alpha=0.12, color=color, zorder=0)


def plot_w8a16(data):
    cats     = [d["category"] for d in data]
    xlabels  = [d["shape"]    for d in data]
    n        = len(data)

    w8_ms    = [d["qc_ms"]    for d in data]
    fp16_ms  = [d["fp16_ms"]  for d in data]
    w8_sp    = [d["speedup"]  for d in data]
    w8_tf    = [d["qc_tflops"] for d in data]
    fp16_tf  = [d["fp16_tflops"] for d in data]

    x = np.arange(n)
    w_single = 0.5
    w_half   = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(
        "QC-GEM W8A16 vs PyTorch FP16 GEMM — Qwen3.5 MoE Shapes (H20 GPU)",
        fontsize=15, fontweight="bold", color="#e8e8f0", y=0.99,
    )

    # ── (0,0) Speedup ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    shade_cats(ax, cats, x, w_single)
    sp_cols = ["#62d17a" if s >= 1.0 else "#e05555" for s in w8_sp]
    ax.bar(x, w8_sp, color=sp_cols, edgecolor="none", width=w_single)
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("Speedup vs FP16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("×")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_ylim(0, max(w8_sp) * 1.18)
    for i, v in enumerate(w8_sp):
        ax.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=7.5, color="#aaaacc")

    # ── (0,1) Latency ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    shade_cats(ax, cats, x, w_half * 2 + 0.02)
    ax.bar(x - w_half/2, fp16_ms, width=w_half, label="FP16 ref",  color=PALETTE["FP16"],  edgecolor="none")
    ax.bar(x + w_half/2, w8_ms,   width=w_half, label="QC W8A16", color=PALETTE["W8A16"], edgecolor="none")
    ax.set_title("Latency — FP16 ref / W8A16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("ms")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.4)

    # ── (1,0) TFLOPS ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    shade_cats(ax, cats, x, w_half * 2 + 0.02)
    ax.bar(x - w_half/2, fp16_tf, width=w_half, label="FP16 ref",  color=PALETTE["FP16"],  edgecolor="none")
    ax.bar(x + w_half/2, w8_tf,   width=w_half, label="QC W8A16", color=PALETTE["W8A16"], edgecolor="none")
    ax.set_title("TFLOPS — FP16 ref / W8A16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("TFLOPS")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.4)

    # ── (1,1) Speedup by N-dim ───────────────────────────────────────────────
    ax = axes[1, 1]
    unique_n = sorted(set(d["n"] for d in data))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for idx, n_val in enumerate(unique_n):
        mask = [d["n"] == n_val for d in data]
        m_vals = [d["m"] for d in data if d["n"] == n_val]
        sp_vals = [d["speedup"] for d in data if d["n"] == n_val]
        sorted_pairs = sorted(zip(m_vals, sp_vals))
        ms, sps = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.plot(
            [m/1024 for m in ms], sps,
            "o-", linewidth=1.8, markersize=6,
            color=cmap(idx),
            label=f"N={n_val}",
        )
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("Speedup by N-dimension", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xlabel("M (tokens) in k")
    ax.set_ylabel("Speedup ×")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.4)

    # ── Category legend ─────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    fig.legend(
        handles=patches, loc="lower center", ncol=4,
        fontsize=9, framealpha=0.35,
        bbox_to_anchor=(0.5, -0.005),
    )

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.003,
        "FlagGems QC-GEM W8A16 Kernel  ·  FP16 reference = PyTorch GEMM  ·  GPU = NVIDIA H20  ·  Qwen3.5 MoE shapes",
        ha="center", fontsize=8, color="#44445a",
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.97])
    return fig


if __name__ == "__main__":
    if not os.path.exists(CSV_W8A16):
        print(f"[ERROR] {CSV_W8A16} not found. Run benchmark first.", file=sys.stderr)
        sys.exit(1)

    print(f"[DATA] Loading: {CSV_W8A16}")
    data = load_data(CSV_W8A16)
    print(f"[DATA] {len(data)} rows — W8A16")

    fig = plot_w8a16(data)

    out_png = os.path.join(SCRIPT_DIR, "qcgem_w8a16_benchmark.png")
    out_pdf = os.path.join(SCRIPT_DIR, "qcgem_w8a16_benchmark.pdf")

    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="#0b0c10")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="#0b0c10")

    print(f"[OK] PNG → {out_png}")
    print(f"[OK] PDF → {out_pdf}")
