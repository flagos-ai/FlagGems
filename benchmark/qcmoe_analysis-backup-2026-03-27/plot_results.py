#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC-MoE W8A16 / W4A16 Benchmark — Combined Plotting Script
==========================================================
Reads qcmoe_w8a16_data.csv AND qcmoe_w4a16_data.csv, generates
one PNG + PDF with four panels:

  (0,0)  Speedup — W8A16 vs FP16
  (0,1)  Speedup — W4A16 vs FP16
  (1,0)  Latency — W8A16 + W4A16 + FP16
  (1,1)  TFLOPS  — W8A16 + W4A16 + FP16

Usage:
    python plot_results.py

Prerequisites:
    qcmoe_w8a16_data.csv  (run: pytest -k "sglang_w8a16")
    qcmoe_w4a16_data.csv  (run: pytest -k "sglang_w4a16")
"""

import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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


# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_W8A16 = os.path.join(SCRIPT_DIR, "qcmoe_w8a16_data.csv")
CSV_W4A16 = os.path.join(SCRIPT_DIR, "qcmoe_w4a16_data.csv")


# ── Load CSV ─────────────────────────────────────────────────────────────────
def load_data(path, mode):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mode":        mode,
                "idx":         int(row["idx"]),
                "category":    row["category"],
                "shape":       row["shape"],
                "seq_len":     int(row["seq_len"]),
                "hidden_dim":  int(row["hidden_dim"]),
                "inter_dim":   int(row["inter_dim"]),
                "qc_ms":       float(row["qc_ms"]),
                "fp16_ms":     float(row["fp16_ms"]),
                "qc_tflops":   float(row["qc_tflops"]),
                "fp16_tflops": float(row["fp16_tflops"]),
                "speedup":     float(row["speedup"]),
                "max_abs_err": float(row["max_abs_err"]),
            })
    return rows


# ── Colour palette ────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Seq len sweep":    "#5b8df0",
    "Hidden dim sweep": "#f09b5b",
    "Inter dim sweep":  "#62d17a",
    "Large shapes":     "#c97df0",
}

PALETTE = {
    "W8A16": "#5b8df0",
    "W4A16": "#62d17a",
    "FP16":  "#f09b5b",
}


# ── Category shading helper ────────────────────────────────────────────────────
def shade_cats(ax, cats, x, w):
    for i, cat in enumerate(cats):
        color = CAT_COLORS[cat]
        ax.axvspan(x[i] - w/2, x[i] + w/2, alpha=0.12, color=color, zorder=0)


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot(w8a16_data, w4a16_data):
    xlabels = [f"S={d['seq_len']}" for d in w8a16_data]
    cats    = [d["category"]       for d in w8a16_data]

    w8_speedup = [d["speedup"]     for d in w8a16_data]
    w4_speedup = [d["speedup"]     for d in w4a16_data]

    w8_ms   = [d["qc_ms"]    for d in w8a16_data]
    w4_ms   = [d["qc_ms"]    for d in w4a16_data]
    fp16_ms = [d["fp16_ms"]  for d in w8a16_data]

    w8_tflops   = [d["qc_tflops"]    for d in w8a16_data]
    w4_tflops   = [d["qc_tflops"]    for d in w4a16_data]
    fp16_tflops = [d["fp16_tflops"] for d in w8a16_data]

    n = len(xlabels)
    x = np.arange(n)
    w_single = 0.26
    w_half    = 0.12

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(
        "QC-MoE W8A16 & W4A16 vs SGLang-style FP16 MoE",
        fontsize=16, fontweight="bold", color="#e8e8f0", y=0.99,
    )

    # ── (0,0) W8A16 Speedup ──────────────────────────────────────────────────
    ax = axes[0, 0]
    shade_cats(ax, cats, x, w_single)
    sp_cols = ["#62d17a" if s >= 1.0 else "#e05555" for s in w8_speedup]
    ax.bar(x, w8_speedup, color=sp_cols, edgecolor="none", width=w_single)
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("W8A16 — Speedup vs FP16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("×")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_ylim(0, max(w8_speedup) * 1.18)
    for i, v in enumerate(w8_speedup):
        ax.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=7, color="#aaaacc")

    # ── (0,1) W4A16 Speedup ──────────────────────────────────────────────────
    ax = axes[0, 1]
    shade_cats(ax, cats, x, w_single)
    sp_cols = ["#62d17a" if s >= 1.0 else "#e05555" for s in w4_speedup]
    ax.bar(x, w4_speedup, color=sp_cols, edgecolor="none", width=w_single)
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("W4A16 — Speedup vs FP16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("×")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_ylim(0, max(w4_speedup) * 1.18)
    for i, v in enumerate(w4_speedup):
        ax.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=7, color="#aaaacc")

    # ── (1,0) Latency ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    shade_cats(ax, cats, x, w_half * 3 + 0.01)
    ax.bar(x - w_half,       fp16_ms, width=w_half, label="FP16 Ref",  color=PALETTE["FP16"],  edgecolor="none")
    ax.bar(x,                w8_ms,   width=w_half, label="QC W8A16", color=PALETTE["W8A16"], edgecolor="none")
    ax.bar(x + w_half,       w4_ms,   width=w_half, label="QC W4A16", color=PALETTE["W4A16"], edgecolor="none")
    ax.set_title("Latency — FP16 Ref / W8A16 / W4A16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("ms")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=8, framealpha=0.4)

    # ── (1,1) TFLOPS ─────────────────────────────────────────────────────────
    ax = axes[1, 1]
    shade_cats(ax, cats, x, w_half * 3 + 0.01)
    ax.bar(x - w_half,       fp16_tflops, width=w_half, label="FP16 Ref",  color=PALETTE["FP16"],  edgecolor="none")
    ax.bar(x,                w8_tflops,   width=w_half, label="QC W8A16", color=PALETTE["W8A16"], edgecolor="none")
    ax.bar(x + w_half,       w4_tflops,  width=w_half, label="QC W4A16", color=PALETTE["W4A16"], edgecolor="none")
    ax.set_title("TFLOPS — FP16 Ref / W8A16 / W4A16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("TFLOPS")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=8, framealpha=0.4)

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        "FlagGems QC-MoE Kernel Benchmark  ·  "
        "FP16 reference = PyTorch SwiGLU MoE  ·  GPU = NVIDIA H20",
        ha="center", fontsize=8, color="#44445a",
    )

    # ── Category legend ─────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    fig.legend(
        handles=patches, loc="lower center", ncol=4,
        fontsize=8.5, framealpha=0.35,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.97])
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(CSV_W8A16):
        print(f"[ERROR] {CSV_W8A16} not found.")
        sys.exit(1)
    if not os.path.exists(CSV_W4A16):
        print(f"[ERROR] {CSV_W4A16} not found.")
        sys.exit(1)

    print(f"[DATA] Loading: {CSV_W8A16}")
    w8a16_data = load_data(CSV_W8A16, "W8A16")
    print(f"[DATA] {len(w8a16_data)} rows — W8A16")

    print(f"[DATA] Loading: {CSV_W4A16}")
    w4a16_data = load_data(CSV_W4A16, "W4A16")
    print(f"[DATA] {len(w4a16_data)} rows — W4A16")

    fig = plot(w8a16_data, w4a16_data)

    out_png = os.path.join(SCRIPT_DIR, "qcmoe_w8a16_benchmark.png")
    out_pdf = os.path.join(SCRIPT_DIR, "qcmoe_w8a16_benchmark.pdf")

    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="#0b0c10")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="#0b0c10")

    print(f"[OK] PNG → {out_png}")
    print(f"[OK] PDF → {out_pdf}")
