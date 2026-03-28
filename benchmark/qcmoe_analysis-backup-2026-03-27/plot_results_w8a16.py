#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC-MoE W8A16 Benchmark — Plotting Script
=========================================
Reads qcmoe_w8a16_data.csv and generates PNG + PDF charts.

Usage:
    python plot_results_w8a16.py

Output:
    qcmoe_w8a16_benchmark.png
    qcmoe_w8a16_benchmark.pdf
"""

import os
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
CSV_PATH   = os.path.join(SCRIPT_DIR, "qcmoe_w8a16_data.csv")


# ── Load CSV ─────────────────────────────────────────────────────────────────
def load_data(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
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


# ── Colour helpers ────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Seq len sweep":    "#5b8df0",
    "Hidden dim sweep": "#f09b5b",
    "Inter dim sweep":  "#62d17a",
    "Large shapes":     "#c97df0",
}


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot(data):
    labels  = [d["shape"]         for d in data]
    xlabels = [f"S={d['seq_len']}" for d in data]
    cats    = [d["category"]      for d in data]
    qc_ms       = [d["qc_ms"]       for d in data]
    fp16_ms     = [d["fp16_ms"]     for d in data]
    qc_tflops   = [d["qc_tflops"]   for d in data]
    fp16_tflops = [d["fp16_tflops"] for d in data]
    speedup     = [d["speedup"]     for d in data]

    x = np.arange(len(xlabels))
    w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "QC-MoE W8A16 vs vLLM-style FP16 MoE",
        fontsize=16, fontweight="bold", color="#e8e8f0", y=0.98,
    )

    # ── 1. Latency ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(x - w/2, qc_ms,   width=w, label="QC W8A16",  color="#5b8df0", edgecolor="none")
    ax.bar(x + w/2, fp16_ms, width=w, label="FP16 Ref",  color="#f09b5b", edgecolor="none")
    ax.set_title("Latency (ms)",  fontsize=12, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ms")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.4)

    # ── 2. Speedup ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    sp_colors = ["#62d17a" if s >= 1.0 else "#e05555" for s in speedup]
    ax.bar(x, speedup, color=sp_colors, edgecolor="none", width=0.6)
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("Speedup (QC / FP16)", fontsize=12, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("×")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_ylim(0, max(speedup) * 1.15)
    for i, v in enumerate(speedup):
        ax.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=7, color="#aaaacc")

    # ── 3. TFLOPS ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x - w/2, qc_tflops,   width=w, label="QC W8A16",  color="#5b8df0", edgecolor="none")
    ax.bar(x + w/2, fp16_tflops, width=w, label="FP16 Ref",  color="#f09b5b", edgecolor="none")
    ax.set_title("TFLOPS", fontsize=12, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TFLOPS")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.4)

    # ── 4. Shape Legend Table ──────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Shape Key", fontsize=12, fontweight="bold", color="#7a7a9a")

    col_labels = ["#", "Shape (S,H,K)", "Category"]
    table_data = [[str(d["idx"]), d["shape"], d["category"]] for d in data]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colWidths=[0.06, 0.38, 0.44],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e2030")
            cell.set_text_props(color="#8888aa", fontweight="bold")
        elif r > 0:
            cat = cats[r - 1]
            col = CAT_COLORS[cat]
            if c == 2:
                cell.set_facecolor(col + "22")
                cell.set_text_props(color=col)
            else:
                cell.set_facecolor("#13141c")
                cell.set_text_props(color="#c8c8d0")

    patches = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax.legend(
        handles=patches, loc="lower center", ncol=2,
        fontsize=8, framealpha=0.4, bbox_to_anchor=(0.5, -0.01),
    )

    fig.text(
        0.5, 0.01,
        "FlagGems QC-MoE W8A16 Kernel Benchmark  ·  "
        "FP16 reference = PyTorch SwiGLU MoE  ·  GPU = NVIDIA H20",
        ha="center", fontsize=8, color="#44445a",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[DATA] Loading: {CSV_PATH}")
    data = load_data(CSV_PATH)
    print(f"[DATA] {len(data)} rows loaded.")

    fig = plot(data)

    out_png = os.path.join(SCRIPT_DIR, "qcmoe_w8a16_benchmark.png")
    out_pdf = os.path.join(SCRIPT_DIR, "qcmoe_w8a16_benchmark.pdf")

    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="#0b0c10")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="#0b0c10")

    print(f"[OK] PNG → {out_png}")
    print(f"[OK] PDF → {out_pdf}")
