#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC-GEM W4A16 Performance Analysis - Dark Theme
================================================

Usage:
    python plot_results_w4a16.py

Output:
    qcgem1_w4a16_benchmark.png
    qcgem1_w4a16_benchmark.pdf
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

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

K_COLORS = {
    1024:  "#5b8df0",
    3584:  "#62d17a",
    7168:  "#f09b5b",
    128:   "#c97df0",
}

PALETTE = {
    "Torch":  "#f09b5b",
    "Gems":   "#5b8df0",
}

data = [
    (32768, 3584, 1024, 2.543360, 2.541600, 1.001, 94.633, 0.119, 0.120),
    (16384, 3584, 1024, 1.358880, 1.348704, 1.008, 89.166, 0.113, 0.113),
    (8192, 3584, 1024, 0.762928, 0.756416, 1.009, 79.493, 0.101, 0.102),
    (4096, 3584, 1024, 0.450848, 0.443664, 1.016, 67.765, 0.088, 0.089),
    (2048, 3584, 1024, 0.282528, 0.372512, 0.758, 40.354, 0.074, 0.056),
    (1024, 3584, 1024, 0.207328, 0.363968, 0.570, 20.651, 0.055, 0.031),
    (512, 3584, 1024, 0.203360, 0.326848, 0.622, 11.498, 0.033, 0.020),
    (256, 3584, 1024, 0.204224, 0.303104, 0.674, 6.199, 0.021, 0.014),
    (32768, 1024, 3584, 2.617888, 2.618096, 1.000, 91.868, 0.116, 0.116),
    (16384, 1024, 3584, 1.366160, 1.352608, 1.010, 88.909, 0.112, 0.113),
    (8192, 1024, 3584, 0.742912, 0.734800, 1.011, 81.831, 0.104, 0.105),
    (4096, 1024, 3584, 0.430704, 0.423664, 1.017, 70.964, 0.092, 0.094),
    (2048, 1024, 3584, 0.274272, 0.362304, 0.757, 41.491, 0.076, 0.057),
    (32768, 1024, 7168, 5.192416, 5.198560, 0.999, 92.533, 0.104, 0.104),
    (16384, 1024, 7168, 2.674864, 2.674784, 1.000, 89.921, 0.102, 0.102),
    (8192, 1024, 7168, 1.443232, 1.437728, 1.004, 83.645, 0.096, 0.096),
    (4096, 1024, 7168, 0.819456, 0.813696, 1.007, 73.897, 0.087, 0.087),
    (2048, 1024, 7168, 0.506368, 0.501792, 1.009, 59.915, 0.074, 0.074),
    (32768, 1024, 128, 0.201216, 0.344624, 0.584, 24.926, 0.376, 0.219),
    (16384, 1024, 128, 0.201840, 0.301728, 0.669, 14.235, 0.187, 0.125),
    (4096, 1024, 128, 0.200960, 0.273616, 0.734, 3.924, 0.047, 0.035),
]

M = np.array([d[0] for d in data])
N = np.array([d[1] for d in data])
K = np.array([d[2] for d in data])
torch_ms = np.array([d[3] for d in data])
gems_ms = np.array([d[4] for d in data])
speedup = np.array([d[5] for d in data])
tflops = np.array([d[6] for d in data])
torch_gbps = np.array([d[7] for d in data])
gems_gbps = np.array([d[8] for d in data])

n = len(data)
x = np.arange(n)
w_single = 0.5
w_half = 0.22


def shade_k_categories(ax, x, w):
    for i, k_val in enumerate(K):
        color = K_COLORS.get(k_val, "#888888")
        ax.axvspan(x[i] - w/2, x[i] + w/2, alpha=0.12, color=color, zorder=0)


def plot_benchmark():
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(
        "QC-GEM W4A16 vs PyTorch FP16 GEMM — Comprehensive Shapes (H20 GPU)",
        fontsize=15, fontweight="bold", color="#e8e8f0", y=0.99,
    )

    # (0,0) Speedup
    ax = axes[0, 0]
    shade_k_categories(ax, x, w_single)
    sp_cols = ["#62d17a" if s >= 1.0 else "#e05555" for s in speedup]
    ax.bar(x, speedup, color=sp_cols, edgecolor="none", width=w_single)
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("Speedup vs FP16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"M={m//1024}k" for m in M], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("×")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_ylim(0, max(speedup) * 1.18)
    for i, v in enumerate(speedup):
        ax.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=7.5, color="#aaaacc")

    # (0,1) Latency
    ax = axes[0, 1]
    shade_k_categories(ax, x, w_half * 2 + 0.02)
    ax.bar(x - w_half/2, torch_ms, width=w_half, label="Torch FP16", color=PALETTE["Torch"], edgecolor="none")
    ax.bar(x + w_half/2, gems_ms,   width=w_half, label="QC W4A16",   color=PALETTE["Gems"],   edgecolor="none")
    ax.set_title("Latency — FP16 ref / W4A16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"M={m//1024}k" for m in M], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("ms")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.4)

    # (1,0) TFLOPS
    ax = axes[1, 0]
    shade_k_categories(ax, x, w_half * 2 + 0.02)
    ax.bar(x - w_half/2, tflops, width=w_half, label="Gems TFLOPS", color=PALETTE["Gems"], edgecolor="none")
    ax.set_title("TFLOPS — QC W4A16", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"M={m//1024}k" for m in M], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("TFLOPS")
    ax.grid(axis="y", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.4)

    # (1,1) Speedup by K-dimension
    ax = axes[1, 1]
    unique_k = sorted(set(K))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for idx, k_val in enumerate(unique_k):
        mask = K == k_val
        m_vals = M[mask]
        sp_vals = speedup[mask]
        sorted_pairs = sorted(zip(m_vals, sp_vals))
        ms, sps = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.plot(
            [m/1024 for m in ms], sps,
            "o-", linewidth=1.8, markersize=6,
            color=K_COLORS.get(k_val, cmap(idx)),
            label=f"K={k_val}",
        )
    ax.axhline(1.0, color="#55556a", linewidth=1, linestyle="--")
    ax.set_title("Speedup by K-dimension", fontsize=11, fontweight="bold", color="#7a7a9a")
    ax.set_xlabel("M (tokens) in k")
    ax.set_ylabel("Speedup ×")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.4)

    # K category legend
    patches = [mpatches.Patch(color=c, label=f"K={k}") for k, c in K_COLORS.items() if k in K]
    fig.legend(
        handles=patches, loc="lower center", ncol=4,
        fontsize=9, framealpha=0.35,
        bbox_to_anchor=(0.5, -0.005),
    )

    # Footer
    fig.text(
        0.5, 0.003,
        "FlagGems QC-GEM W4A16 Kernel  ·  FP16 reference = PyTorch GEMM  ·  GPU = NVIDIA H20",
        ha="center", fontsize=8, color="#44445a",
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.97])
    return fig


if __name__ == "__main__":
    fig = plot_benchmark()

    out_png = SCRIPT_DIR / "qcgem1_w4a16_benchmark.png"
    out_pdf = SCRIPT_DIR / "qcgem1_w4a16_benchmark.pdf"

    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="#0b0c10")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="#0b0c10")

    print(f"[OK] PNG → {out_png}")
    print(f"[OK] PDF → {out_pdf}")

    # Summary
    summary = f"""
================================================================================
QC GEM W4A16 Performance Summary (dtype=torch.float16)
================================================================================

Total Configurations Tested: {len(data)}

Speedup Statistics:
  Mean Speedup:   {np.mean(speedup):.4f}x
  Min Speedup:    {np.min(speedup):.4f}x (M={M[np.argmin(speedup)]}, N={N[np.argmin(speedup)]}, K={K[np.argmin(speedup)]})
  Max Speedup:    {np.max(speedup):.4f}x (M={M[np.argmax(speedup)]}, N={N[np.argmax(speedup)]}, K={K[np.argmax(speedup)]})
  Configurations with speedup >= 1.0: {np.sum(speedup >= 1.0)} / {len(speedup)}

TFLOPS Statistics:
  Mean TFLOPS:    {np.mean(tflops):.2f}
  Max TFLOPS:     {np.max(tflops):.2f} (M={M[np.argmax(tflops)]}, N={N[np.argmax(tflops)]}, K={K[np.argmax(tflops)]})
================================================================================
"""
    print(summary)

    with open(SCRIPT_DIR / 'qcgem1_w4a16_summary.txt', 'w') as f:
        f.write(summary)
    print(f"[OK] Summary → {SCRIPT_DIR / 'qcgem1_w4a16_summary.txt'}")
