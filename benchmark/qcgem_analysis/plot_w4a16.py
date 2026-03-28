#!/usr/bin/env python3
"""
QC-GEM w4A16 Benchmark Results Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data extracted from w4A16 benchmark output (FP16)
data_w4a16 = [
    (32768, 1024, 3584, 2.645280, 2.562960, 1.032, 93.844),
    (16384, 1024, 3584, 1.396656, 1.341408, 1.041, 89.651),
    (8192, 1024, 3584, 0.757536, 0.750592, 1.009, 80.109),
    (4096, 1024, 3584, 0.449088, 0.441664, 1.017, 68.072),
    (2048, 1024, 3584, 0.282464, 0.337968, 0.836, 44.479),
    (1024, 1024, 3584, 0.208288, 0.325152, 0.641, 23.116),
    (512, 1024, 3584, 0.185600, 0.288464, 0.643, 13.028),
    (256, 1024, 3584, 0.185632, 0.281024, 0.661, 6.686),
    (32768, 3584, 1024, 2.709680, 2.679200, 1.011, 89.772),
    (16384, 3584, 1024, 1.351952, 1.346432, 1.004, 89.317),
    (8192, 3584, 1024, 0.738368, 0.730752, 1.010, 82.284),
    (4096, 3584, 1024, 0.427952, 0.421792, 1.015, 71.279),
    (2048, 3584, 1024, 0.273824, 0.335520, 0.816, 44.803),
    (32768, 7168, 1024, 5.365376, 5.288864, 1.014, 90.953),
    (16384, 7168, 1024, 2.662064, 2.653136, 1.003, 90.654),
    (8192, 7168, 1024, 1.431824, 1.426048, 1.004, 84.330),
    (4096, 7168, 1024, 0.815616, 0.808896, 1.008, 74.335),
    (2048, 7168, 1024, 0.504864, 0.499776, 1.010, 60.156),
    (32768, 128, 1024, 0.186464, 0.309296, 0.603, 27.773),
    (16384, 128, 1024, 0.186656, 0.274208, 0.681, 15.663),
    (4096, 128, 1024, 0.182384, 0.250480, 0.728, 4.287),
]

M = np.array([d[0] for d in data_w4a16])
N = np.array([d[1] for d in data_w4a16])
K = np.array([d[2] for d in data_w4a16])
torch_latency = np.array([d[3] for d in data_w4a16])
gems_latency = np.array([d[4] for d in data_w4a16])
speedup = np.array([d[5] for d in data_w4a16])
tflops = np.array([d[6] for d in data_w4a16])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('QC-GEM w4A16 Benchmark Results (FP16)\nQwen3.5-397B MoE Shapes', fontsize=14, fontweight='bold')

# Plot 1: Latency Comparison
ax1 = axes[0, 0]
x = np.arange(len(data_w4a16))
width = 0.35
ax1.bar(x - width/2, torch_latency, width, label='Torch', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, gems_latency, width, label='QC-GEM', color='coral', alpha=0.8)
ax1.set_xlabel('Shape Index')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Latency Comparison: Torch vs QC-GEM')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{m//1024}k' for m in M], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Speedup
ax2 = axes[0, 1]
colors = ['green' if s >= 1.0 else 'red' for s in speedup]
bars = ax2.bar(x, speedup, color=colors, alpha=0.8)
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (1.0x)')
ax2.set_xlabel('Shape Index')
ax2.set_ylabel('Speedup (x)')
ax2.set_title('QC-GEM Speedup vs Torch')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{m//1024}k' for m in M], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, s in zip(bars, speedup):
    height = bar.get_height()
    ax2.annotate(f'{s:.3f}x', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Plot 3: TFLOPS
ax3 = axes[1, 0]
ax3.plot(x, tflops, 'o-', color='purple', linewidth=2, markersize=6)
ax3.fill_between(x, tflops, alpha=0.3, color='purple')
ax3.set_xlabel('Shape Index')
ax3.set_ylabel('TFLOPS')
ax3.set_title('QC-GEM Throughput (TFLOPS)')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{m//1024}k' for m in M], rotation=45, ha='right')
ax3.grid(alpha=0.3)
for i, t in enumerate(tflops):
    ax3.annotate(f'{t:.1f}', xy=(i, t), xytext=(0, 5),
                textcoords="offset points", ha='center', va='bottom', fontsize=7)

# Plot 4: Speedup by N dimension
ax4 = axes[1, 1]
unique_n = sorted(set(N))
for n_val in unique_n:
    mask = N == n_val
    m_vals = M[mask] // 1024
    s_vals = speedup[mask]
    ax4.plot(m_vals, s_vals, 'o-', linewidth=2, markersize=8, label=f'N={n_val}')
ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
ax4.set_xlabel('M (Batch*Seq) in thousands')
ax4.set_ylabel('Speedup (x)')
ax4.set_title('Speedup by Problem Size (N dimension)')
ax4.set_xscale('log', base=2)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()

output_dir = '/data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems/benchmark/qcgemanalysis'
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, 'benchmark_w4a16_fp16.png')
pdf_path = os.path.join(output_dir, 'benchmark_w4a16_fp16.pdf')

plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print(f"Charts saved to:")
print(f"  PNG: {png_path}")
print(f"  PDF: {pdf_path}")

print("\n" + "="*80)
print("BENCHMARK SUMMARY: QC-GEM w4A16 (FP16)")
print("="*80)
print(f"{'Shape':<25} {'Torch(ms)':<12} {'QC-GEM(ms)':<12} {'Speedup':<10} {'TFLOPS':<10}")
print("-"*80)
for d in data_w4a16:
    m, n, k, torch_ms, gems_ms, sp, tflp = d
    shape_str = f"M={m}, N={n}, K={k}"
    print(f"{shape_str:<25} {torch_ms:<12.3f} {gems_ms:<12.3f} {sp:<10.3f} {tflp:<10.2f}")

print("-"*80)
print(f"Average Speedup: {np.mean(speedup):.3f}x")
print(f"Max Speedup: {np.max(speedup):.3f}x")
print(f"Min Speedup: {np.min(speedup):.3f}x")
print(f"Average TFLOPS: {np.mean(tflops):.2f}")
print(f"Max TFLOPS: {np.max(tflops):.2f}")
