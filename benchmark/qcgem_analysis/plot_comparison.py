#!/usr/bin/env python3
"""
QC-GEM w4A16 vs w8A16 Benchmark Comparison Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

output_dir = '/data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems/benchmark/qcgemanalysis'

df_w4a16 = pd.read_csv(os.path.join(output_dir, 'benchmark_w4a16_fp16.csv'))
df_w8a16 = pd.read_csv(os.path.join(output_dir, 'benchmark_w8a16_fp16.csv'))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('QC-GEM w4A16 vs w8A16 Benchmark Comparison (FP16)\nQwen3.5-397B MoE Shapes', fontsize=14, fontweight='bold')

# Align by index
x = np.arange(len(df_w4a16))
m_vals = df_w4a16['m'].values

# Plot 1: Speedup Comparison
ax1 = axes[0, 0]
width = 0.35
bars1 = ax1.bar(x - width/2, df_w4a16['speedup'], width, label='w4A16', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, df_w8a16['speedup'], width, label='w8A16', color='coral', alpha=0.8)
ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('Shape Index')
ax1.set_ylabel('Speedup (x)')
ax1.set_title('Speedup: w4A16 vs w8A16')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: TFLOPS Comparison
ax2 = axes[0, 1]
ax2.bar(x - width/2, df_w4a16['tflops'], width, label='w4A16', color='steelblue', alpha=0.8)
ax2.bar(x + width/2, df_w8a16['tflops'], width, label='w8A16', color='coral', alpha=0.8)
ax2.set_xlabel('Shape Index')
ax2.set_ylabel('TFLOPS')
ax2.set_title('Throughput: w4A16 vs w8A16')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Speedup by N dimension (w4A16)
ax3 = axes[1, 0]
n_unique = sorted(df_w4a16['n'].unique())
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(n_unique)))
for idx, n_val in enumerate(n_unique):
    mask = df_w4a16['n'] == n_val
    m_filtered = df_w4a16.loc[mask, 'm'].values // 1024
    s_filtered = df_w4a16.loc[mask, 'speedup'].values
    ax3.plot(m_filtered, s_filtered, 'o-', linewidth=2, markersize=8, 
             color=colors[idx], label=f'N={n_val}')
ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('M (Batch*Seq) in thousands')
ax3.set_ylabel('Speedup (x)')
ax3.set_title('w4A16 Speedup by Problem Size')
ax3.set_xscale('log', base=2)
ax3.legend(loc='lower left')
ax3.grid(alpha=0.3)

# Plot 4: Speedup by N dimension (w8A16)
ax4 = axes[1, 1]
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(n_unique)))
for idx, n_val in enumerate(n_unique):
    mask = df_w8a16['n'] == n_val
    m_filtered = df_w8a16.loc[mask, 'm'].values // 1024
    s_filtered = df_w8a16.loc[mask, 'speedup'].values
    ax4.plot(m_filtered, s_filtered, 'o-', linewidth=2, markersize=8, 
             color=colors[idx], label=f'N={n_val}')
ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
ax4.set_xlabel('M (Batch*Seq) in thousands')
ax4.set_ylabel('Speedup (x)')
ax4.set_title('w8A16 Speedup by Problem Size')
ax4.set_xscale('log', base=2)
ax4.legend(loc='lower left')
ax4.grid(alpha=0.3)

plt.tight_layout()

png_path = os.path.join(output_dir, 'benchmark_comparison.png')
pdf_path = os.path.join(output_dir, 'benchmark_comparison.pdf')
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"Comparison charts saved: {png_path}, {pdf_path}")

# Summary statistics
print("\n" + "="*80)
print("BENCHMARK COMPARISON SUMMARY: w4A16 vs w8A16")
print("="*80)
print(f"\n{'Metric':<30} {'w4A16':<15} {'w8A16':<15} {'Difference':<15}")
print("-"*80)
print(f"{'Average Speedup':<30} {df_w4a16['speedup'].mean():<15.3f} {df_w8a16['speedup'].mean():<15.3f} {df_w4a16['speedup'].mean() - df_w8a16['speedup'].mean():<+15.3f}")
print(f"{'Max Speedup':<30} {df_w4a16['speedup'].max():<15.3f} {df_w8a16['speedup'].max():<15.3f} {df_w4a16['speedup'].max() - df_w8a16['speedup'].max():<+15.3f}")
print(f"{'Min Speedup':<30} {df_w4a16['speedup'].min():<15.3f} {df_w8a16['speedup'].min():<15.3f} {df_w4a16['speedup'].min() - df_w8a16['speedup'].min():<+15.3f}")
print(f"{'Average TFLOPS':<30} {df_w4a16['tflops'].mean():<15.2f} {df_w8a16['tflops'].mean():<15.2f} {df_w4a16['tflops'].mean() - df_w8a16['tflops'].mean():<+15.2f}")
print(f"{'Max TFLOPS':<30} {df_w4a16['tflops'].max():<15.2f} {df_w8a16['tflops'].max():<15.2f} {df_w4a16['tflops'].max() - df_w8a16['tflops'].max():<+15.2f}")

# Large shape analysis (M >= 4096)
large_mask = df_w4a16['m'] >= 4096
print(f"\nLarge Shapes (M >= 4096):")
print(f"{'Average Speedup':<30} {df_w4a16.loc[large_mask, 'speedup'].mean():<15.3f} {df_w8a16.loc[large_mask, 'speedup'].mean():<15.3f}")
print(f"{'Average TFLOPS':<30} {df_w4a16.loc[large_mask, 'tflops'].mean():<15.2f} {df_w8a16.loc[large_mask, 'tflops'].mean():<15.2f}")

# Small shape analysis (M < 4096)
small_mask = df_w4a16['m'] < 4096
print(f"\nSmall Shapes (M < 4096):")
print(f"{'Average Speedup':<30} {df_w4a16.loc[small_mask, 'speedup'].mean():<15.3f} {df_w8a16.loc[small_mask, 'speedup'].mean():<15.3f}")
print(f"{'Average TFLOPS':<30} {df_w4a16.loc[small_mask, 'tflops'].mean():<15.2f} {df_w8a16.loc[small_mask, 'tflops'].mean():<15.2f}")
