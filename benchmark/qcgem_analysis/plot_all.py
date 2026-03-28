#!/usr/bin/env python3
"""
QC-GEM Benchmark Analysis - Complete Plotting Script
Generates all visualization charts for w4A16 and w8A16 benchmarks
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

output_dir = '/data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems/benchmark/qcgemanalysis'

def load_data():
    """Load benchmark data from CSV files"""
    df_w4a16 = pd.read_csv(os.path.join(output_dir, 'benchmark_w4a16_fp16.csv'))
    df_w8a16 = pd.read_csv(os.path.join(output_dir, 'benchmark_w8a16_fp16.csv'))
    return df_w4a16, df_w8a16

def plot_w4a16_single(df, output_dir):
    """Generate w4A16 standalone charts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QC-GEM w4A16 Benchmark Results (FP16)\nQwen3.5-397B MoE Shapes', fontsize=14, fontweight='bold')
    
    x = np.arange(len(df))
    m_vals = df['m'].values
    torch_latency = df['torch_latency_ms'].values
    gems_latency = df['gems_latency_ms'].values
    speedup = df['speedup'].values
    tflops = df['tflops'].values
    
    # Plot 1: Latency Comparison
    ax1 = axes[0, 0]
    width = 0.35
    ax1.bar(x - width/2, torch_latency, width, label='Torch', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, gems_latency, width, label='QC-GEM', color='coral', alpha=0.8)
    ax1.set_xlabel('Shape Index')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Comparison: Torch vs QC-GEM')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
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
    ax2.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
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
    ax3.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Speedup by N dimension
    ax4 = axes[1, 1]
    n_unique = sorted(df['n'].unique())
    for n_val in n_unique:
        mask = df['n'] == n_val
        m_filtered = df.loc[mask, 'm'].values // 1024
        s_filtered = df.loc[mask, 'speedup'].values
        ax4.plot(m_filtered, s_filtered, 'o-', linewidth=2, markersize=8, label=f'N={n_val}')
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
    ax4.set_xlabel('M (Batch*Seq) in thousands')
    ax4.set_ylabel('Speedup (x)')
    ax4.set_title('Speedup by Problem Size (N dimension)')
    ax4.set_xscale('log', base=2)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_w4a16_fp16.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'benchmark_w4a16_fp16.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: benchmark_w4a16_fp16.png/pdf")

def plot_w8a16_single(df, output_dir):
    """Generate w8A16 standalone charts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QC-GEM w8A16 Benchmark Results (FP16)\nQwen3.5-397B MoE Shapes', fontsize=14, fontweight='bold')
    
    x = np.arange(len(df))
    m_vals = df['m'].values
    torch_latency = df['torch_latency_ms'].values
    gems_latency = df['gems_latency_ms'].values
    speedup = df['speedup'].values
    tflops = df['tflops'].values
    
    # Plot 1: Latency Comparison
    ax1 = axes[0, 0]
    width = 0.35
    ax1.bar(x - width/2, torch_latency, width, label='Torch', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, gems_latency, width, label='QC-GEM', color='coral', alpha=0.8)
    ax1.set_xlabel('Shape Index')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Comparison: Torch vs QC-GEM')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
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
    ax2.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
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
    ax3.set_xticklabels([f'{m//1024}k' for m in m_vals], rotation=45, ha='right')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Speedup by N dimension
    ax4 = axes[1, 1]
    n_unique = sorted(df['n'].unique())
    for n_val in n_unique:
        mask = df['n'] == n_val
        m_filtered = df.loc[mask, 'm'].values // 1024
        s_filtered = df.loc[mask, 'speedup'].values
        ax4.plot(m_filtered, s_filtered, 'o-', linewidth=2, markersize=8, label=f'N={n_val}')
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
    ax4.set_xlabel('M (Batch*Seq) in thousands')
    ax4.set_ylabel('Speedup (x)')
    ax4.set_title('Speedup by Problem Size (N dimension)')
    ax4.set_xscale('log', base=2)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_w8a16_fp16.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'benchmark_w8a16_fp16.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: benchmark_w8a16_fp16.png/pdf")

def plot_comparison(df_w4a16, df_w8a16, output_dir):
    """Generate w4A16 vs w8A16 comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QC-GEM w4A16 vs w8A16 Benchmark Comparison (FP16)\nQwen3.5-397B MoE Shapes', fontsize=14, fontweight='bold')
    
    x = np.arange(len(df_w4a16))
    m_vals = df_w4a16['m'].values
    
    width = 0.35
    
    # Plot 1: Speedup Comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, df_w4a16['speedup'], width, label='w4A16', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, df_w8a16['speedup'], width, label='w8A16', color='coral', alpha=0.8)
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
        ax3.plot(m_filtered, s_filtered, 'o-', linewidth=2, markersize=8, color=colors[idx], label=f'N={n_val}')
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
        ax4.plot(m_filtered, s_filtered, 'o-', linewidth=2, markersize=8, color=colors[idx], label=f'N={n_val}')
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('M (Batch*Seq) in thousands')
    ax4.set_ylabel('Speedup (x)')
    ax4.set_title('w8A16 Speedup by Problem Size')
    ax4.set_xscale('log', base=2)
    ax4.legend(loc='lower left')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: benchmark_comparison.png/pdf")

def print_summary(df_w4a16, df_w8a16):
    """Print summary statistics"""
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
    
    large_mask = df_w4a16['m'] >= 4096
    print(f"\nLarge Shapes (M >= 4096):")
    print(f"{'Average Speedup':<30} {df_w4a16.loc[large_mask, 'speedup'].mean():<15.3f} {df_w8a16.loc[large_mask, 'speedup'].mean():<15.3f}")
    print(f"{'Average TFLOPS':<30} {df_w4a16.loc[large_mask, 'tflops'].mean():<15.2f} {df_w8a16.loc[large_mask, 'tflops'].mean():<15.2f}")
    
    small_mask = df_w4a16['m'] < 4096
    print(f"\nSmall Shapes (M < 4096):")
    print(f"{'Average Speedup':<30} {df_w4a16.loc[small_mask, 'speedup'].mean():<15.3f} {df_w8a16.loc[small_mask, 'speedup'].mean():<15.3f}")
    print(f"{'Average TFLOPS':<30} {df_w4a16.loc[small_mask, 'tflops'].mean():<15.2f} {df_w8a16.loc[small_mask, 'tflops'].mean():<15.2f}")

if __name__ == '__main__':
    df_w4a16, df_w8a16 = load_data()
    plot_w4a16_single(df_w4a16, output_dir)
    plot_w8a16_single(df_w8a16, output_dir)
    plot_comparison(df_w4a16, df_w8a16, output_dir)
    print_summary(df_w4a16, df_w8a16)
    print("\nAll charts generated successfully!")
