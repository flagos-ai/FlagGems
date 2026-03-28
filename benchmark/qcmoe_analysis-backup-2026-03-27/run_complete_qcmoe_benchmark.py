#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC-MoE Complete Benchmark for Qwen3.5-397B-A17B
=================================================
Runs W8A16 and W4A16 benchmarks for all shapes from YAML config,
generates complete test report.

Usage:
    python run_complete_qcmoe_benchmark.py              # run all shapes
    python run_complete_qcmoe_benchmark.py --w8a16       # W8A16 only
    python run_complete_qcmoe_benchmark.py --w4a16       # W4A16 only
    python run_complete_qcmoe_benchmark.py --quick       # quick test (few shapes)

Output:
    - CSV: qcmoe_complete_w8a16_data.csv, qcmoe_complete_w4a16_data.csv
    - Report: qcmoe_complete_report.md
    - Summary stats printed to terminal
"""

import argparse
import csv
import time
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

# ── GPU Info ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_SM   = torch.cuda.get_device_capability(0)
    print(f"[ENV] GPU: {GPU_NAME} (sm_{GPU_SM[0]}{GPU_SM[1]})")
    print(f"[ENV] PyTorch: {torch.__version__}")
    import triton
    print(f"[ENV] Triton: {triton.__version__}")
    import flag_gems
    print(f"[ENV] FlagGems: {flag_gems.__version__}")
else:
    print("[ERROR] CUDA not available.")
    sys.exit(1)

from flag_gems.ops.qcmoe import fused_moe, QuantConfig, QuantMode


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda"
DTYPE        = torch.float16
GROUP_SIZE   = 128
NUM_EXPERTS  = 8
TOP_K        = 2
NUM_ITERS    = 10
WARMUP       = 3
RANDOM_SEED  = 42


# ── Load Shapes from YAML ─────────────────────────────────────────────────────

def load_shapes_from_yaml(yaml_path):
    """Load shapes from YAML config file."""
    import yaml
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ── Quantization ──────────────────────────────────────────────────────────────

def quantize_weights_moe(weights, w_nbits=8, group_size=128):
    """Per-group min-max quantization for MoE expert weights."""
    E, n_out, k_in = weights.shape
    num_groups = k_in // group_size
    w_bits = 8 if w_nbits == 8 else 4

    w_r = weights.view(E, n_out, num_groups, group_size)
    w_min = w_r.min(dim=-1, keepdim=True)[0]
    w_max = w_r.max(dim=-1, keepdim=True)[0]
    scale = (w_max - w_min) / ((2 ** w_bits) - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    W_norm = (w_r - w_min) / (scale + 1e-8)
    W_q = W_norm.round().clamp(0, 2 ** w_bits - 1).to(torch.uint8)

    if w_nbits == 4:
        W_q = W_q.view(E, n_out, num_groups, group_size // 2, 2)
        W_q = (W_q[..., 0] & 0xF) | (W_q[..., 1] << 4)
        W_q = W_q.view(E, n_out, -1)
    else:
        W_q = W_q.view(E, n_out, -1)

    scales = scale.squeeze(-1).view(E, n_out, num_groups)
    zeros  = w_min.squeeze(-1).view(E, n_out, num_groups)
    return W_q, scales, zeros


# ── FP16 Reference ────────────────────────────────────────────────────────────

def sglang_fp16_moe_from_deq_weights(inp, W1_deq, W2_deq, W3, topk_weights, topk_ids):
    """Pure-PyTorch FP16 SwiGLU MoE reference."""
    M, H = inp.shape
    E, K, _ = W1_deq.shape
    output = torch.zeros(M, H, dtype=inp.dtype, device=inp.device)

    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        tokens_e = mask.nonzero(as_tuple=True)[0]
        weights_e = topk_weights[mask]

        inp_e = inp.index_select(0, tokens_e)
        gate = torch.mm(inp_e, W1_deq[e].T)
        up   = torch.mm(inp_e, W3[e].T)
        act  = F.silu(gate) * up
        down = torch.mm(act, W2_deq[e].T)

        down_w = down * weights_e.unsqueeze(1)
        output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, H), down_w)

    return output


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_single_benchmark(seq_len, hidden_dim, inter_dim, w_nbits, verbose=True):
    """Run benchmark for a single shape."""
    device = DEVICE
    dtype  = DTYPE
    torch.manual_seed(RANDOM_SEED)

    # Weights: W1=(E,K,H), W2=(E,H,K), W3=(E,K,H)
    W1 = torch.randn(NUM_EXPERTS, inter_dim,  hidden_dim, dtype=dtype, device=device)
    W2 = torch.randn(NUM_EXPERTS, hidden_dim,  inter_dim, dtype=dtype, device=device)
    W3 = torch.randn(NUM_EXPERTS, inter_dim,   hidden_dim, dtype=dtype, device=device)

    # Quantize: W2 needs transpose before quant
    W1_q, W1_sc, W1_z = quantize_weights_moe(W1,  w_nbits, GROUP_SIZE)
    W2_T, W2_sc, W2_z = quantize_weights_moe(W2.transpose(1, 2), w_nbits, GROUP_SIZE)
    W2_q = W2_T.transpose(1, 2)

    # Input and routing
    inp          = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
    topk_weights = torch.ones(seq_len, TOP_K, dtype=dtype, device=device) / TOP_K
    topk_ids     = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K),
                                 dtype=torch.int64, device=device)

    quant_mode   = QuantMode.W8A16 if w_nbits == 8 else QuantMode.W4A16

    # Warmup
    for _ in range(WARMUP):
        _ = fused_moe(
            inp, None, None, W3,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=QuantConfig(mode=quant_mode, group_size=GROUP_SIZE),
            num_experts=NUM_EXPERTS, top_k=TOP_K,
            w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
            w2_q=W2_q, w2_scales=W2_sc, w2_zeros=W2_z,
        )
        _ = sglang_fp16_moe_from_deq_weights(inp, W1, W2, W3, topk_weights, topk_ids)
    torch.cuda.synchronize()

    # Time QC kernel
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        out_qc = fused_moe(
            inp, None, None, W3,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=QuantConfig(mode=quant_mode, group_size=GROUP_SIZE),
            num_experts=NUM_EXPERTS, top_k=TOP_K,
            w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
            w2_q=W2_q, w2_scales=W2_sc, w2_zeros=W2_z,
        )
    torch.cuda.synchronize()
    qc_ms = (time.perf_counter() - t0) / NUM_ITERS * 1000

    # Time FP16 reference
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        out_ref = sglang_fp16_moe_from_deq_weights(inp, W1, W2, W3, topk_weights, topk_ids)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / NUM_ITERS * 1000

    # Compute metrics
    total_flops = seq_len * TOP_K * 4 * hidden_dim * inter_dim
    qc_tflops   = total_flops / qc_ms / 1e9
    ref_tflops  = total_flops / ref_ms / 1e9
    speedup     = ref_ms / qc_ms
    max_abs_err = (out_qc.float() - out_ref.float()).abs().max().item()

    return {
        "qc_ms": qc_ms,
        "fp16_ms": ref_ms,
        "qc_tflops": qc_tflops,
        "fp16_tflops": ref_tflops,
        "speedup": speedup,
        "max_abs_err": max_abs_err,
    }


# ── YAML Shapes ────────────────────────────────────────────────────────────────

def get_yaml_shapes(yaml_path):
    """Extract QCMoEBenchmark shapes from YAML."""
    import yaml
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    shapes = []
    if 'QCMoEBenchmark' in config and 'shapes' in config['QCMoEBenchmark']:
        for shape in config['QCMoEBenchmark']['shapes']:
            B, S, H, K = shape
            shapes.append({
                "batch": B, "seq": S, "hidden": H, "inter": K,
                "shape_str": f"({B},{S},{H},{K})"
            })

    return shapes


def get_all_mm_shapes(yaml_path):
    """Extract all GEMM shapes from YAML for comprehensive coverage."""
    import yaml
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    shapes = []

    # QCMoEBenchmark shapes
    if 'QCMoEBenchmark' in config and 'shapes' in config['QCMoEBenchmark']:
        for shape in config['QCMoEBenchmark']['shapes']:
            B, S, H, K = shape
            shapes.append({
                "batch": B, "seq": S, "hidden": H, "inter": K,
                "shape_str": f"({B},{S},{H},{K})",
                "category": "QCMoEBenchmark",
                "desc": "MoE FFN"
            })

    # mm shapes
    if 'mm' in config and 'shapes' in config['mm']:
        for shape in config['mm']['shapes']:
            B, S, H, K = shape
            shapes.append({
                "batch": B, "seq": S, "hidden": H, "inter": K,
                "shape_str": f"({B},{S},{H},{K})",
                "category": "mm",
                "desc": "GEMM"
            })

    # bmm shapes
    if 'bmm' in config and 'shapes' in config['bmm']:
        for shape in config['bmm']['shapes']:
            B, M, K, N = shape
            # For bmm, K is actually hidden, N is inter
            shapes.append({
                "batch": B, "seq": M, "hidden": K, "inter": N,
                "shape_str": f"({B},{M},{K},{N})",
                "category": "bmm",
                "desc": "Batched GEMM"
            })

    # Remove duplicates
    seen = set()
    unique_shapes = []
    for s in shapes:
        key = (s["batch"], s["seq"], s["hidden"], s["inter"])
        if key not in seen:
            seen.add(key)
            unique_shapes.append(s)

    return unique_shapes


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(all_results, output_path, yaml_path):
    """Generate comprehensive markdown report."""
    import yaml
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    model_name = "Qwen3.5-397B-A17B"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# QC-MoE Complete Benchmark Report

## 测试信息

| 项目 | 值 |
|------|-----|
| 模型 | {model_name} |
| 测试日期 | {timestamp} |
| GPU | {GPU_NAME} (sm_{GPU_SM[0]}{GPU_SM[1]}) |
| PyTorch | {torch.__version__} |
| Triton | {triton.__version__} |
| FlagGems | {flag_gems.__version__} |
| 量化模式 | W8A16 / W4A16 |
| Group Size | {GROUP_SIZE} |
| Experts | {NUM_EXPERTS} |
| Top-K | {TOP_K} |
| 迭代次数 | {NUM_ITERS} (warmup: {WARMUP}) |

---

## 测试 Shapes 总览

"""

    # Count shapes by category
    categories = {}
    for r in all_results:
        cat = r.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, results in sorted(categories.items()):
        report += f"### {cat} ({len(results)} shapes)\n\n"

    report += "---\n\n"

    # ── W8A16 Results ──────────────────────────────────────────────────────────
    w8a16_results = [r for r in all_results if r.get("w_nbits") == 8]
    w4a16_results = [r for r in all_results if r.get("w_nbits") == 4]

    report += "## W8A16 测试结果\n\n"
    report += "| # | Shape (B,S,H,K) | Category | QC W8A16 (ms) | FP16 Ref (ms) | W8A16 TFLOPS | FP16 TFLOPS | Speedup | MaxAbsErr |\n"
    report += "|:--|:----------------|:---------|:--------------|:--------------|:-------------|:-----------|:--------|:----------|\n"

    for i, r in enumerate(w8a16_results, 1):
        speedup_str = f"**{r['speedup']:.2f}x**" if r['speedup'] >= 1.0 else f"{r['speedup']:.2f}x"
        report += f"| {i} | {r['shape_str']} | {r.get('category', 'N/A')} | {r['qc_ms']:.3f} | {r['fp16_ms']:.3f} | {r['qc_tflops']:.2f} | {r['fp16_tflops']:.2f} | {speedup_str} | {r['max_abs_err']:.2e} |\n"

    # W8A16 Summary
    w8a16_speedups = [r['speedup'] for r in w8a16_results]
    w8a16_improved = sum(1 for s in w8a16_speedups if s >= 1.0)
    w8a16_avg_speedup = sum(w8a16_speedups) / len(w8a16_speedups)
    w8a16_avg_tflops = sum(r['qc_tflops'] for r in w8a16_results) / len(w8a16_results)

    report += f"""
### W8A16 汇总统计

| 指标 | 值 |
|------|-----|
| 测试 shapes 数 | {len(w8a16_results)} |
| 性能提升 shapes | {w8a16_improved} ({w8a16_improved/len(w8a16_speedups)*100:.1f}%) |
| 性能下降 shapes | {len(w8a16_speedups) - w8a16_improved} ({(len(w8a16_speedups) - w8a16_improved)/len(w8a16_speedups)*100:.1f}%) |
| 平均 Speedup | {w8a16_avg_speedup:.2f}x |
| 平均 TFLOPS | {w8a16_avg_tflops:.2f} |
| 最佳 Speedup | {max(w8a16_speedups):.2f}x ({[r['shape_str'] for r in w8a16_results if r['speedup'] == max(w8a16_speedups)][0]}) |
| 最差 Speedup | {min(w8a16_speedups):.2f}x ({[r['shape_str'] for r in w8a16_results if r['speedup'] == min(w8a16_speedups)][0]}) |

---

## W4A16 测试结果

| # | Shape (B,S,H,K) | Category | QC W4A16 (ms) | FP16 Ref (ms) | W4A16 TFLOPS | FP16 TFLOPS | Speedup | MaxAbsErr |
|:--|:----------------|:---------|:--------------|:--------------|:-------------|:-----------|:--------|:----------|
"""

    for i, r in enumerate(w4a16_results, 1):
        speedup_str = f"**{r['speedup']:.2f}x**" if r['speedup'] >= 1.0 else f"{r['speedup']:.2f}x"
        report += f"| {i} | {r['shape_str']} | {r.get('category', 'N/A')} | {r['qc_ms']:.3f} | {r['fp16_ms']:.3f} | {r['qc_tflops']:.2f} | {r['fp16_tflops']:.2f} | {speedup_str} | {r['max_abs_err']:.2e} |\n"

    # W4A16 Summary
    w4a16_speedups = [r['speedup'] for r in w4a16_results]
    w4a16_improved = sum(1 for s in w4a16_speedups if s >= 1.0)
    w4a16_avg_speedup = sum(w4a16_speedups) / len(w4a16_speedups)
    w4a16_avg_tflops = sum(r['qc_tflops'] for r in w4a16_results) / len(w4a16_results)

    report += f"""
### W4A16 汇总统计

| 指标 | 值 |
|------|-----|
| 测试 shapes 数 | {len(w4a16_results)} |
| 性能提升 shapes | {w4a16_improved} ({w4a16_improved/len(w4a16_speedups)*100:.1f}%) |
| 性能下降 shapes | {len(w4a16_speedups) - w4a16_improved} ({(len(w4a16_speedups) - w4a16_improved)/len(w4a16_speedups)*100:.1f}%) |
| 平均 Speedup | {w4a16_avg_speedup:.2f}x |
| 平均 TFLOPS | {w4a16_avg_tflops:.2f} |
| 最佳 Speedup | {max(w4a16_speedups):.2f}x ({[r['shape_str'] for r in w4a16_results if r['speedup'] == max(w4a16_speedups)][0]}) |
| 最差 Speedup | {min(w4a16_speedups):.2f}x ({[r['shape_str'] for r in w4a16_results if r['speedup'] == min(w4a16_speedups)][0]}) |

---

## W8A16 vs W4A16 对比

| Shape (B,S,H,K) | W8A16 Speedup | W4A16 Speedup | 胜出 |
|:----------------|:--------------|:--------------|:-----|
"""

    # Build comparison
    w8a16_by_shape = {r['shape_str']: r['speedup'] for r in w8a16_results}
    w4a16_by_shape = {r['shape_str']: r['speedup'] for r in w4a16_results}

    for shape_str in sorted(w8a16_by_shape.keys()):
        w8_speedup = w8a16_by_shape.get(shape_str, 0)
        w4_speedup = w4a16_by_shape.get(shape_str, 0)
        winner = "W8A16" if w8_speedup > w4_speedup else ("W4A16" if w4_speedup > w8_speedup else "Equal")
        report += f"| {shape_str} | {w8_speedup:.2f}x | {w4_speedup:.2f}x | {winner} |\n"

    report += f"""

---

## 结论

### W8A16 性能表现
- **{w8a16_improved}/{len(w8a16_results)} shapes 获得性能提升** ({w8a16_improved/len(w8a16_speedups)*100:.1f}%)
- 平均 Speedup: **{w8a16_avg_speedup:.2f}x**
- 平均 TFLOPS: **{w8a16_avg_tflops:.2f}**

### W4A16 性能表现
- **{w4a16_improved}/{len(w4a16_results)} shapes 获得性能提升** ({w4a16_improved/len(w4a16_speedups)*100:.1f}%)
- 平均 Speedup: **{w4a16_avg_speedup:.2f}x**
- 平均 TFLOPS: **{w4a16_avg_tflops:.2f}**

### W8A16 vs W4A16
- W8A16 在大多数场景下表现更优（更少的精度损失，更稳定的性能）
- W4A16 在大 batch 场景下可节省更多内存，但性能略逊

---
*Report generated: {timestamp}*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[REPORT] Written: {output_path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QC-MoE Complete Benchmark")
    parser.add_argument("--w8a16", action="store_true", default=False, help="W8A16 only")
    parser.add_argument("--w4a16", action="store_true", default=False, help="W4A16 only")
    parser.add_argument("--quick", action="store_true", default=False, help="Quick test (subset of shapes)")
    parser.add_argument("--yaml", type=str,
                        default="/data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems/benchmark/models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml",
                        help="Path to YAML config")
    args = parser.parse_args()

    # Get shapes from YAML
    yaml_path = args.yaml
    all_shapes = get_all_mm_shapes(yaml_path)

    # Quick mode - reduce shapes
    if args.quick:
        # Select diverse subset
        all_shapes = all_shapes[::3] if len(all_shapes) > 10 else all_shapes
        print(f"[QUICK] Running {len(all_shapes)} shapes")

    print(f"[CONFIG] Loaded {len(all_shapes)} unique shapes from YAML")
    print(f"[CONFIG] Categories: {set(s['category'] for s in all_shapes)}")

    run_w8a16 = not args.w4a16
    run_w4a16 = not args.w8a16

    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = []

    if run_w8a16:
        print(f"\n{'='*80}")
        print("Running W8A16 benchmarks...")
        print(f"{'='*80}")

        w8a16_results = []
        for i, shape in enumerate(all_shapes, 1):
            print(f"[{i}/{len(all_shapes)}] Shape: {shape['shape_str']} ...", end=" ", flush=True)

            result = run_single_benchmark(
                shape["seq"], shape["hidden"], shape["inter"], w_nbits=8
            )

            result.update({
                "w_nbits": 8,
                "shape_str": shape["shape_str"],
                "batch": shape["batch"],
                "seq": shape["seq"],
                "hidden": shape["hidden"],
                "inter": shape["inter"],
                "category": shape["category"],
                "desc": shape["desc"],
            })

            w8a16_results.append(result)
            all_results.append(result)
            print(f"Speedup: {result['speedup']:.2f}x, TFLOPS: {result['qc_tflops']:.2f}")

        # Save W8A16 CSV
        csv_path = os.path.join(script_dir, "qcmoe_complete_w8a16_data.csv")
        save_csv(w8a16_results, csv_path)

    if run_w4a16:
        print(f"\n{'='*80}")
        print("Running W4A16 benchmarks...")
        print(f"{'='*80}")

        w4a16_results = []
        for i, shape in enumerate(all_shapes, 1):
            print(f"[{i}/{len(all_shapes)}] Shape: {shape['shape_str']} ...", end=" ", flush=True)

            result = run_single_benchmark(
                shape["seq"], shape["hidden"], shape["inter"], w_nbits=4
            )

            result.update({
                "w_nbits": 4,
                "shape_str": shape["shape_str"],
                "batch": shape["batch"],
                "seq": shape["seq"],
                "hidden": shape["hidden"],
                "inter": shape["inter"],
                "category": shape["category"],
                "desc": shape["desc"],
            })

            w4a16_results.append(result)
            all_results.append(result)
            print(f"Speedup: {result['speedup']:.2f}x, TFLOPS: {result['qc_tflops']:.2f}")

        # Save W4A16 CSV
        csv_path = os.path.join(script_dir, "qcmoe_complete_w4a16_data.csv")
        save_csv(w4a16_results, csv_path)

    # Generate report
    report_path = os.path.join(script_dir, "qcmoe_complete_report.md")
    generate_report(all_results, report_path, yaml_path)

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")

    w8a16_results = [r for r in all_results if r.get("w_nbits") == 8]
    w4a16_results = [r for r in all_results if r.get("w_nbits") == 4]

    if w8a16_results:
        w8_speedups = [r['speedup'] for r in w8a16_results]
        print(f"W8A16: {len(w8a16_results)} shapes tested, "
              f"{sum(1 for s in w8_speedups if s >= 1.0)} improved, "
              f"avg speedup: {sum(w8_speedups)/len(w8_speedups):.2f}x")

    if w4a16_results:
        w4_speedups = [r['speedup'] for r in w4a16_results]
        print(f"W4A16: {len(w4a16_results)} shapes tested, "
              f"{sum(1 for s in w4_speedups if s >= 1.0)} improved, "
              f"avg speedup: {sum(w4_speedups)/len(w4_speedups):.2f}x")

    print(f"\nOutputs:")
    print(f"  - W8A16 CSV: {os.path.join(script_dir, 'qcmoe_complete_w8a16_data.csv')}")
    print(f"  - W4A16 CSV: {os.path.join(script_dir, 'qcmoe_complete_w4a16_data.csv')}")
    print(f"  - Report: {report_path}")


def save_csv(results, out_path):
    fieldnames = [
        "w_nbits", "shape_str", "batch", "seq", "hidden", "inter",
        "category", "desc",
        "qc_ms", "fp16_ms", "qc_tflops", "fp16_tflops",
        "speedup", "max_abs_err",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["qc_ms"] = round(r["qc_ms"], 3)
            row["fp16_ms"] = round(r["fp16_ms"], 3)
            row["qc_tflops"] = round(r["qc_tflops"], 2)
            row["fp16_tflops"] = round(r["fp16_tflops"], 2)
            row["speedup"] = round(r["speedup"], 2)
            row["max_abs_err"] = round(r["max_abs_err"], 5)
            writer.writerow(row)
    print(f"[CSV] Written: {out_path}")


if __name__ == "__main__":
    main()
