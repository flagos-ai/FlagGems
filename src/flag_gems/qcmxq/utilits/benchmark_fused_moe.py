# SPDX-License-Identifier: Apache-2.0
# benchmark_fused_moe.py
# QC-MoE W8A16 性能测试

import torch
import time
import argparse
import os
import csv as csv_module

from ..fused_moe_mxq import fused_moe
from ..ultis import (
    QuantConfig,
    quantize_weights_moe,
    fp16_moe_w1_only_reference,
    QWEN3_SHAPES,
    QWEN3_DEFAULT_CONFIG,
)


NUM_EXPERTS = 8
TOP_K = 2
GROUP_SIZE = 128
WARMUP = 10
NUM_ITERS = 100
RANDOM_SEED = 42


def run_benchmark_w8a16(
    seq_len: int,
    hidden_dim: int,
    inter_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> dict:
    torch.manual_seed(RANDOM_SEED)
    
    W1_fp16 = torch.randn(NUM_EXPERTS, inter_dim, hidden_dim, dtype=dtype, device=device)
    W1_q, W1_sc, W1_z = quantize_weights_moe(W1_fp16, w_nbits=8, group_size=GROUP_SIZE)
    
    inp = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
    topk_weights = torch.ones(seq_len, TOP_K, dtype=dtype, device=device) / TOP_K
    topk_ids = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K), dtype=torch.int64, device=device)
    
    ref_output = fp16_moe_w1_only_reference(inp, W1_fp16, topk_weights, topk_ids)
    
    quant_config = QuantConfig(mode="w8a16", group_size=GROUP_SIZE)
    for _ in range(WARMUP):
        _ = fused_moe(
            inp, None, None, None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=quant_config,
            num_experts=NUM_EXPERTS, top_k=TOP_K,
            w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
        )
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        out_qc = fused_moe(
            inp, None, None, None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=quant_config,
            num_experts=NUM_EXPERTS, top_k=TOP_K,
            w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
        )
    torch.cuda.synchronize()
    qc_ms = (time.perf_counter() - t0) / NUM_ITERS * 1000
    
    total_flops = seq_len * TOP_K * hidden_dim * inter_dim
    qc_tflops = total_flops / qc_ms / 1e9
    max_abs_err = (out_qc.float() - ref_output.float()).abs().max().item()
    
    return {
        "qc_ms": qc_ms,
        "qc_tflops": qc_tflops,
        "max_abs_err": max_abs_err,
    }


def run_benchmark_fp16(
    seq_len: int,
    hidden_dim: int,
    inter_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> dict:
    torch.manual_seed(RANDOM_SEED)
    
    W1_fp16 = torch.randn(NUM_EXPERTS, inter_dim, hidden_dim, dtype=dtype, device=device)
    
    inp = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
    topk_weights = torch.ones(seq_len, TOP_K, dtype=dtype, device=device) / TOP_K
    topk_ids = torch.randint(0, NUM_EXPERTS, (seq_len, TOP_K), dtype=torch.int64, device=device)
    
    quant_config = QuantConfig(mode="fp16")
    
    for _ in range(WARMUP):
        _ = fused_moe(
            inp, W1_fp16, None, None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=quant_config,
            num_experts=NUM_EXPERTS, top_k=TOP_K,
        )
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        _ = fused_moe(
            inp, W1_fp16, None, None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=quant_config,
            num_experts=NUM_EXPERTS, top_k=TOP_K,
        )
    torch.cuda.synchronize()
    fp16_ms = (time.perf_counter() - t0) / NUM_ITERS * 1000
    
    return {"fp16_ms": fp16_ms}


def run_full_benchmark():
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return []
    
    device = "cuda"
    dtype = torch.float16
    
    print("\n" + "=" * 90)
    print("QC-MoE W8A16 vs FP16 参考实现性能对比")
    print("=" * 90)
    
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"[环境] GPU: {GPU_NAME}")
    print(f"[环境] PyTorch: {torch.__version__}")
    import triton
    print(f"[环境] Triton: {triton.__version__}")
    print()
    
    print(f"{'Shape (S,H,K)':<24} {'W8A16 ms':<14} {'FP16 ms':<14} {'加速比':<10} {'W8A16 TFLOPS':<14} {'最大误差':<12}")
    print("-" * 90)
    
    results = []
    
    for name, (S, H, K) in QWEN3_SHAPES.items():
        w8a16_result = run_benchmark_w8a16(S, H, K, dtype, device)
        fp16_result = run_benchmark_fp16(S, H, K, dtype, device)
        
        speedup = fp16_result["fp16_ms"] / w8a16_result["qc_ms"]
        
        row = (
            f"({S},{H},{K})      "
            f"{w8a16_result['qc_ms']:<14.3f} "
            f"{fp16_result['fp16_ms']:<14.3f} "
            f"{speedup:<10.2f}x "
            f"{w8a16_result['qc_tflops']:<14.2f} "
            f"{w8a16_result['max_abs_err']:<12.5f}"
        )
        print(row)
        
        results.append({
            "shape": f"({S},{H},{K})",
            "seq_len": S,
            "hidden_dim": H,
            "inter_dim": K,
            "w8a16_ms": round(w8a16_result["qc_ms"], 3),
            "fp16_ms": round(fp16_result["fp16_ms"], 3),
            "speedup": round(speedup, 2),
            "w8a16_tflops": round(w8a16_result["qc_tflops"], 2),
            "max_abs_err": round(w8a16_result["max_abs_err"], 5),
        })
    
    print("=" * 90)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="QC-MoE W8A16 性能测试")
    parser.add_argument("--csv", action="store_true", default=False, help="保存 CSV 结果")
    args = parser.parse_args()
    
    results = run_full_benchmark()
    
    if args.csv and results:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, "qcmoe_w8a16_benchmark.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[保存] CSV 结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
