"""Benchmark: Triton topk_softplus_sqrt vs vLLM CUDA kernel (direct call).

Calls torch.ops._moe_C.topk_softplus_sqrt directly, bypassing fused_topk_bias
dispatch wrapper for fair kernel-to-kernel comparison.

Production MoE gating shapes (30 configs total):
  Standard mode (bias routing):
    - DeepSeek-V3/R1: 256 experts, topk=8 (specialized k=8 kernel)
    - Qwen3-235B/30B: 128 experts, topk=8 (specialized k=8 kernel)
    - Generic path: 256 experts, topk=6/16 (generic kernel)
  Hash mode (DeepSeek-V4-Pro): 384 experts, topk=6
  num_tokens: 8, 32, 128, 512, 2048, 4096 (decode → prefill)

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark_topk_softplus_sqrt.py
"""

import time
import sys
import torch

from flag_gems.fused.topk_softplus_sqrt import topk_softplus_sqrt as triton_topk

# Import vLLM CUDA kernel directly (bypass fused_topk_bias dispatch)
try:
    from vllm._custom_ops import topk_hash_softplus_sqrt as _cuda_topk_softplus_sqrt
    HAS_CUDA = True
except Exception as e:
    HAS_CUDA = False
    print(f"WARNING: vLLM CUDA kernel not available ({e}), skipping CUDA comparison.\n")


def cuda_topk_call(gating_output, topk, renormalize, routed_scaling_factor,
                   correction_bias, topk_weights, topk_indices, token_expert_indices):
    """Direct CUDA kernel call with pre-allocated output tensors."""
    _cuda_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        None,  # input_tokens
        None,  # hash_indices_table
    )


def cuda_hash_call(gating_output, topk, renormalize, routed_scaling_factor,
                   input_ids, tid2eid, topk_weights, topk_indices, token_expert_indices):
    """Direct CUDA kernel call for hash mode with pre-allocated output tensors."""
    _cuda_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        None,  # e_score_correction_bias
        input_ids,
        tid2eid,
    )


def triton_topk_wrapper(gating_output, topk, renormalize, routed_scaling_factor, correction_bias):
    return triton_topk(
        gating_output, topk, renormalize, routed_scaling_factor,
        correction_bias=correction_bias,
    )


def triton_hash_wrapper(gating_output, topk, renormalize, routed_scaling_factor, input_ids, tid2eid):
    return triton_topk(
        gating_output, topk, renormalize, routed_scaling_factor,
        input_ids=input_ids, tid2eid=tid2eid,
    )


def bench_fn(fn, *args, warmup=10, n_iter=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6  # microseconds


# Production shapes from trace: DeepSeek-V4-Flash on H20 (TP=4)
# Model config: 256 experts, topk=6, renormalize=True, rsf=1.5
#   43 layers total: 40 standard + 3 hash
#   Decode (num_tokens=1) dominates ~99.5% of calls
#
# Standard mode: (num_tokens, num_experts, topk, label, weight%)
BENCH_SHAPES = [
    # Decode — 92.57% of all calls
    (1, 256, 6, "1tok-256exp-k6", 92.57),
    # Prefill — from trace (various prompt lengths)
    (10, 256, 6, "10tok-256exp-k6", 0.05),
    (16, 256, 6, "16tok-256exp-k6", 0.05),
    (255, 256, 6, "255tok-256exp-k6", 0.05),
    (805, 256, 6, "805tok-256exp-k6", 0.05),
    (1024, 256, 6, "1024tok-256exp-k6", 0.05),
    (2005, 256, 6, "2005tok-256exp-k6", 0.05),
]

# Hash mode: (num_tokens, num_experts, topk, label, weight%)
# 3 hash layers per forward pass
HASH_BENCH_SHAPES = [
    # Decode — 6.94% of all calls
    (1, 256, 6, "1tok-256exp-k6-hash", 6.94),
    # Prefill
    (10, 256, 6, "10tok-256exp-k6-hash", 0.00),
    (16, 256, 6, "16tok-256exp-k6-hash", 0.00),
    (255, 256, 6, "255tok-256exp-k6-hash", 0.00),
    (805, 256, 6, "805tok-256exp-k6-hash", 0.00),
    (1024, 256, 6, "1024tok-256exp-k6-hash", 0.00),
    (2005, 256, 6, "2005tok-256exp-k6-hash", 0.00),
]


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16

    print("=" * 90)
    print("Benchmark: Triton topk_softplus_sqrt vs vLLM CUDA kernel (direct, bf16)")
    print("  renormalize=True, routed_scaling_factor=1.0")
    print("=" * 90)

    hdr = f"  {'config':>25s}  {'triton(us)':>10s}"
    if HAS_CUDA:
        hdr += f"  {'cuda(us)':>10s}  {'speedup':>8s}"
    print(hdr)
    print("-" * len(hdr))

    results = {}

    # --- Standard mode (bias routing) ---
    for num_tokens, num_experts, topk, label, weight in BENCH_SHAPES:
        gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
        correction_bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")

        tri_us = bench_fn(triton_topk_wrapper, gating_output, topk, True, 1.0, correction_bias)

        if HAS_CUDA:
            # Pre-allocate output tensors for fair comparison
            topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
            topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
            token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

            cuda_us = bench_fn(
                cuda_topk_call, gating_output, topk, True, 1.0, correction_bias,
                topk_weights, topk_indices, token_expert_indices,
            )
            speedup = cuda_us / tri_us if tri_us > 0 else 0
            print(f"  {label:>25s}  {tri_us:10.1f}  {cuda_us:10.1f}  {speedup:7.4f}x")
            results[label] = {"triton_us": tri_us, "cuda_us": cuda_us, "speedup": speedup}
        else:
            print(f"  {label:>25s}  {tri_us:10.1f}")
            results[label] = {"triton_us": tri_us}

    # --- Hash mode (DeepSeek-V4 routing) ---
    print()
    print("--- Hash mode (DeepSeek-V4-Pro: 384 experts, topk=6) ---")
    hdr2 = f"  {'config':>25s}  {'triton(us)':>10s}"
    if HAS_CUDA:
        hdr2 += f"  {'cuda(us)':>10s}  {'speedup':>8s}"
    print(hdr2)
    print("-" * len(hdr2))

    vocab_size = 1024
    for num_tokens, num_experts, topk, label, weight in HASH_BENCH_SHAPES:
        gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
        tid2eid = torch.stack(
            [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
        ).to(device="cuda", dtype=torch.int32)
        input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32, device="cuda")

        tri_us = bench_fn(triton_hash_wrapper, gating_output, topk, True, 2.5, input_ids, tid2eid)

        if HAS_CUDA:
            topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
            topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
            token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

            cuda_us = bench_fn(
                cuda_hash_call, gating_output, topk, True, 2.5,
                input_ids, tid2eid,
                topk_weights, topk_indices, token_expert_indices,
            )
            speedup = cuda_us / tri_us if tri_us > 0 else 0
            print(f"  {label:>25s}  {tri_us:10.1f}  {cuda_us:10.1f}  {speedup:7.4f}x")
            results[label] = {"triton_us": tri_us, "cuda_us": cuda_us, "speedup": speedup}
        else:
            print(f"  {label:>25s}  {tri_us:10.1f}")
            results[label] = {"triton_us": tri_us}

    return results


if __name__ == "__main__":
    main()
