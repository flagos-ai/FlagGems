#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Measure the performance impact of precompilation optimization.

Compare:
- Before: cold start (includes JIT compilation overhead)
- After: precompiled (steady-state performance)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import torch
import flag_gems

def benchmark(fn, warmup=5, iters=20):
    """Return average time per call (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


def test_config(name, batch, q_heads, kv_heads, seq_len, head_dim, dtype=torch.bfloat16):
    """Test one config: cold start vs precompiled."""
    device = "cuda"
    scale = float(1.0 / (head_dim ** 0.5))

    # Prepare inputs
    q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device)

    def call_fa():
        return flag_gems.flash_attention_forward(
            q, k, v, None, None, q_heads, kv_heads,
            0.0, True, False, scale=scale
        )

    # Test 1: Cold start (without precompilation)
    with flag_gems.use_gems():
        # First call includes JIT compilation
        cold_start = time.time()
        _ = call_fa()
        torch.cuda.synchronize()
        cold_first_call = (time.time() - cold_start) * 1000

        # Subsequent calls (steady state after JIT)
        cold_steady = benchmark(call_fa, warmup=3, iters=10)

    # Test 2: With precompilation
    with flag_gems.use_gems():
        # Precompile
        precompile_start = time.time()
        flag_gems.precompile_flash_attention(
            head_dims=[head_dim],
            seq_lens=[seq_len],
            gqa_configs=[(q_heads, kv_heads)],
            batch_size=batch,
            dtype=dtype,
            device=device,
            verbose=False,
            scale=scale,
        )
        torch.cuda.synchronize()
        precompile_time = (time.time() - precompile_start) * 1000

        # First call after precompilation (should be fast)
        warm_start = time.time()
        _ = call_fa()
        torch.cuda.synchronize()
        warm_first_call = (time.time() - warm_start) * 1000

        # Steady state
        warm_steady = benchmark(call_fa, warmup=3, iters=10)

    # Calculate improvement
    first_call_speedup = cold_first_call / warm_first_call
    overhead_eliminated = cold_first_call - warm_first_call

    print(f"\n{name} (Q{q_heads}/KV{kv_heads}, d{head_dim}, seq{seq_len})")
    print(f"  Cold first call:     {cold_first_call:7.2f} ms (includes JIT)")
    print(f"  Cold steady state:   {cold_steady:7.2f} ms")
    print(f"  Precompile time:     {precompile_time:7.2f} ms")
    print(f"  Warm first call:     {warm_first_call:7.2f} ms")
    print(f"  Warm steady state:   {warm_steady:7.2f} ms")
    print(f"  → First-call speedup: {first_call_speedup:.2f}x")
    print(f"  → Overhead eliminated: {overhead_eliminated:.2f} ms")

    return {
        "name": name,
        "cold_first": cold_first_call,
        "cold_steady": cold_steady,
        "precompile": precompile_time,
        "warm_first": warm_first_call,
        "warm_steady": warm_steady,
        "speedup": first_call_speedup,
        "overhead_saved": overhead_eliminated,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("FlashAttention Precompilation Optimization Impact")
    print("=" * 80)

    # Test representative configs
    configs = [
        ("Qwen2.5-7B", 1, 28, 4, 1024, 128),
        ("Qwen2.5-1.5B", 1, 12, 2, 1024, 128),
        ("GLM-4-9B", 1, 32, 2, 1024, 128),
        ("Llama-3.2-3B", 1, 24, 8, 1024, 128),
    ]

    results = []
    for cfg in configs:
        result = test_config(*cfg)
        results.append(result)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_overhead = sum(r["overhead_saved"] for r in results) / len(results)
    print(f"Average first-call speedup: {avg_speedup:.2f}x")
    print(f"Average overhead eliminated: {avg_overhead:.2f} ms")
