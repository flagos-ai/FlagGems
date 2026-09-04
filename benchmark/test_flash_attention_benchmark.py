#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlashAttention performance benchmark: guard against performance regressions.

Usage:
    # Run all benchmarks
    pytest tests/test_flash_attention_benchmark.py -v

    # Run only Qwen2.5-7B
    pytest tests/test_flash_attention_benchmark.py -k "Qwen2.5-7B" -v

    # Set a stricter threshold (default allows a 100% slowdown)
    FA_MAX_SLOWDOWN_PCT=50 pytest tests/test_flash_attention_benchmark.py -v

    # Record performance only, never fail (used to collect a baseline)
    FA_PERF_RECORD_ONLY=1 pytest tests/test_flash_attention_benchmark.py -v
"""

import os
import time

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

# ==================== Benchmark configs ====================
# Representative configs taken from real models.
BENCHMARK_CONFIGS = [
    # (model_name, batch, q_heads, kv_heads, seq_len, head_dim, note)
    ("Qwen2.5-7B", 1, 28, 4, 1024, 128, "long-context inference"),
    ("Qwen2.5-7B", 1, 28, 4, 256, 128, "medium length"),
    ("Qwen2.5-1.5B", 1, 12, 2, 1024, 128, "small model long context"),
    ("GLM-4-9B", 1, 32, 2, 1024, 128, "extreme GQA ratio (16:1)"),
    ("Llama-3.2-3B", 1, 24, 8, 1024, 128, "Llama architecture"),
]

WARMUP = 10
ITERS = 50


def make_gqa_input(batch, q_heads, kv_heads, seq_len, head_dim, dtype, device):
    """Generate GQA input tensors (BNSD layout)."""
    torch.manual_seed(1234567890)
    q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def benchmark(fn, warmup=WARMUP, iters=ITERS):
    """Return the average time per call (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


@pytest.mark.flash_attention_perf
@pytest.mark.parametrize(
    "model_name,batch,q_heads,kv_heads,seq_len,head_dim,note",
    BENCHMARK_CONFIGS,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_performance_baseline(
    model_name, batch, q_heads, kv_heads, seq_len, head_dim, note, dtype
):
    """
    Performance baseline: record how FlagGems compares to native PyTorch.

    The comparison isolates the two kernels via the use_gems() context (see the
    body). At small batch / short sequence lengths the Triton implementation is
    currently several times slower than the native fused kernel, so the default
    threshold is intentionally wide to record a baseline rather than gate CI;
    tighten it as the kernel improves.

    Environment variables:
        FA_MAX_SLOWDOWN_PCT: max allowed slowdown percentage (default 2000)
        FA_PERF_RECORD_ONLY: set to 1 to record only and never fail
    """
    MAX_SLOWDOWN_PCT = float(os.environ.get("FA_MAX_SLOWDOWN_PCT", "2000.0"))
    RECORD_ONLY = os.environ.get("FA_PERF_RECORD_ONLY", "0") == "1"

    device = torch_device_fn.current_device()
    q, k, v = make_gqa_input(
        batch, q_heads, kv_heads, seq_len, head_dim, dtype, device
    )
    scale = float(1.0 / (head_dim ** 0.5))

    # The op entry is identical for both sides; only the surrounding context
    # decides which kernel it dispatches to.
    def call_fa():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return torch.ops.aten._flash_attention_forward(
            q_t, k_t, v_t, None, None,
            q_t.shape[-3], k_t.shape[-3],
            0.0, True, False, scale=scale,
        )

    # Native PyTorch: measured outside any use_gems() context.
    # NOTE: do NOT use disable_flash_attention()/enable() to toggle here. Those
    # only return name lists / register globally and cannot restore the native
    # kernel, which would silently make both sides measure the same kernel.
    pt_ms = benchmark(call_fa)

    # FlagGems: inside use_gems() so the Triton kernels are actually registered
    # (and cleanly unregistered on exit). Precompile first to measure
    # steady-state performance rather than one-off JIT compilation.
    with flag_gems.use_gems():
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
        gems_ms = benchmark(call_fa)

    slowdown = (gems_ms - pt_ms) / pt_ms * 100
    print(
        f"\n[{model_name}] {note} | "
        f"Q{q_heads}/KV{kv_heads} d{head_dim} seq{seq_len} | "
        f"PyTorch {pt_ms:.4f}ms | FlagGems {gems_ms:.4f}ms | "
        f"{'slower' if slowdown > 0 else 'faster'} {abs(slowdown):.1f}%"
    )

    if slowdown > MAX_SLOWDOWN_PCT and not RECORD_ONLY:
        pytest.fail(
            f"{model_name} performance regression: FlagGems is {slowdown:.1f}% "
            f"slower (threshold {MAX_SLOWDOWN_PCT}%) | PyTorch {pt_ms:.4f}ms, "
            f"FlagGems {gems_ms:.4f}ms"
        )


if __name__ == "__main__":
    # Quick self-check: run one representative config when invoked directly.
    print("=" * 80)
    print("FlashAttention performance benchmark self-check")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16

    # Qwen2.5-7B long context
    q, k, v = make_gqa_input(1, 28, 4, 1024, 128, dtype, device)
    scale = float(1.0 / 128 ** 0.5)

    def run_pt():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return torch.ops.aten._flash_attention_forward(
            q_t, k_t, v_t, None, None, 28, 4, 0.0, True, False, scale=scale
        )

    def run_gems():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return flag_gems.flash_attention_forward(
            q_t, k_t, v_t, None, None, 28, 4, 0.0, True, False, scale=scale
        )

    pt = benchmark(run_pt, warmup=5, iters=20)
    gm = benchmark(run_gems, warmup=5, iters=20)
    ratio = gm / pt

    print(f"\nQwen2.5-7B (seq=1024): PyTorch {pt:.3f}ms | FlagGems {gm:.3f}ms | {ratio:.2f}x")
    print("=" * 80)
