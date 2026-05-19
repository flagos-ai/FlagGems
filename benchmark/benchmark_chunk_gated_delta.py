"""
benchmark_chunk_gated_delta.py
===============================
Standalone performance benchmark for chunk_gated_delta_rule.

Compares the Triton (FlagGems) implementation against the PyTorch sequential
reference across a range of sequence lengths.

Usage::
    python benchmark/benchmark_chunk_gated_delta.py

Expected output (example on A100 with D=64):
    chunk_gated_delta_performance:
       SEQ_LEN  FlagGems (Triton)  Baseline (PyTorch)  Speedup
    0    128.0             XX.X               YY.Y       Z.Zx
    ...
"""

import math

import pandas as pd
import torch

from flag_gems.ops.chunk_gated_delta import (
    chunk_gated_delta_rule,
    torch_chunk_gated_delta_rule,
)

# ── benchmark settings ───────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE = torch.float32  # use float32 for deterministic precision comparison
B, H, D = 2, 4, 64  # batch, heads, head-dim (typical attention shape)
BT = 16  # chunk-size arg (kept for API compat)
WARMUP = 10  # kernel warm-up iterations
REPS = 50  # timed repetitions


def _make_inputs(T: int):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE) / math.sqrt(D)
    v = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE) / math.sqrt(D)
    beta = torch.rand(B, H, T, device=DEVICE, dtype=DTYPE).clamp(min=0.01)
    return q, k, v, beta


def _measure_us(fn, *args, warmup: int = WARMUP, reps: int = REPS) -> float:
    """Return median latency in microseconds."""
    # warm-up
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000.0 / reps  # ms → µs


def main():
    seq_lens = [128, 256, 512, 1024, 2048]
    rows = []

    for T in seq_lens:
        q, k, v, beta = _make_inputs(T)

        t_triton = _measure_us(chunk_gated_delta_rule, q, k, v, beta, BT)
        t_pytorch = _measure_us(torch_chunk_gated_delta_rule, q, k, v, beta)

        rows.append(
            {
                "SEQ_LEN": float(T),
                "FlagGems (Triton)": t_triton,
                "Baseline (PyTorch)": t_pytorch,
                "Speedup": t_pytorch / t_triton,
            }
        )
        print(
            f"T={T:5d}  Triton={t_triton:8.2f} µs  "
            f"PyTorch={t_pytorch:8.2f} µs  "
            f"Speedup={t_pytorch / t_triton:.2f}x"
        )

    df = pd.DataFrame(rows)
    print("\nchunk_gated_delta_performance:")
    print(df.to_string(index=True))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available – skipping benchmark")
    main()
