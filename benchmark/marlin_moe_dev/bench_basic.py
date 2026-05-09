"""
Phase-1 latency benchmark for fused_marlin_moe (FlagGems wrapper).

Reports per-call latency (ms) across representative MoE shapes for:
  - fused_marlin_moe (W4A16 wrapper, this work)
  - fused_experts_impl (FP16 path, same underlying kernel)
  - reference PyTorch naive SwiGLU MoE (only on tiny shapes)

The wrapper is expected to be on par with the FP16 baseline at this stage
(the underlying fused_experts_impl dequantizes int4/int8 weights to FP16
before GEMM, so no real Marlin speedup yet). Phase 2 will replace that
with a true fused-dequant Triton kernel.
"""
import argparse
import time
import torch

from flag_gems.fused.fused_marlin_moe import (
    fused_marlin_moe,
    QUANT_TYPE_UINT4B8,
)
from flag_gems.fused.fused_moe import fused_experts_impl


# ---------------------------------------------------------------------------
# Reference (slow but obvious). Only runs on small shapes (M*K*N small).
# ---------------------------------------------------------------------------
def reference_swiglu_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    M, _ = hidden_states.shape
    _, two_N, _ = w1.shape
    N = two_N // 2
    topk = topk_ids.shape[1]
    out = torch.zeros_like(hidden_states)
    for m in range(M):
        for k in range(topk):
            e = topk_ids[m, k].item()
            w_topk = topk_weights[m, k]
            x = hidden_states[m]
            gate_up = w1[e] @ x
            gate = gate_up[:N]
            up = gate_up[N:]
            act = torch.nn.functional.silu(gate) * up
            y = w2[e] @ act
            out[m] += w_topk.to(y.dtype) * y
    return out


BENCH_CASES = [
    # (label, M, K, N, E, topk, run_naive)
    ("tiny",         8,   128,  256,   8,  2, True),
    ("medium",       64,  1024, 2048,  8,  2, False),
    ("mixtral-b1",   1,   4096, 14336, 8,  2, False),
    ("mixtral-b16", 16,   4096, 14336, 8,  2, False),
    ("mixtral-b64", 64,   4096, 14336, 8,  2, False),
    ("ds-b16",      16,   2048, 1408,  64, 6, False),
    ("ds-b64",      64,   2048, 1408,  256, 8, False),
]


def bench(fn, n_warmup=3, n_iter=10):
    """Warmup + sync + median of n_iter runs (ms)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]   # median


def make_inputs(M, K, N, E, topk, dtype, device='cuda', seed=0):
    torch.manual_seed(seed)
    scale = 0.05
    hidden = torch.randn(M, K, dtype=dtype, device=device) * scale
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device=device) * scale
    w2 = torch.randn(E, K, N,    dtype=dtype, device=device) * scale
    topk_weights = torch.softmax(torch.randn(M, topk, device=device), dim=-1).float()
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.long, device=device)
    w1_scale = torch.ones(E, 2 * N, dtype=dtype, device=device)
    w2_scale = torch.ones(E, K,     dtype=dtype, device=device)
    return hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


def call_wrapper(hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
    return fused_marlin_moe(
        hidden_states=hidden, w1=w1, w2=w2,
        bias1=None, bias2=None,
        w1_scale=w1_scale, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )


def call_fp16_baseline(hidden, w1, w2, topk_weights, topk_ids):
    """Same underlying kernel via fused_experts_impl, no quantization flags."""
    return fused_experts_impl(
        hidden_states=hidden, w1=w1, w2=w2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        activation="silu",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cases = BENCH_CASES
    if args.only:
        cases = [c for c in cases if args.only in c[0]]

    print(f"\n[dtype={args.dtype}, n_iter={args.n_iter}, device={torch.cuda.get_device_name(0)}]")
    print("=" * 100)
    print(f"{'LABEL':<14} {'SHAPE':<32} {'WRAPPER(ms)':<12} "
          f"{'FP16(ms)':<10} {'NAIVE(ms)':<12} {'WRAP/FP16':<10}")
    print("-" * 100)

    for label, M, K, N, E, topk, run_naive in cases:
        hidden, w1, w2, tw, ti, w1s, w2s = make_inputs(M, K, N, E, topk, dtype)

        wrap_ms = bench(
            lambda: call_wrapper(hidden, w1, w2, tw, ti, w1s, w2s),
            n_iter=args.n_iter,
        )
        fp16_ms = bench(
            lambda: call_fp16_baseline(hidden, w1, w2, tw, ti),
            n_iter=args.n_iter,
        )
        ratio = wrap_ms / fp16_ms if fp16_ms > 0 else float('inf')

        if run_naive:
            naive_ms = bench(
                lambda: reference_swiglu_moe(hidden, w1, w2, tw, ti),
                n_warmup=1, n_iter=3,
            )
            naive_str = f"{naive_ms:>9.1f}"
        else:
            naive_str = "        -"

        shape_str = f"M={M} K={K} N={N} E={E} k={topk}"
        print(f"{label:<14} {shape_str:<32} "
              f"{wrap_ms:>9.3f}    {fp16_ms:>7.3f}   {naive_str:<12} "
              f"{ratio:>7.2f}x")

    print("=" * 100)


if __name__ == "__main__":
    main()
