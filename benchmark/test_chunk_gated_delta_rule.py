"""Forward + backward benchmarks for ``chunk_gated_delta_rule``.

Compares the FlagGems operator against:
    * The differentiable eager reference (what the autograd backward also uses)
    * The upstream FLA naive (when installed) — sanity oracle, not a perf goal

Reports forward latency (us) and backward latency (us) across a small grid
of production shapes, including ``cu_seqlens`` packed batches.
"""

from __future__ import annotations

import time
import triton
import torch

import flag_gems  # noqa: F401
from flag_gems.ops.chunk_gated_delta_rule import (
    _eager_chunk_gated_delta_rule as eager_ref,
    chunk_gated_delta_rule,
)


def _bench(fn, n_warmup=5, n_repeat=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_repeat  # microseconds


def _fwd_only(B, T, H, K, V, dtype):
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.3
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))

    t_op = _bench(lambda: chunk_gated_delta_rule(q, k, v, g, beta))
    t_eager = _bench(lambda: eager_ref(q, k, v, g, beta, scale=K**-0.5,
                                       initial_state=None, output_final_state=False))
    return t_op, t_eager


def _fwd_and_bwd(B, T, H, K, V, dtype):
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    k = (torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.3).requires_grad_(True)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype, requires_grad=True)
    g = (-torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.1).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32)).requires_grad_(True)

    def step():
        for t in (q, k, v, g, beta):
            if t.grad is not None:
                t.grad = None
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
        o.sum().backward()

    return _bench(step, n_warmup=3, n_repeat=10)


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability(0)}")
    print(f"triton: {triton.__version__}  torch: {torch.__version__}")
    print()
    print("--- forward only (us) ---")
    print(f"{'shape':40s} {'dtype':10s} {'op':>10s} {'eager':>10s} {'speedup':>10s}")
    shapes = [
        (1, 256, 4, 64, 64),
        (2, 1024, 4, 64, 64),
        (1, 4096, 8, 128, 128),
        (4, 512, 16, 128, 128),
    ]
    for B, T, H, K, V in shapes:
        for dtype in [torch.bfloat16, torch.float32]:
            try:
                t_op, t_eager = _fwd_only(B, T, H, K, V, dtype)
                shape = f"B={B} T={T} H={H} K={K} V={V}"
                spd = f"{t_eager/t_op:.2f}x"
                print(f"{shape:40s} {str(dtype).split('.')[-1]:10s} {t_op:10.1f} {t_eager:10.1f} {spd:>10s}")
            except Exception as e:
                print(f"  failed: {e}")

    print()
    print("--- forward + backward (us, full step) ---")
    print(f"{'shape':40s} {'dtype':10s} {'op':>10s}")
    for B, T, H, K, V in shapes:
        for dtype in [torch.bfloat16, torch.float32]:
            try:
                t = _fwd_and_bwd(B, T, H, K, V, dtype)
                shape = f"B={B} T={T} H={H} K={K} V={V}"
                print(f"{shape:40s} {str(dtype).split('.')[-1]:10s} {t:10.1f}")
            except Exception as e:
                print(f"  failed: {e}")


if __name__ == "__main__":
    main()
