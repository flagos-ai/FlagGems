"""
Benchmark: fused_add_rms_norm
Compares: FlagGems vs torch.compile vs vLLM (if available)
"""

import time

import torch

import flag_gems


# ── reference: naive torch ──────────────────────────────────────────────
def torch_fused_add_rms_norm(x, residual, weight, eps=1e-5):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight, x


# ── reference: torch.compile ────────────────────────────────────────────
@torch.compile
def compiled_fused_add_rms_norm(x, residual, weight, eps=1e-5):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight, x


# ── reference: vLLM ─────────────────────────────────────────────────────
try:
    import os

    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    from vllm._custom_ops import fused_add_rms_norm as vllm_fused_add_rms_norm

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    print("vLLM not available, skipping vLLM baseline\n")


# ── benchmark helper ────────────────────────────────────────────────────
def bench_fn(fn, warmup=20, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / rep * 1000  # ms


# ── main ────────────────────────────────────────────────────────────────
shapes = [
    (1, 4096),
    (32, 4096),
    (128, 4096),
    (512, 4096),
    (1024, 4096),
    (4096, 4096),
    (128, 8192),
    (128, 11008),
]
dtypes = [torch.float16, torch.bfloat16]

print(f"{'shape':>18s} {'dtype':>10s} | {'naive':>8s} {'compile':>8s}", end="")
if HAS_VLLM:
    print(f" {'vllm':>8s}", end="")
print(f" {'flaggems':>8s}  (ms)")
print("-" * 80)

for shape in shapes:
    for dtype in dtypes:
        M, N = shape
        device = "cuda"
        eps = 1e-5

        x_ref = torch.randn(M, N, dtype=dtype, device=device)
        r_ref = torch.randn(M, N, dtype=dtype, device=device)
        w = torch.randn(N, dtype=dtype, device=device)

        # ── naive torch ──
        t_naive = bench_fn(
            lambda: torch_fused_add_rms_norm(x_ref.clone(), r_ref.clone(), w, eps)
        )

        # ── torch.compile ──
        # warmup compile
        _ = compiled_fused_add_rms_norm(x_ref.clone(), r_ref.clone(), w, eps)
        t_compile = bench_fn(
            lambda: compiled_fused_add_rms_norm(x_ref.clone(), r_ref.clone(), w, eps)
        )

        # ── vLLM ──
        if HAS_VLLM:
            # vLLM's fused_add_rms_norm is in-place: (x, residual) modified
            def run_vllm():
                xc = x_ref.clone()
                rc = r_ref.clone()
                vllm_fused_add_rms_norm(xc, rc, w, eps)

            t_vllm = bench_fn(run_vllm)

        # ── FlagGems ──
        def run_gems():
            xc = x_ref.clone()
            rc = r_ref.clone()
            flag_gems.fused_add_rms_norm(xc, rc, (N,), w, eps)

        t_gems = bench_fn(run_gems)

        # ── print ──
        tag = f"({M}, {N})"
        print(f"{tag:>18s} {str(dtype):>10s} | {t_naive:8.3f} {t_compile:8.3f}", end="")
        if HAS_VLLM:
            print(f" {t_vllm:8.3f}", end="")
        print(f" {t_gems:8.3f}")
