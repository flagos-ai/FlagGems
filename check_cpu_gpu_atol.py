"""Compare PyTorch CPU vs GPU linalg.matrix_norm for ord=-2, float64.

Uses the same input generation as test_linalg_matrix_norm.py (seed=0).
"""

import torch

_SEED = 0
_SHAPE = (256, 16)
_DTYPE = torch.float64
_ORD = -2


def make_input(shape, dtype, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(_SEED)
    return torch.randn(shape, dtype=torch.float32, generator=g).to(
        dtype=dtype, device=device
    )


def main():
    print(f"shape={_SHAPE}, dtype={_DTYPE}, ord={_ORD}")
    print()

    # Generate identical inputs on CPU and GPU
    A_cpu = make_input(_SHAPE, _DTYPE, "cpu")
    A_gpu = make_input(_SHAPE, _DTYPE, "cuda")

    # Verify inputs match
    input_diff = (A_cpu - A_gpu.cpu()).abs().max().item()
    print(f"Input max diff (cpu vs gpu): {input_diff:.2e}")
    print(f"Input norm (CPU): {A_cpu.norm().item():.10f}")
    print(f"Input norm (GPU): {A_gpu.norm().item():.10f}")
    print()

    # Compute reference on CPU
    ref_cpu = torch.linalg.matrix_norm(A_cpu, _ORD).item()
    print(f"CPU result:   {ref_cpu:.15e}")

    # Compute reference on GPU (cuSOLVER)
    ref_gpu = torch.linalg.matrix_norm(A_gpu, _ORD).item()
    print(f"GPU result:   {ref_gpu:.15e}")

    # Also compute both SVDs for deeper inspection
    print()
    print("--- SVD comparison ---")
    _, S_cpu, _ = torch.linalg.svd(A_cpu)
    _, S_gpu, _ = torch.linalg.svd(A_gpu)

    min_sv_cpu = S_cpu.min().item()
    min_sv_gpu = S_gpu.min().item()
    print(f"min  SV cpu: {min_sv_cpu:.15e}")
    print(f"min  SV gpu: {min_sv_gpu:.15e}")
    print(f"max  SV cpu: {S_cpu.max().item():.15e}")
    print(f"max  SV gpu: {S_gpu.max().item():.15e}")

    sv_diff = (S_cpu - S_gpu.cpu()).abs().max().item()
    print(f"SV max diff:  {sv_diff:.2e}")

    # Differences
    abs_diff = abs(ref_cpu - ref_gpu)
    rel_diff = abs_diff / (abs(ref_cpu) + 1e-30)

    print()
    print("--- CPU vs GPU atol / rtol ---")
    print(f"absolute difference: {abs_diff:.6e}")
    print(f"relative difference: {rel_diff:.6e}")

    # Also compare with fp64 SVD computed on CPU (gold standard)
    print()
    print("--- ord=-2 = 1/max_sigma ---")
    # ord=-2 is the minimum singular value. Actually ord=-2 is the MINIMUM
    # singular value (the smallest sigma, not 1/sigma).
    # Let's verify:
    from math import isclose

    # ord=-2 should equal min singular value
    assert isclose(ref_cpu, min_sv_cpu, rel_tol=1e-12), f"{ref_cpu} != {min_sv_cpu}"
    assert isclose(ref_gpu, min_sv_gpu, rel_tol=1e-12), f"{ref_gpu} != {min_sv_gpu}"
    print("Verified: ord=-2 == min(singular_values) ✓")
    print()


if __name__ == "__main__":
    main()
