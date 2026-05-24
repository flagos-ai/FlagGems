"""
ARM CPU fused silu_and_mul — TLE NEON SWIGLU for decode, ATen for prefill.

For decode (M=1): TLE cpu_swiglu (NEON fast exp + fused silu*mul) = 33μs
For prefill (M>1): ATen F.silu(x1) * x2 (fallback)

Benchmarks (CIX P1 CD8180, BF16, OMP=8):
  N=6144 decode:  ATen ~76μs → TLE SWIGLU ~33μs (2.3x speedup)
  28 layers × savings = 1.2ms/tok
"""

import os
import torch
import torch.nn.functional as F

_TLE_SWIGLU_AVAILABLE = None


def _check_tle_swiglu():
    global _TLE_SWIGLU_AVAILABLE
    if _TLE_SWIGLU_AVAILABLE is not None:
        return _TLE_SWIGLU_AVAILABLE
    try:
        import ctypes
        import pathlib
        import triton
        so_path = pathlib.Path(triton.__file__).parent / "_C" / "libTritonCPURuntime.so"
        if so_path.exists():
            lib = ctypes.CDLL(str(so_path))
            lib.swiglu_bf16.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
            ]
            _TLE_SWIGLU_AVAILABLE = lib
            return lib
    except Exception:
        pass
    _TLE_SWIGLU_AVAILABLE = False
    return False


def arm_silu_and_mul(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """ARM CPU fused silu_and_mul: silu(x1) * x2.

    Decode (1D/2D with M=1): TLE NEON SWIGLU (2.3x faster than ATen).
    Otherwise: ATen fallback.
    """
    # Decode path: contiguous BF16, small batch
    if (x1.dtype == torch.bfloat16
            and x1.is_contiguous() and x2.is_contiguous()
            and x1.numel() == x1.shape[-1]):  # M=1
        lib = _check_tle_swiglu()
        if lib:
            N = x1.numel()
            out = torch.empty_like(x1)
            lib.swiglu_bf16(
                x1.data_ptr(), x2.data_ptr(), out.data_ptr(), N)
            return out
    return F.silu(x1) * x2


def arm_silu_and_mul_out(
    x1: torch.Tensor, x2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """ARM CPU fused silu_and_mul with pre-allocated output."""
    result = arm_silu_and_mul(x1, x2)
    out.copy_(result)
    return out
