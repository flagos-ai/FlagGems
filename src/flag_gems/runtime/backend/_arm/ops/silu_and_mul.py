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
ARM CPU fused silu_and_mul — TLE NEON SWIGLU for decode, ATen for prefill.

For decode (M=1): TLE cpu_swiglu (NEON fast exp + fused silu*mul) via a
@triton.jit kernel (no ctypes — goes through the create_cpu_swiglu TLE path).
For prefill (M>1): ATen F.silu(x1) * x2 (fallback).

Benchmarks (CIX P1 CD8180, BF16, OMP=8):
  N=6144 decode:  ATen ~76μs → TLE SWIGLU ~33μs (2.3x speedup)
  28 layers × savings = 1.2ms/tok
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import swiglu as _tle_swiglu

# None = not yet tried, True = TLE path works, False = fall back to ATen.
_TLE_SWIGLU_OK = None


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, N: tl.constexpr):
    # One coarse TLE op = the whole SWIGLU (silu(gate) * up over N elements),
    # OMP-parallelized inside the C runtime → 1 kernel launch.
    _tle_swiglu(gate_ptr, up_ptr, out_ptr, N)


def arm_silu_and_mul(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """ARM CPU fused silu_and_mul: silu(x1) * x2.

    Decode (1D/2D with M=1): TLE NEON SWIGLU (2.3x faster than ATen).
    Otherwise: ATen fallback.
    """
    global _TLE_SWIGLU_OK
    # Decode path: contiguous BF16, single row.
    if (
        _TLE_SWIGLU_OK is not False
        and x1.dtype == torch.bfloat16
        and x1.is_contiguous()
        and x2.is_contiguous()
        and x1.numel() == x1.shape[-1]
    ):  # M=1
        try:
            N = x1.numel()
            out = torch.empty_like(x1)
            _swiglu_kernel[(1,)](x1, x2, out, N=N)
            _TLE_SWIGLU_OK = True
            return out
        except Exception:
            _TLE_SWIGLU_OK = False
    return F.silu(x1) * x2


def arm_silu_and_mul_out(
    x1: torch.Tensor, x2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """ARM CPU fused silu_and_mul with pre-allocated output."""
    result = arm_silu_and_mul(x1, x2)
    out.copy_(result)
    return out
