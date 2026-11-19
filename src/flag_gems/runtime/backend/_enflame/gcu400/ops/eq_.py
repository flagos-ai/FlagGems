# Copyright 2026, The FlagOS Contributors.
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
#
"""Element-wise in-place equality (eq_ / eq_scalar_), GCU400-optimized.

The previous path ran every shape through the global ``pointwise_dynamic``
template, whose ENFLAME ``CodeGenConfig`` caps the launch grid at
``(12, 1, 1)`` with 4096-element tiles: every shape beyond ~2**24 elements
serialized onto 12 CTAs (bench core baseline 2026-11-13: eq_ 2**30
12.56-12.82 ms, ~0.5-0.9 TB/s effective; eq_scalar_ 11.08-11.45 ms) and
paid the fp32 up-cast compare on top.

This module mirrors the out-of-place ``.eq`` and the in-place
``.greater_equal_`` siblings: large contiguous same-dtype float inputs are
dispatched to the flat / strided 1D kernels reused from ``.eq``, passing
``A`` as ``OUT`` -- the kernels store ``r.to(OUT.dtype.element_ty)``, so the
bool result lands in ``A`` cast to A's dtype, matching torch ``eq_``
semantics (0.0/1.0 written into the float self tensor).

* comparisons run in the native input dtype (bit-exact for same-dtype
  floats; the fp32 up-cast is an injective map, so ``==`` is preserved);
* the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator semantics (the fp32-compare
  template deviates from torch on fp16 round-down scalars such as 0.1);
* small (< 2**16) / non-contiguous / broadcast / non-float inputs fall back
  to the ``pointwise_dynamic`` kernels from ``.eq``.

In-place-validated tiers (do_bench kernel-mode median, w1000/r100,
2026-11-13; the in-place store is itemsize-wide vs the 1-byte bool store
out-of-place, which shifts the optimum toward the 48-CTA persistent strided
form on the larger shapes):

  tensor n < 2**16              -> generic pointwise (launch-overhead floor)
  tensor f32 n >= 2**16         -> strided 48xB=65536 nw4 (best at 2^22..2^30)
  tensor f16/bf16 < 2**22       -> flat BLOCK=65536 nw4
  tensor f16/bf16 <= 48*2**17   -> flat BLOCK=131072 nw4 (grid-exact, <=48 tiles)
  tensor f16/bf16 larger        -> strided 48xB=131072 nw4 (beats flat & loop)
  scalar n < 2**16              -> flat BLOCK=next_pow2(n) nw4
  scalar 2**16..2**22           -> flat BLOCK=65536 nw4
  scalar f16/bf16 2**22..2**28  -> flat BLOCK=131072 nw4
  scalar f16/bf16 n >= 2**28    -> flat BLOCK=262144 nw4 (roofline ~3 TB/s)
  scalar f32 n >= 2**22         -> strided 48xB=65536 nw4
"""

import logging

import torch
import triton

from .eq import (
    _BIG_N,
    _FLOAT_DTYPES,
    _INT32_SAFE_LIMIT,
    _MID_N,
    _NP2,
    _NPROGS,
    _SMALL_N,
    _eq_cmp_kernel,
    _eq_cmp_scalar_kernel,
    _eq_cmp_scalar_strided_kernel,
    _eq_cmp_strided_kernel,
    _launch_with_retry,
    eq_func,
    eq_func_scalar,
)

logger = logging.getLogger(__name__)

# 48 persistent CTAs x 131072-element tiles: at or below this element count
# the strided form is identical to a grid-exact flat launch, so the plain
# flat kernel (no loop machinery) is used instead.
_F16_STRIDED_MIN_N = _NPROGS * 131072


def _eq_inplace_run(A, B):
    """Launch the tuned in-place comparison kernel; False falls back."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n > _INT32_SAFE_LIMIT - 65536:
            return False
        _launch_with_retry(
            lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                A, B, A, n, BLOCK=65536, num_warps=4
            )
        )
        return True
    if n > _F16_STRIDED_MIN_N:
        if n > _INT32_SAFE_LIMIT - 131072:
            return False
        _launch_with_retry(
            lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                A, B, A, n, BLOCK=131072, num_warps=4
            )
        )
        return True
    block = 131072 if n >= _MID_N else 65536
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _eq_cmp_kernel[(triton.cdiv(n, block),)](
            A, B, A, n, BLOCK=block, num_warps=4
        )
    )
    return True


def eq_(A, B):
    logger.debug("GEMS_ENFLAME EQ_")
    if A.device != B.device:
        B = B.to(A.device)
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        if _eq_inplace_run(A, B):
            return A
    return eq_func(A, B, out0=A)


def eq_scalar_(A, B):
    logger.debug("GEMS_ENFLAME EQ_SCALAR_")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        try:
            y = float(B)
        except (OverflowError, ValueError):
            return eq_func_scalar(A, B, out0=A)
        n = A.numel()
        if n > 0:
            if n >= _MID_N:
                if A.dtype == torch.float32:
                    if n > _INT32_SAFE_LIMIT - 65536:
                        return eq_func_scalar(A, B, out0=A)
                    _launch_with_retry(
                        lambda: _eq_cmp_scalar_strided_kernel[(_NPROGS,)](
                            A, A, n, y, BLOCK=65536, num_warps=4
                        )
                    )
                    return A
                # DSM Required counts loads only (probe-verified in-place:
                # BLOCK=262144 compiles, Required = itemsize * BLOCK = 512 KB)
                block = 262144 if n >= _BIG_N else 131072
            elif n >= _SMALL_N:
                block = 65536
            else:
                # shape-matched tile for launch-floor shapes: a fixed 16384-wide
                # vector wastes 4x lanes on (64,64)-class fp32 inputs (+1.5us)
                block = max(256, min(16384, _NP2(n)))
            if n <= _INT32_SAFE_LIMIT - block:
                _launch_with_retry(
                    lambda: _eq_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                        A, A, n, y, BLOCK=block, num_warps=4
                    )
                )
                return A
    return eq_func_scalar(A, B, out0=A)
