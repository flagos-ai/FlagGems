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

In-place-validated tiers (do_bench kernel-mode median, w1000/r100; the
2026-11-13 tiers plus the 2026-11-18 re-tune, which moves the tensor arms
and the f16/bf16 scalar mid band to DSM dual-program (loads <= 256KB per
program) x num_warps=2 forms -- an interaction win: each single-variable
change alone loses. The in-place store is itemsize-wide vs the 1-byte bool
store out-of-place, which keeps the 48-CTA persistent strided form dominant
on the large shapes):

  tensor n < 2**16              -> generic pointwise (launch-overhead floor)
  tensor f32 [2**16, 2**21)     -> flat BLOCK=32768 nw2 (2^17 0.995x / 2^18
                                   1.074x / 2^20 0.995x vs old strided
                                   0.67-0.87x)
  tensor f32 n >= 2**21         -> strided 48xB=32768 nw2 (2^21 1.001x /
                                   2^22 1.011x / 2^24 1.043x / 2^26 1.058x;
                                   2^28-2^30 tie the old 64k/nw4 form at
                                   the ~3.0TB/s roofline)
  tensor f16/bf16 < 2**22       -> flat BLOCK=65536 nw4 (2^20 1.290x, beats
                                   the new strided form and flat-131k)
  tensor f16/bf16 n >= 2**22   -> strided 48xB=65536 nw2 (2^22 1.002x /
                                   2^24 1.041x / 2^26 1.060x / 2^28 1.063x /
                                   2^30 1.055x -- beats flat-131k and the
                                   old strided-131k/nw4 on every point)
  scalar n < 2**16              -> flat BLOCK=next_pow2(n) nw4
  scalar 2**16..2**22           -> flat BLOCK=65536 nw4
  scalar f16/bf16 2**22..2**28  -> strided 48xB=65536 nw2 (2^22 1.078x /
                                   2^24 1.072x / 2^26 1.077x vs flat-131k
                                   1.043-1.067x)
  scalar f16/bf16 n >= 2**28    -> flat BLOCK=262144 nw4 (roofline ~3 TB/s;
                                   strided-64k ties within noise, prod kept)
  scalar f32 n >= 2**22         -> strided 48xB=65536 nw4 (unchanged: best at
                                   every band point incl. 2^26/2^30)

The in-place tiers keep the masked arm everywhere (``EVEN=False``): the
2026-11-16 re-measure showed the maskless EVEN arm that helps the
out-of-place kernels regresses in-place mid shapes (eq_ f16/bf16 2^24
+4.1..4.8%, eq_scalar_ f16/bf16 2^24 +4.4..9.9% -- the store writes back
through the same pointer the loads read, and dropping the store predicate
degrades the emitted access pattern), so the in-place tiers stay on the
masked form the 2026-11-13 tuning was measured on.
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

# fp32 tensor boundary between the flat-32k launch and the persistent
# strided-48x32k form (2026-11-18 probes: flat wins 2^16..2^20, strided wins
# from 2^21 up).
_F32_MID_N = 2**21


def _eq_inplace_run(A, B):
    """Launch the tuned in-place comparison kernel; False falls back."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        # 2026-11-18 re-tune: DSM dual-program (2 loads x 4B x 32768 = 256KB
        # <= 448KB/program) combined with num_warps=2 wins the whole fp32
        # range over the previous strided 48x65536/nw4 (524KB, single
        # program/SP). Flat 32k below 2^21 (launch-floor band: 2^17 0.995x /
        # 2^18 1.074x / 2^20 0.995x vs old 0.67-0.87x; flat wins there because
        # the strided loop only spans 2-32 tiles), persistent strided above
        # (2^21 1.001x / 2^22 1.011x / 2^24 1.043x / 2^26 1.058x; 2^28-2^30
        # tie with the old form at the ~3.0TB/s roofline).
        block = 32768
        if n > _INT32_SAFE_LIMIT - block:
            return False
        if n >= _F32_MID_N:
            _launch_with_retry(
                lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                    A, B, A, n, BLOCK=block, EVEN=False, num_warps=2
                )
            )
        else:
            _launch_with_retry(
                lambda: _eq_cmp_kernel[(triton.cdiv(n, block),)](
                    A, B, A, n, BLOCK=block, EVEN=False, num_warps=2
                )
            )
        return True
    if n >= _MID_N:
        # f16/bf16 >= 2^22: strided 48x65536 nw2 -- DSM dual-program
        # (2 x 2B x 65536 = 256KB) x nw2. Beats the previous flat-131072 /
        # strided-48x131072/nw4 tiers on every band point (2^22 1.002x /
        # 2^24 1.041x / 2^26 1.060x / 2^28 1.063x / 2^30 1.055x); the win is
        # the interaction -- single-variable probes all lose (64k+nw4
        # 0.929x, 131k+nw2 1.000x, flat131k 0.939x, st1/st2/st3 neutral).
        if n > _INT32_SAFE_LIMIT - 65536:
            return False
        _launch_with_retry(
            lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                A, B, A, n, BLOCK=65536, EVEN=False, num_warps=2
            )
        )
        return True
    # [2^16, 2^22) keeps the 2026-11-13 flat launch (2^20: 1.290x, beats
    # strided-64k-nw2 1.196x and flat-131k 1.192x).
    block = 65536
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _eq_cmp_kernel[(triton.cdiv(n, block),)](
            A, B, A, n, BLOCK=block, EVEN=False, num_warps=4
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
                            A, A, n, y, BLOCK=65536, EVEN=False, num_warps=4
                        )
                    )
                    return A
                # f16/bf16: [2^22, 2^28) strided 48x65536 nw2 (2026-11-18:
                # beats flat-131072/nw4 on the whole band -- 2^22 1.078x /
                # 2^24 1.072x / 2^26 1.077x vs flat 1.043-1.067x; single load
                # x 2B x 65536 = 128KB, dual-program 48-way); the giant band
                # keeps the flat 262144/nw4 launch (2^28 337.56 vs 336.77us,
                # 2^30 1331.9 vs 1333.3us -- within noise, prod tier kept).
                # DSM Required counts loads only (probe-verified in-place:
                # BLOCK=262144 compiles, Required = itemsize * BLOCK = 512 KB)
                if n < _BIG_N:
                    if n > _INT32_SAFE_LIMIT - 65536:
                        return eq_func_scalar(A, B, out0=A)
                    _launch_with_retry(
                        lambda: _eq_cmp_scalar_strided_kernel[(_NPROGS,)](
                            A, A, n, y, BLOCK=65536, EVEN=False, num_warps=2
                        )
                    )
                    return A
                block = 262144
            elif n >= _SMALL_N:
                block = 65536
            else:
                # shape-matched tile for launch-floor shapes: a fixed 16384-wide
                # vector wastes 4x lanes on (64,64)-class fp32 inputs (+1.5us)
                block = max(256, min(16384, _NP2(n)))
            if n <= _INT32_SAFE_LIMIT - block:
                _launch_with_retry(
                    lambda: _eq_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                        A, A, n, y, BLOCK=block, EVEN=False, num_warps=4
                    )
                )
                return A
    return eq_func_scalar(A, B, out0=A)
