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

"""Element-wise greater-than, GCU400-optimized (mirrors greater_equal.py).

The pointwise_dynamic path keeps small / broadcast / non-contiguous / non-float
inputs (launch-overhead floor there), while large contiguous same-dtype float
inputs dispatch to dedicated flat 1D and grid-stride (cap 48 CTA) kernels
(L600 probe, kernel mode do_bench):

* tensor & in-place share one kernel; out-of-place writes bool, in-place casts
  the result back into ``A`` (same semantics as ``gt_func(..., out0=A)``).
* comparisons run in the native input dtype (exact for same-dtype floats; the
  fp32 up-cast of the pointwise kernel only wastes ALU on the large tiers), and
  the scalar overload casts the Python scalar to ``A.dtype`` before comparing,
  which matches torch TensorIterator semantics.
* per-dtype DSM cap (max_dsm = 917504 B, compile-time enforced) bounds the
  tile size: 2-input tiles 2*itemsize*CHUNK <= 524288 B, scalar 1-input
  itemsize*CHUNK <= 524288 B (f16@2^18 / f32@2^17 tops).
* grid-stride kernels pin the grid at 48 persistent CTAs (one per SP) instead
  of cdiv grids that pay the per-wave tax at thousands of CTAs; EVEN_N drops
  the per-lane tail mask when N is a multiple of the chunk.

Probe-validated tiers (L600 kernel do_bench, r1/r2/r3 sweeps 2026-11-16):
  tensor/inplace n < 2**16    -> pointwise (launch-overhead floor)
  tensor/inplace 2**16..2**20 (f16/bf16) -> flat BLOCK=16384, nw=4
  tensor/inplace 2**20..2**22 (f16/bf16) -> flat BLOCK=65536, nw=4
  tensor/inplace 2**22..2**25 (f16/bf16) -> gs48 CHUNK=65536, nw=4, st=2
  tensor/inplace n >= 2**25 (f16/bf16)   -> gs48 CHUNK=131072, nw=4, st=2
  tensor/inplace 2**16..2**20 (f32) -> flat BLOCK=16384, nw=4
  tensor/inplace 2**20..2**21 (f32) -> flat BLOCK=65536, nw=4
  tensor/inplace 2**21..2**24 (f32) -> gs48 CHUNK=32768, nw=4, st=2
  tensor/inplace 2**24..2**28 (f32) -> gs48 CHUNK=65536, nw=4, st=2
  tensor/inplace n >= 2**28 (f32)   -> pointwise (2^30 probe: pointwise
                                        3.725 ms < gs48 3.794 / native 3.777)
  scalar n < 2**14 (contiguous float) -> single-CTA next_pow2 kernel
  scalar 2**14..2**20        -> flat BLOCK=16384, nw=4
  scalar 2**20..2**22        -> flat BLOCK=131072 (f16/bf16) / 65536 (f32)
  scalar 2**22..2**23 (f16/bf16) -> flat BLOCK=131072 (ties gs48, simpler)
  scalar 2**23..2**28 (f16/bf16) -> gs48 CHUNK=131072, nw=4, st=2
  scalar n >= 2**28 (f16/bf16)   -> flat BLOCK=262144 (r4 A/B: flat 1.0825
                                        vs gs48 1.0885 ms at 2^30)
  scalar 2**22..2**28 (f32)   -> gs48 CHUNK=65536, nw=4, st=2
  scalar n >= 2**28 (f32)     -> gs48 CHUNK=131072, nw=4, st=2

In-place overrides (gt_scalar_, full-width write-back profile, probed
2026-11-16 with same-session interleaved A/B, kernel do_bench w1000/r100;
the rows above were validated on the bool-out profile):
  scalar in-place n >= 2**22 (f32) -> gs48 CHUNK=65536 through the largest
      tiers: at 2^30 the 65536 chunk hits the pure-copy floor (2657.7 vs
      copy 2658.4 us) while the 131072 chunk pays 2665.2-2666.1 us (0/7);
      2^28/2^29 repeat 5/5. The >=2**28 CHUNK=131072 switch above is a
      bool-out-profile result and stays out-of-place only.
  scalar in-place n >= 2**28 (f16/bf16, n % 131072 == 0) -> interleaved gs48
      BLOCK=131072, NS=2 (2^30 f16 1329.6 vs flat 1332.8, bf16 1328.2 vs
      1332.8, 2^28 335.6 vs 337.7 us; flat-262144 loses every round).
      Odd sizes keep the flat 262144 arm; [2**23, 2**28) keeps gs48
      CHUNK=131072 (interleaved loses 5/5 at 2^26/2^27 there).
"""

import logging
import time

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_SCALAR_SMALL_N = 2**14
_LOW_N = 2**20
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_GS_CAP = 48
_F16_GS_N = 2**22
_F16_GS_BIG_N = 2**25
_F32_GS_N = 2**21
_F32_GS_BIG_N = 2**24
_SCALAR_F16_GS_N = 2**23
_SCALAR_F32_GS_N = 2**22
_SCALAR_F32_GS_BIG_N = 2**28


def _launch_with_retry(launch, attempts=4, delay=1.0):
    """Retry flaky cold compiles (P02 toolchain-segv family); re-raise on exhaust."""
    for i in range(attempts):
        try:
            return launch()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


@triton.jit
def _gt_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        x = tl.load(A + off)
        y = tl.load(B + off)
        tl.store(OUT + off, (x > y).to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        tl.store(OUT + off, (x > y).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_gs_kernel(
    A, B, OUT, N, CHUNK: tl.constexpr, CAP: tl.constexpr, EVEN_N: tl.constexpr
):
    pid = tl.program_id(0)
    for s in range(pid * CHUNK, N, CAP * CHUNK):
        if EVEN_N:
            off = s + tl.arange(0, CHUNK)
            x = tl.load(A + off)
            y = tl.load(B + off)
            tl.store(OUT + off, (x > y).to(OUT.dtype.element_ty))
        else:
            off = s + tl.arange(0, CHUNK)
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            y = tl.load(B + off, mask=m, other=0)
            tl.store(OUT + off, (x > y).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        x = tl.load(A + off)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_scalar_gs_kernel(
    A, OUT, N, Y, CHUNK: tl.constexpr, CAP: tl.constexpr, EVEN_N: tl.constexpr
):
    pid = tl.program_id(0)
    for s in range(pid * CHUNK, N, CAP * CHUNK):
        if EVEN_N:
            off = s + tl.arange(0, CHUNK)
            x = tl.load(A + off)
            r = x > Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            off = s + tl.arange(0, CHUNK)
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            r = x > Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_scalar_ilv_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, NS: tl.constexpr):
    """Interleaved grid-stride scalar compare (block_id = pid, pid+G, pid+2G, ...).

    Requires N % BLOCK == 0 (EXACT gate on the host side); the strided
    ``tl.range`` form with software pipelining (NS) beats the contiguous-chunk
    assignment at the in-place write-back wall.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = N // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        x = tl.load(A + off)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))


def _gt_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _BIG_N:
            # The pointwise path is the fastest tier for fp32 at the largest
            # shapes (L600 2^30 probe: pointwise 3.725 ms < gs48 3.794 ms,
            # native 3.777 ms); fp16/bf16 prefer the grid-stride kernels there.
            return False
        if n >= _F32_GS_N:
            if n > _INT32_SAFE_LIMIT:
                return False
            chunk = 65536 if n >= _F32_GS_BIG_N else 32768
            _launch_with_retry(
                lambda: _gt_cmp_gs_kernel[(_GS_CAP,)](
                    A,
                    B,
                    out,
                    n,
                    CHUNK=chunk,
                    CAP=_GS_CAP,
                    EVEN_N=(n % chunk == 0),
                    num_warps=4,
                    num_stages=2,
                )
            )
            return True
        block = 65536 if n >= _LOW_N else 16384
    else:
        if n >= _F16_GS_N:
            if n > _INT32_SAFE_LIMIT:
                return False
            chunk = 131072 if n >= _F16_GS_BIG_N else 65536
            _launch_with_retry(
                lambda: _gt_cmp_gs_kernel[(_GS_CAP,)](
                    A,
                    B,
                    out,
                    n,
                    CHUNK=chunk,
                    CAP=_GS_CAP,
                    EVEN_N=(n % chunk == 0),
                    num_warps=4,
                    num_stages=2,
                )
            )
            return True
        block = 65536 if n >= _LOW_N else 16384
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _gt_cmp_kernel[(triton.cdiv(n, block),)](
            A, B, out, n, BLOCK=block, EVEN_N=(n % block == 0), num_warps=4
        )
    )
    return True


def _gt_scalar_launch(A, out, n, scalar, inplace=False):
    """Try the scalar fast paths; True if launched, False -> pointwise.

    ``inplace`` (gt_scalar_) writes the 0/1 result back into ``A`` at full
    width, which shifts the large-tier winners relative to the bool-out
    profile; the overrides are documented in the module docstring.
    """
    if n >= _SCALAR_SMALL_N:
        if A.dtype == torch.float32:
            if n >= _SCALAR_F32_GS_N:
                if n > _INT32_SAFE_LIMIT:
                    return False
                # In-place probe (2026-11-16): the 65536 chunk stays optimal
                # through 2^30 (2657.7 vs 2665.2-2666.1 us, at the pure-copy
                # floor); the 131072 switch is a bool-out-profile result.
                if inplace:
                    chunk = 65536
                else:
                    chunk = 131072 if n >= _SCALAR_F32_GS_BIG_N else 65536
                _launch_with_retry(
                    lambda: _gt_cmp_scalar_gs_kernel[(_GS_CAP,)](
                        A,
                        out,
                        n,
                        scalar,
                        CHUNK=chunk,
                        CAP=_GS_CAP,
                        EVEN_N=(n % chunk == 0),
                        num_warps=4,
                        num_stages=2,
                    )
                )
                return True
            block = 65536 if n >= _LOW_N else 16384
        else:
            if n >= _BIG_N:
                # Probe r4 (interleaved A/B, 2026-11-16): the flat kernel keeps a
                # small edge over gs48 here on the bool-out profile (flat 1.0825
                # vs gs 1.0885 ms at 2^30), so the largest scalar shapes stay on
                # flat BLOCK=262144 out-of-place.
                if inplace and n % 131072 == 0 and n <= _INT32_SAFE_LIMIT - 131072:
                    # In-place probe (2026-11-16): the full-width write-back
                    # flips the largest tiers to the interleaved grid-stride
                    # form (2^30 f16 1329.6 vs 1332.8, bf16 1328.2 vs 1332.8,
                    # 2^28 335.6 vs 337.7 us; flat loses every round).
                    _launch_with_retry(
                        lambda: _gt_cmp_scalar_ilv_kernel[(_GS_CAP,)](
                            A,
                            out,
                            n,
                            scalar,
                            BLOCK=131072,
                            NS=2,
                            num_warps=4,
                            num_stages=2,
                        )
                    )
                    return True
                block = 262144
            elif n >= _SCALAR_F16_GS_N:
                if n > _INT32_SAFE_LIMIT:
                    return False
                chunk = 131072
                _launch_with_retry(
                    lambda: _gt_cmp_scalar_gs_kernel[(_GS_CAP,)](
                        A,
                        out,
                        n,
                        scalar,
                        CHUNK=chunk,
                        CAP=_GS_CAP,
                        EVEN_N=(n % chunk == 0),
                        num_warps=4,
                        num_stages=2,
                    )
                )
                return True
            else:
                block = 131072 if n >= _LOW_N else 16384
        if n > _INT32_SAFE_LIMIT - block:
            return False
        _launch_with_retry(
            lambda: _gt_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                A, out, n, scalar, BLOCK=block, EVEN_N=(n % block == 0), num_warps=4
            )
        )
        return True
    if n > 0:
        # For very small shapes (e.g. 64x64), pointwise_dynamic's dynamic dispatch and specialization overhead
        # exceeds that of a single-CTA flat kernel; switch to a next_pow2 block to cover in one shot
        block = max(32, triton.next_power_of_2(n))
        _launch_with_retry(
            lambda: _gt_cmp_scalar_kernel[(1,)](
                A, out, n, scalar, BLOCK=block, EVEN_N=False, num_warps=4
            )
        )
        return True
    return False


def gt(A, B):
    logger.debug("GEMS_ENFLAME GT")
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _gt_cmp_run(A, B, out):
            return out
    return gt_func(A, B)


def gt_scalar(A, B):
    logger.debug("GEMS_ENFLAME GT_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        out = torch.empty_like(A, dtype=torch.bool)
        if _gt_scalar_launch(A, out, A.numel(), float(B)):
            return out
    return gt_func_scalar(A, B)


def gt_tensor_(A, B):
    logger.debug("GEMS_ENFLAME GT_TENSOR_")
    if A.device != B.device:
        B = B.to(A.device)
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        if _gt_cmp_run(A, B, A):
            return A
    return gt_func(A, B, out0=A)


def gt_scalar_(A, B):
    logger.debug("GEMS_ENFLAME GT_SCALAR_ ")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        if _gt_scalar_launch(A, A, A.numel(), float(B), inplace=True):
            return A
    return gt_func_scalar(A, B, out0=A)
