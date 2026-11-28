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

"""Element-wise greater-or-equal, GCU400-optimized (sibling of ``gt``).

Small / broadcast / non-contiguous / non-float inputs stay on the
``pointwise_dynamic`` fallback, while contiguous same-dtype float inputs
dispatch to dedicated flat 1D and grid-stride (cap-48 CTA) kernels
(order-balanced probe 2026-11-16, kernel-mode do_bench median):

* comparisons run in the native input dtype (exact for same-dtype floats);
  the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator wrapped-number semantics.
* grid-stride kernels pin the grid at 48 persistent CTAs (one per SP); the
  old coverage-loop tiers paid the per-wave tax at cdiv grids (2048 CTAs at
  2**30) and broke DRAM bank interleave with per-CTA contiguous spans.
* EVEN_N drops the per-lane tail mask when N is a multiple of the chunk
  (ragged N keeps the masked arm: 2^24+123 probe shows no mask tax).
* per-dtype DSM cap (max_dsm = 917504 B) bounds the chunk size: dual-load
  fp32 tops at CHUNK=65536 (2*4*65536 = 512 KB), scalar fp32 flat at
  BLOCK=131072.

Probe-validated tiers (L600 kernel do_bench, 2026-11-16):
  tensor    n <= 4096              -> single-CTA next_pow2 kernel, nw=2
  tensor    4096..2**16            -> pointwise (launch-overhead floor)
  tensor    2**16..2**20 (f16/bf16)-> flat BLOCK=16384, nw=4
  tensor    2**20..2**22 (f16/bf16)-> flat BLOCK=65536, nw=4
  tensor    2**22..2**25 (f16/bf16)-> gs48 CHUNK=65536, nw=2, st=3
  tensor    n >= 2**25 (f16/bf16)  -> gs48 CHUNK=131072, nw=4, st=2
  tensor    2**16..2**20 (f32)     -> flat BLOCK=16384, nw=4
  tensor    2**20..2**21 (f32)     -> flat BLOCK=65536, nw=4
  tensor    2**21..2**24 (f32)     -> gs48 CHUNK=32768, nw=4, st=2
  tensor    2**24..2**28 (f32)     -> gs48 CHUNK=65536, nw=2, st=3
  tensor    n >= 2**28 (f32)       -> gs48 CHUNK=16384, nw=1, st=2
  scalar    n < 2**20              -> next_pow2 tile (<= 16384), nw=4
  scalar    2**20..2**23 (f16/bf16)-> flat BLOCK=131072, nw=4
  scalar    n >= 2**23 (f16/bf16)  -> gs48 CHUNK=131072, nw=4, st=2
  scalar    2**20..2**22 (f32)     -> flat BLOCK=65536, nw=4
  scalar    n >= 2**22 (f32)       -> gs48 CHUNK=65536, nw=4, st=2

Order-balanced A/B (median, vs the previous tiers at the bench core shapes):
2^24 tensor f16 40.8 -> 36.9 us, f32 70.0 -> 65.0 us; 2^30 tensor f16
2034.9 -> 1981.8 us, bf16 2007.5 -> 1937.8 us, f32 (pointwise) 3806.6 ->
3771.4 us; 2^24 scalar f16 24.1 -> 23.4 us, f32 38.8 -> 37.3 us; 2^30
scalar f32 2044.7 -> 1965.4 us (flat 8192-CTA DDR-contention fix).
"""

import logging
import time

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TINY_N = 4096
_SMALL_N = 2**16
_SCALAR_TILE_N = 2**20
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
def ge_func(x, y):
    return x.to(tl.float32) >= y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


@triton.jit
def _ge_flat_kernel(A, B, OUT, N, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        x = tl.load(A + off)
        y = tl.load(B + off)
        tl.store(OUT + off, (x >= y).to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        tl.store(OUT + off, (x >= y).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_gs_kernel(
    A, B, OUT, N, CHUNK: tl.constexpr, CAP: tl.constexpr, EVEN_N: tl.constexpr
):
    pid = tl.program_id(0)
    for s in range(pid * CHUNK, N, CAP * CHUNK):
        off = s + tl.arange(0, CHUNK)
        if EVEN_N:
            x = tl.load(A + off)
            y = tl.load(B + off)
            tl.store(OUT + off, (x >= y).to(OUT.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            y = tl.load(B + off, mask=m, other=0)
            tl.store(OUT + off, (x >= y).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_flat_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        x = tl.load(A + off)
        r = x >= Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        r = x >= Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_gs_scalar_kernel(
    A, OUT, N, Y, CHUNK: tl.constexpr, CAP: tl.constexpr, EVEN_N: tl.constexpr
):
    pid = tl.program_id(0)
    for s in range(pid * CHUNK, N, CAP * CHUNK):
        off = s + tl.arange(0, CHUNK)
        if EVEN_N:
            x = tl.load(A + off)
            r = x >= Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            r = x >= Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _ge_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n <= 0:
        return False
    if n <= _TINY_N:
        # Launch-floor shapes: one CTA with a next_pow2 tile beats the
        # pointwise dispatch machinery (f32 3.6 -> 3.3 us at (64, 64)).
        block = max(32, triton.next_power_of_2(n))
        _launch_with_retry(
            lambda: _ge_flat_kernel[(1,)](
                A, B, out, n, BLOCK=block, EVEN_N=(n % block == 0), num_warps=2
            )
        )
        return True
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _F32_GS_N:
            if n >= _BIG_N:
                chunk, nw, st = 16384, 1, 2
            elif n >= _F32_GS_BIG_N:
                chunk, nw, st = 65536, 2, 3
            else:
                chunk, nw, st = 32768, 4, 2
            if n > _INT32_SAFE_LIMIT - chunk:
                return False
            _launch_with_retry(
                lambda: _ge_gs_kernel[(_GS_CAP,)](
                    A,
                    B,
                    out,
                    n,
                    CHUNK=chunk,
                    CAP=_GS_CAP,
                    EVEN_N=(n % chunk == 0),
                    num_warps=nw,
                    num_stages=st,
                )
            )
            return True
        block = 65536 if n >= _LOW_N else 16384
    else:
        if n >= _F16_GS_N:
            if n >= _F16_GS_BIG_N:
                chunk, nw, st = 131072, 4, 2
            else:
                chunk, nw, st = 65536, 2, 3
            if n > _INT32_SAFE_LIMIT - chunk:
                return False
            _launch_with_retry(
                lambda: _ge_gs_kernel[(_GS_CAP,)](
                    A,
                    B,
                    out,
                    n,
                    CHUNK=chunk,
                    CAP=_GS_CAP,
                    EVEN_N=(n % chunk == 0),
                    num_warps=nw,
                    num_stages=st,
                )
            )
            return True
        block = 65536 if n >= _LOW_N else 16384
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _ge_flat_kernel[(triton.cdiv(n, block),)](
            A, B, out, n, BLOCK=block, EVEN_N=(n % block == 0), num_warps=4
        )
    )
    return True


def _ge_scalar_launch(A, out, n, scalar):
    """Try the scalar fast paths; True if launched, False -> pointwise."""
    if n <= 0:
        return False
    if n < _SCALAR_TILE_N:
        # shape-matched tile for launch-floor shapes: a fixed 16384-wide
        # vector wastes lanes on (64,64)-class inputs (14.0us -> 3.2us)
        block = max(256, min(16384, triton.next_power_of_2(n)))
        _launch_with_retry(
            lambda: _ge_flat_scalar_kernel[(triton.cdiv(n, block),)](
                A, out, n, scalar, BLOCK=block, EVEN_N=(n % block == 0), num_warps=4
            )
        )
        return True
    if A.dtype == torch.float32:
        if n >= _SCALAR_F32_GS_N:
            chunk, nw, st = 65536, 4, 2
            if n > _INT32_SAFE_LIMIT - chunk:
                return False
            _launch_with_retry(
                lambda: _ge_gs_scalar_kernel[(_GS_CAP,)](
                    A,
                    out,
                    n,
                    scalar,
                    CHUNK=chunk,
                    CAP=_GS_CAP,
                    EVEN_N=(n % chunk == 0),
                    num_warps=nw,
                    num_stages=st,
                )
            )
            return True
        block = 65536 if n >= _LOW_N else 16384
    else:
        if n >= _SCALAR_F16_GS_N:
            chunk, nw, st = 131072, 4, 2
            if n > _INT32_SAFE_LIMIT - chunk:
                return False
            _launch_with_retry(
                lambda: _ge_gs_scalar_kernel[(_GS_CAP,)](
                    A,
                    out,
                    n,
                    scalar,
                    CHUNK=chunk,
                    CAP=_GS_CAP,
                    EVEN_N=(n % chunk == 0),
                    num_warps=nw,
                    num_stages=st,
                )
            )
            return True
        block = 131072 if n >= _LOW_N else 16384
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _ge_flat_scalar_kernel[(triton.cdiv(n, block),)](
            A, out, n, scalar, BLOCK=block, EVEN_N=(n % block == 0), num_warps=4
        )
    )
    return True


def ge(A, B):
    logger.debug("GEMS_ENFLAME GE")
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _ge_cmp_run(A, B, out):
            return out
    return ge_func(A, B)


def ge_scalar(A, B):
    logger.debug("GEMS_ENFLAME GE_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        out = torch.empty_like(A, dtype=torch.bool)
        if _ge_scalar_launch(A, out, A.numel(), float(B)):
            return out
    return ge_func_scalar(A, B)
