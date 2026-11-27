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

"""Element-wise in-place not-equal, GCU400-optimized flat 1D dispatch.

Round 1 (2026-11-13) replaced the public ``pointwise_dynamic`` delegation
(12-CTA codegen + fp32 up-cast, 0.18-0.37x on the bench-core mid/large
shapes) with flat 1D kernels + loop tiers. Round 2 (this file) adopts the
``ne`` out-of-place round-2 findings, re-probed for the in-place store
width (OUT aliases A, so the store carries A.dtype: 2/4-byte writes vs
the 1-byte bool store of the out-of-place sibling):

* nw2 replaces nw4 on the wide flat streaming bands (16-bit B65536,
  f32 B32768 [2^22,2^28)): 2^24 f16 43.0 -> 40.4us (-6.1%, 1.01x
  native), 2^24 f32 77.1 -> 73.1us (-5.1%, 1.04x native), 2^26 f32
  285.9 -> 271.5us (-5.0%).
* EVEN (n % BLOCK == 0) mask-free arm on n < 2^28: removes the
  per-element predicate tax on divisible shapes (family judgment from
  the eq/ne sibling tasks: EVEN gains at n >= 2^28 reverse in the
  official bench, so the giant band stays masked).
* 16-bit loop tiers removed: flat 65536 nw2 masked beats
  loop(1048576,131072,nw4,st2) at n = 2^30 (2158 -> 2122us) and the
  odd-n masked arm beats the old loop tier at 2^24+4k (49.3 -> 42.9us).
  The single clean band is [2^20, guard) flat 65536 nw2 (2^20/2^21:
  -7.5%/-13.4% vs the old flat16384 nw4 tier).
* f32 keeps the loop tier only for n >= 2^28 (flatM32768nw2 is
  WITHIN-NOISE equal there: 4283.5 vs 4284.0us).
* scalar: EVEN arms on [2^22,2^28) (B131072 f16 / B65536 f32, odd n
  keeps the identical masked arm - no odd-n pathology in-place,
  probed 2^24+4k f32: masked B65536 48.1us best); giant band masked
  B262144 nw4 (f16, -0.6% vs B131072 at 2^30, no odd-n cliff at
  2^28+4k) / B131072 nw4 (f32, -0.6% vs B65536 at 2^30).
* comparisons run in the native input dtype; the scalar is cast to
  ``A.dtype`` before comparing (torch TensorIterator semantics for
  fp16 round-down scalars). Fallback keeps the backend-local 48-CTA
  ``ne_func``/``ne_func_scalar`` (not the 12-CTA public one) for
  small / non-contiguous / broadcast / mixed-dtype / bool inputs.
* flat BLOCK bounds: compile-time DSM budget (num_load_tiles *
  itemsize * BLOCK <= 917504 B, loads-only) and grid.x <= 65535.
"""

import logging

import torch
import triton
import triton.language as tl

from .ne import ne_func, ne_func_scalar

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_SMALL_MID_N = 2**19
_WIDE_N = 2**20
_MID_N = 2**22
_HUGE_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_GRID_LIMIT = 65535


@triton.jit
def _ne_inplace_kernel(A, B, OUT, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(A + off)
        y = tl.load(B + off)
        tl.store(OUT + off, (x != y).to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        tl.store(OUT + off, (x != y).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ne_inplace_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x != y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ne_inplace_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(A + off)
        tl.store(OUT + off, (x != Y.to(x.dtype)).to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        tl.store(OUT + off, (x != Y.to(x.dtype)).to(OUT.dtype.element_ty), mask=m)


def _even(n, block):
    # Family judgment (eq/ne sibling tasks): EVEN gains at n >= 2^28
    # reverse in the official bench, so the giant band stays masked.
    return n < _HUGE_N and n % block == 0


def _launch_flat(A, B, n, block, nw):
    if n > _INT32_SAFE_LIMIT - block:
        return False
    grid = triton.cdiv(n, block)
    if grid > _GRID_LIMIT:
        return False
    _ne_inplace_kernel[(grid,)](
        A, B, A, n, BLOCK=block, EVEN=_even(n, block), num_warps=nw
    )
    return True


def _launch_loop(A, B, n, cov, chunk, nw, st):
    if n > _INT32_SAFE_LIMIT - cov:
        return False
    _ne_inplace_loop_kernel[(triton.cdiv(n, cov),)](
        A, B, A, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
    )
    return True


def _ne__cmp_run(A, B):
    """Launch the tuned in-place comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if A.dtype == torch.float32:
        if n >= _HUGE_N:
            return _launch_loop(A, B, n, 1048576, 65536, 8, 1)
        if n >= _MID_N:
            return _launch_flat(A, B, n, 32768, 2)
        if n >= _SMALL_MID_N:
            return _launch_flat(A, B, n, 32768, 4)
        if n >= _SMALL_N:
            return _launch_flat(A, B, n, 8192, 4)
        return False
    if n >= _WIDE_N:
        return _launch_flat(A, B, n, 65536, 2)
    if n >= _SMALL_MID_N:
        return _launch_flat(A, B, n, 16384, 4)
    return _launch_flat(A, B, n, 8192, 4)


def ne_(A, B):
    logger.debug("GEMS_ENFLAME NE_")
    if (
        A.numel() >= _SMALL_N
        and A.device == B.device
        and A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        if _ne__cmp_run(A, B):
            return A
    ne_func(A, B, out0=A)
    return A


def _ne_scalar_tier(dtype, n):
    """(BLOCK, num_warps) for the scalar overload, None -> generic path."""
    if dtype == torch.float32:
        if n >= _HUGE_N:
            return (131072, 4)
        if n >= _MID_N:
            return (65536, 4)
        if n >= _SMALL_MID_N:
            return (32768, 4)
        return (min(8192, max(1024, triton.next_power_of_2(n))), 4)
    if n >= _HUGE_N:
        return (262144, 4)
    if n >= _MID_N:
        return (131072, 4)
    return (min(16384, max(1024, triton.next_power_of_2(n))), 4)


def ne_scalar_(A, B):
    logger.debug("GEMS_ENFLAME NE__SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n >= _SMALL_N:
            block, nw = _ne_scalar_tier(A.dtype, n)
            if n <= _INT32_SAFE_LIMIT - block and triton.cdiv(n, block) <= _GRID_LIMIT:
                _ne_inplace_scalar_kernel[(triton.cdiv(n, block),)](
                    A,
                    A,
                    n,
                    float(B),
                    BLOCK=block,
                    EVEN=_even(n, block),
                    num_warps=nw,
                )
                return A
    ne_func_scalar(A, B, out0=A)
    return A
