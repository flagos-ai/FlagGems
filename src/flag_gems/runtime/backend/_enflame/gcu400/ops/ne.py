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

"""Element-wise not-equal, GCU400-optimized flat 1D dispatch.

The generic ``pointwise_dynamic`` path (bptr tile 65536, grid capped at 48,
grid-stride loop, fp32 up-cast compare) leaves the medium/large shapes at
0.74-0.9x of torch native (bench core, L600 kernel mode: ne_scalar 0.75x at
2^24). Same-shaped comparison ops were already fixed this way
(greater_equal / copysign); ``ne`` reuses the probe-validated flat tiers:

* tensor & tensor:   n < 2^16 generic; 2^16..2^22 flat BLOCK=16384;
                     2^22..2^28 f16/bf16 flat BLOCK=131072, f32 loop
                     COV=262144/CHUNK=65536 nw8; >= 2^28 f16/bf16 loop
                     COV=524288/CHUNK=131072 nw4 st2. f32 >= 2^26 falls
                     back to the 48-CTA generic path which is already
                     optimal there (probe: 0.96-0.99x vs native).
* scalar overload:   f16/bf16 >= 2^22 flat BLOCK=262144 (single load
                     stream); f32 2^22..2^26 flat BLOCK=131072, larger
                     shapes back to generic (probe 0.96-1.03x).
* comparisons run in the native input dtype; the scalar is cast to
  ``A.dtype`` before comparing, matching torch TensorIterator semantics
  (the generic fp32-compare kernel deviates from torch on fp16
  round-down scalars, same trap as the gt/lt/le/ge family).
* flat BLOCK bounds come from the compile-time DSM budget
  (num_load_tiles * itemsize * BLOCK <= 917504 B) and grid.x <= 65535.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


@triton.jit
def _ne_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x != y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ne_cmp_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x != y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ne_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x != Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_MID_N = 2**22
_LATE_N = 2**26
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_GRID_LIMIT = 65535


def _ne_flat_launch(A, B, out, n, block, nw):
    if n > _INT32_SAFE_LIMIT - block:
        return False
    grid = triton.cdiv(n, block)
    if grid > _GRID_LIMIT:
        return False
    _ne_cmp_kernel[(grid,)](A, B, out, n, BLOCK=block, num_warps=nw)
    return True


def _ne_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _LATE_N:
            return False
        if n < _MID_N:
            return _ne_flat_launch(A, B, out, n, 16384, 4)
        cov, chunk, nw, st = 262144, 65536, 8, 1
        if n > _INT32_SAFE_LIMIT - cov:
            return False
        _ne_cmp_loop_kernel[(triton.cdiv(n, cov),)](
            A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
        )
        return True
    if n >= _BIG_N:
        cov, chunk, nw, st = 524288, 131072, 4, 2
        if n > _INT32_SAFE_LIMIT - cov:
            return False
        _ne_cmp_loop_kernel[(triton.cdiv(n, cov),)](
            A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
        )
        return True
    block = 131072 if n >= _MID_N else 16384
    return _ne_flat_launch(A, B, out, n, block, 4)


def ne(A, B):
    logger.debug("GEMS_ENFLAME NE")
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _ne_cmp_run(A, B, out):
            return out
    return ne_func(A, B)


def _ne_scalar_tier(dtype, n):
    """(BLOCK, num_warps) for the scalar overload, None -> generic path."""
    if dtype == torch.float32:
        return (131072, 8) if n < _LATE_N else None
    return (262144, 4) if n >= _MID_N else (16384, 4)


def ne_scalar(A, B):
    logger.debug("GEMS_ENFLAME NE_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        tier = _ne_scalar_tier(A.dtype, n) if n >= _SMALL_N else None
        if tier is not None:
            block, nw = tier
            if n <= _INT32_SAFE_LIMIT - block and triton.cdiv(n, block) <= _GRID_LIMIT:
                out = torch.empty_like(A, dtype=torch.bool)
                _ne_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                    A, out, n, float(B), BLOCK=block, num_warps=nw
                )
                return out
    return ne_func_scalar(A, B)
