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

"""Element-wise less-than, GCU400-optimized.

The previous ``pointwise_dynamic`` path (gcu400-local codegen: 48-CTA grid
cap, 65536-lane block-pointer tiles, grid-stride loop, fp32 up-cast) loses
~10-27% versus the native kernel on mid shapes and ~6-15% on the largest
f16/bf16 shapes (bench core, L600 baseline: less 0.98x geomean /
less_scalar 0.95x; the 2**24-element shapes are the worst at 0.74-0.91x —
the QA 0.743x less_scalar regression).

This file keeps the pointwise path for small / non-contiguous / broadcast /
non-float inputs, and dispatches large contiguous same-dtype float inputs to
dedicated flat 1D kernels (comparison-family template; tiers probe-validated
on the same board and bench protocol):

* comparisons run in the native input dtype (exact for same-dtype floats;
  the fp32 up-cast of the pointwise kernel only wastes ALU here);
* the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator semantics (the fp32-compare
  pointwise kernel deviates from torch on fp16 round-down scalars, e.g. 0.1);
* per-dtype DSM cap (max_dsm = 917504 B, compile-time enforced) bounds the
  tile size; a coverage/chunk loop kernel extends per-CTA coverage beyond
  the single-tile DSM limit for the largest shapes.

Probe-validated tiers (latency vs torch native, do_bench median):
  tensor (f16/bf16):
    n < 2**16                -> pointwise fallback (launch-overhead floor)
    2**16..2**22             -> flat BLOCK=16384, nw=4
    2**22..2**28             -> flat BLOCK=131072, nw=4
    n >= 2**28               -> loop COV=524288 CHUNK=131072 nw=4 st=2
  tensor (f32):
    n < 2**16                -> pointwise fallback
    2**16..2**19             -> flat BLOCK=8192, nw=4
    2**19..2**22 / n >= 2**28 -> pointwise fallback (fastest arm in those bands)
    2**22..2**24             -> loop COV=262144 CHUNK=65536 nw=8
    2**24..2**28             -> loop COV=1048576 CHUNK=65536 nw=8
  scalar (f16/bf16):
    0 < n < 2**22            -> flat BLOCK=16384, nw=4
    n >= 2**22               -> flat BLOCK=262144, nw=4
  scalar (f32):
    0 < n < 2**19            -> flat BLOCK=min(8192, max(1024, np2(n))), nw=4
    2**19..2**22 / n >= 2**28 -> pointwise fallback
    2**22..2**28             -> flat BLOCK=131072, nw=4

The fp32 bands differ from the f16/bf16 picture because of wave
quantization and DRAM-page behaviour: flat kernels whose grid crosses the
48-slot wave boundary (fp32 BLOCK=16384 above ~2**19) lose to the persistent
pointwise path, which in turn scales best at the very largest fp32 shapes
(2**28+, ~0.97-0.99x vs native) where every flat/loop variant plateaus at
~0.95-0.98x. The pointwise fallback keeps the gcu400-local codegen (48-CTA,
not the 12-CTA public default), so every non-fast-path corner keeps its
previous behaviour and latency.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_FP32_MID_N = 2**19
_MID_N = 2**22
_FP32_BIG_N = 2**24
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


@triton.jit
def _lt_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x < y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _lt_cmp_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x < y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _lt_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x < Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _launch_flat(A, B, out, n, block, nw):
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _lt_cmp_kernel[(triton.cdiv(n, block),)](A, B, out, n, BLOCK=block, num_warps=nw)
    return True


def _launch_loop(A, B, out, n, cov, chunk, nw, st):
    if n > _INT32_SAFE_LIMIT - cov:
        return False
    _lt_cmp_loop_kernel[(triton.cdiv(n, cov),)](
        A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
    )
    return True


def _lt_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if _FP32_MID_N <= n < _MID_N or n >= _BIG_N:
            return False
        if n >= _MID_N:
            cov = 1048576 if n >= _FP32_BIG_N else 262144
            return _launch_loop(A, B, out, n, cov, 65536, 8, 1)
        return _launch_flat(A, B, out, n, 8192, 4)
    if n >= _BIG_N:
        return _launch_loop(A, B, out, n, 1048576, 131072, 4, 2)
    return _launch_flat(A, B, out, n, (131072 if n >= _MID_N else 16384), 4)


def lt(A, B):
    logger.debug("GEMS_ENFLAME LT")
    if (
        A.numel() >= _SMALL_N
        and A.device == B.device
        and A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _lt_cmp_run(A, B, out):
            return out
    return lt_func(A, B)


def lt_scalar(A, B):
    logger.debug("GEMS_ENFLAME LT_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n > 0:
            if A.dtype == torch.float32:
                if _FP32_MID_N <= n < _MID_N or n >= _BIG_N:
                    return lt_func_scalar(A, B)
                if n >= _MID_N:
                    block = 131072
                else:
                    block = min(8192, max(1024, triton.next_power_of_2(n)))
            else:
                block = 262144 if n >= _MID_N else 16384
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _lt_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                    A, out, n, float(B), BLOCK=block, num_warps=4
                )
                return out
    return lt_func_scalar(A, B)
