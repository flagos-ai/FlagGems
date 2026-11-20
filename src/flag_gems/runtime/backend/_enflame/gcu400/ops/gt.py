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
inputs dispatch to dedicated flat 1D kernels (L600 probe, kernel mode do_bench):

* tensor & in-place share one kernel; out-of-place writes bool, in-place casts
  the result back into ``A`` (same semantics as ``gt_func(..., out0=A)``).
* comparisons run in the native input dtype (exact for same-dtype floats; the
  fp32 up-cast of the pointwise kernel only wastes ALU on the large tiers), and
  the scalar overload casts the Python scalar to ``A.dtype`` before comparing,
  which matches torch TensorIterator semantics.
* per-dtype DSM cap (max_dsm = 917504 B, compile-time enforced) bounds the
  tile size; a coverage/chunk loop kernel extends per-CTA coverage beyond the
  single-tile DSM limit for the largest shapes.

Probe-validated tiers (L600 kernel do_bench; ladder from greater_equal/ge plus
an fp32-specific huge-shape probe):
  tensor/inplace n < 2**16   -> pointwise (launch-overhead floor)
  tensor/inplace 2**16..2**22 (f16/bf16) -> flat BLOCK=16384, nw=4
  tensor/inplace 2**16..2**20 (f32) -> flat BLOCK=16384, nw=4
  tensor/inplace 2**20..2**22 (f32) -> flat BLOCK=65536, nw=4
  tensor/inplace 2**22..2**28 (f16/bf16) -> flat BLOCK=131072, nw=4
  tensor/inplace 2**22..2**28 (f32)   -> loop COV=262144 CHUNK=65536 nw=8
  tensor/inplace n >= 2**28 (f16/bf16) -> loop COV=524288 CHUNK=131072 nw=4 st=2
  tensor/inplace n >= 2**28 (f32)   -> pointwise (fastest at 2^30: 3.744 ms
                                        vs loop 3.800 / native 3.784 ms)
  scalar n < 2**14           -> pointwise (launch-overhead floor)
  scalar 2**14..2**20        -> flat BLOCK=16384, nw=4
  scalar 2**20..2**22        -> flat BLOCK=65536, nw=4
  scalar n >= 2**22          -> flat BLOCK=262144 (f16/bf16) / 131072 (f32)
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
_MID_N = 2**22
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF


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
def _gt_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x > y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x > y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x > Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _gt_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _BIG_N:
            # The pointwise path is the fastest tier for fp32 at the largest
            # shapes (L600 2^30 probe: pointwise 3.744 ms < loop 3.800 ms,
            # native 3.784 ms); fp16/bf16 prefer the flat/loop kernels there.
            return False
        if n >= _MID_N:
            cov, chunk, nw, st = 262144, 65536, 8, 1
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _launch_with_retry(
                lambda: _gt_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                    A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
                )
            )
            return True
        block, nw = (65536, 4) if n >= _LOW_N else (16384, 4)
    else:
        if n >= _BIG_N:
            cov, chunk, nw, st = 524288, 131072, 4, 2
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _launch_with_retry(
                lambda: _gt_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                    A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
                )
            )
            return True
        block, nw = (131072, 4) if n >= _MID_N else (16384, 4)
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _gt_cmp_kernel[(triton.cdiv(n, block),)](
            A, B, out, n, BLOCK=block, num_warps=nw
        )
    )
    return True


def _gt_scalar_block(A, n):
    if n >= _MID_N:
        return 262144 if A.dtype != torch.float32 else 131072
    if n >= _LOW_N:
        return 65536
    return 16384


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
        n = A.numel()
        if n >= _SCALAR_SMALL_N:
            block = _gt_scalar_block(A, n)
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _launch_with_retry(
                    lambda: _gt_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                        A, out, n, float(B), BLOCK=block, num_warps=4
                    )
                )
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
        n = A.numel()
        if n >= _SCALAR_SMALL_N:
            block = _gt_scalar_block(A, n)
            if n <= _INT32_SAFE_LIMIT - block:
                _launch_with_retry(
                    lambda: _gt_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                        A, A, n, float(B), BLOCK=block, num_warps=4
                    )
                )
                return A
    return gt_func_scalar(A, B, out0=A)
