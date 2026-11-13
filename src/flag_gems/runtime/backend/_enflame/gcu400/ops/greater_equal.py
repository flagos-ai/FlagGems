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

"""Element-wise greater-or-equal, GCU400-optimized.

The previous delegation to the public ``ge`` implementation ran through
``pointwise_dynamic`` whose ENFLAME ``CodeGenConfig`` caps the launch grid at
``(12, 1, 1)``; every shape beyond ~2**24 elements was serialized onto 12 CTAs
and reached only ~240-425 GB/s versus ~1.4-2.8 TB/s for the native kernel
(bench core, L600: greater_equal 0.215x / greater_equal_scalar 0.148x /
greater_equal_ 0.489x).

This file keeps the generic pointwise path for small / non-contiguous /
broadcast / non-float inputs, and dispatches large contiguous same-dtype float
inputs to dedicated flat 1D kernels (L600 probe, kernel mode do_bench):

* tensor & in-place share one kernel; out-of-place writes bool, in-place casts
  the result back into ``A`` (same semantics as ``ge_func(..., out0=A)``).
* comparisons run in the native input dtype (exact for same-dtype floats;
  the fp32 up-cast of the public kernel only wastes ALU here), and the
  scalar overload casts the Python scalar to ``A.dtype`` before comparing,
  which matches torch TensorIterator semantics; the public fp32-compare
  kernel deviates from torch on fp16 round-down scalars (e.g. 0.1).
* per-dtype DSM cap (max_dsm = 917504 B, compile-time enforced) bounds the
  tile size; a coverage/chunk loop kernel extends per-CTA coverage beyond
  the single-tile DSM limit for the largest shapes.

Probe-validated tiers (latency vs torch native, do_bench median):
  tensor/inplace n < 2**16   -> generic pointwise (launch-overhead floor)
  tensor/inplace 2**16..2**22 -> flat BLOCK=16384, nw=4   (~1.4-2.1x)
  tensor/inplace 2**22..2**28 (f16) -> flat BLOCK=131072, nw=4 (~0.96-1.12x)
  tensor/inplace n >= 2**22 (f32)   -> loop COV=262144 CHUNK=65536 nw=8 (~0.96-0.98x)
  tensor/inplace n >= 2**28 (f16/bf16) -> loop COV=524288 CHUNK=131072 nw=4 st=2 (~0.94-0.98x)
  scalar 0 < n < 2**22       -> flat BLOCK=16384, nw=4
  scalar n >= 2**22          -> flat BLOCK=262144 (f16/bf16) / 131072 (f32) (~0.98-1.06x)
``ge_func`` / ``ge_func_scalar`` (public pointwise kernels) remain the
fallback, so untuned corners keep their previous behaviour.
"""  # b2

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.ge import ge_func, ge_func_scalar

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_MID_N = 2**22
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF


@triton.jit
def _ge_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x >= y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_cmp_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x >= y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x >= Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _ge_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _MID_N:
            cov, chunk, nw, st = 262144, 65536, 8, 1
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _ge_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
            )
            return True
        block, nw = 16384, 4
    else:
        if n >= _BIG_N:
            cov, chunk, nw, st = 524288, 131072, 4, 2
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _ge_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
            )
            return True
        block, nw = (131072, 4) if n >= _MID_N else (16384, 4)
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _ge_cmp_kernel[(triton.cdiv(n, block),)](A, B, out, n, BLOCK=block, num_warps=nw)
    return True


def greater_equal(A, B):
    logger.debug("GEMS_ENFLAME GREATER_EQUAL")
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


def greater_equal_scalar(A, B):
    logger.debug("GEMS_ENFLAME GREATER_EQUAL_SCALAR ")  # b2
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n > 0:
            if n >= _MID_N:
                block = 262144 if A.dtype != torch.float32 else 131072
            else:
                block = 16384
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _ge_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                    A, out, n, float(B), BLOCK=block, num_warps=4
                )
                return out
    return ge_func_scalar(A, B)


def greater_equal_(A, B):
    logger.debug("GEMS_ENFLAME GREATER_EQUAL_")
    if A.device != B.device:
        B = B.to(A.device)
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        if _ge_cmp_run(A, B, A):
            return A
    return ge_func(A, B, out0=A)
