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

The previous path delegated to the public ``pointwise_dynamic`` kernel
(12-CTA codegen + fp32 up-cast) which moves only ~0.5 TB/s where the native
kernel sustains ~2.4-2.9 TB/s on contiguous float inputs, losing 63-83% on
the bench-core mid/large shapes (baseline: ne_ 0.18-0.37x, ne_scalar_
0.12-0.27x; the 2**24/2**30 bands). Same in-place pathology as ``less_``
(family sibling): OUT aliases A and the store carries A.dtype (2/4-byte
writes instead of the 1-byte bool store of the out-of-place kernel), which
shifts the optimal tile per band and tightens the DSM caps (loads-only
budget, max 917504 B): flat/CHUNK cap at 131072 (f16/bf16, 4 B/elem loads)
and 65536 (f32, 8 B/elem loads).

Probe-validated tiers (latency vs torch native, do_bench median, kernel
mode warmup 1000ms/rep 100ms, contiguous float inputs):

* tensor (f16/bf16): n < 2**16 pointwise; 2**16..2**19 flat 8192 nw4;
  2**19..2**22 flat 16384 nw4; 2**22..2**24 flat 131072 nw4 (0.94x);
  2**24..2**28 loop COV=1048576 CHUNK=131072 nw4 st1 (0.96x);
  n >= 2**28 loop COV=1048576 CHUNK=131072 nw4 st2 (1.03x).
* tensor (f32): n < 2**16 pointwise; 2**16..2**19 flat 8192 nw4;
  2**19..2**28 flat 32768 nw4 (0.98x); n >= 2**28 loop COV=1048576
  CHUNK=65536 nw8 st1 (1.04x).
* scalar (f16/bf16): n < 2**16 pointwise; 2**16..2**22 adaptive flat
  (<=16384); n >= 2**22 flat 131072 nw4 (1.03-1.05x).
* scalar (f32): n < 2**16 pointwise; 2**16..2**19 adaptive flat (<=8192);
  2**19..2**22 flat 32768 nw4; n >= 2**22 flat 65536 nw4 (1.02-1.04x).
* comparisons run in the native input dtype (exact for same-dtype floats;
  the fp32 up-cast of the pointwise kernel only wastes ALU); the scalar is
  cast to ``A.dtype`` before comparing, matching torch TensorIterator
  semantics (fp16 round-down scalars such as 0.1).
* fallback keeps the backend-local 48-CTA ``ne_func``/``ne_func_scalar``
  (not the 12-CTA public one) for small / non-contiguous / broadcast /
  mixed-dtype / bool inputs.
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
_MID_N = 2**22
_BIG_N = 2**24
_HUGE_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_GRID_LIMIT = 65535


@triton.jit
def _ne_inplace_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x != y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


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
def _ne_inplace_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x != Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _launch_flat(A, B, n, block, nw):
    if n > _INT32_SAFE_LIMIT - block:
        return False
    grid = triton.cdiv(n, block)
    if grid > _GRID_LIMIT:
        return False
    _ne_inplace_kernel[(grid,)](A, B, A, n, BLOCK=block, num_warps=nw)
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
        if n >= _SMALL_MID_N:
            return _launch_flat(A, B, n, 32768, 4)
        if n >= _SMALL_N:
            return _launch_flat(A, B, n, 8192, 4)
        return False
    if n >= _HUGE_N:
        return _launch_loop(A, B, n, 1048576, 131072, 4, 2)
    if n >= _BIG_N:
        return _launch_loop(A, B, n, 1048576, 131072, 4, 1)
    if n >= _MID_N:
        return _launch_flat(A, B, n, 131072, 4)
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


def ne_scalar_(A, B):
    logger.debug("GEMS_ENFLAME NE__SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n >= _SMALL_N:
            if A.dtype == torch.float32:
                if n >= _MID_N:
                    block = 65536
                elif n >= _SMALL_MID_N:
                    block = 32768
                else:
                    block = min(8192, max(1024, triton.next_power_of_2(n)))
            else:
                block = (
                    131072
                    if n >= _MID_N
                    else min(16384, max(1024, triton.next_power_of_2(n)))
                )
            if n <= _INT32_SAFE_LIMIT - block and triton.cdiv(n, block) <= _GRID_LIMIT:
                _ne_inplace_scalar_kernel[(triton.cdiv(n, block),)](
                    A, A, n, float(B), BLOCK=block, num_warps=4
                )
                return A
    ne_func_scalar(A, B, out0=A)
    return A
