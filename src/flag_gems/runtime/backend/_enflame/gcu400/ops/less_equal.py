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

"""Element-wise less-or-equal (tensor + scalar), GCU400-optimized.

Baseline (vendor seed, bench core, kernel mode do_bench median w1000/r100):
the tensor path was a grid-stride kernel with an fp32 up-cast and the scalar
path the ``max_grid_size=(1024, 1, 1)`` pointwise variant, losing
* tensor f16/bf16: 0.75-0.91x at n >= 2**24 (fp32 up-cast ALU tax + GSL cap 48)
* scalar (all dtypes): 0.29-0.53x at n >= 2**24 (oversized-grid pointwise codegen)

Rewrite following the probe-validated comparison-family recipe (lt/ge/ne
twins, same board and bench protocol) plus a fresh probe for this op
(probe_less_equal.json, do_bench median w100/r100):

* comparisons run in the native input dtype (exact for same-dtype floats);
  the fp32 up-cast only wastes ALU on 16-bit inputs;
* the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator semantics (the fp32-compare
  pointwise kernel deviates from torch on fp16 round-up scalars, e.g. 0.3);
* flat 1D kernels with per-dtype DSM-bounded tiles (max_dsm = 917504 B)
  and a coverage/chunk loop kernel for the largest 16-bit shapes.

Probe-validated tiers (latency vs torch native, do_bench median):
  tensor (f16/bf16):
    n < 2**16                -> pointwise fallback (launch-overhead floor)
    2**16..2**22             -> flat BLOCK=16384, nw=4        (lt-validated band)
    2**22..2**28             -> flat BLOCK=131072, nw=4       (2**24: 1.00-1.01x)
    n >= 2**28               -> loop COV=1048576 CHUNK=131072 nw=4 st=2
                                (2**30 f16: 1980.6us vs native 1951.8-1986.0us)
  tensor (f32):
    all n                    -> legacy persistent GSL kernel, BLOCK=65536,
                                grid cap 48 (unchanged vendor arm; probe: it
                                beats every flat/loop/pointwise variant at
                                2**24 (68.9us, 1.00x) and 2**30 (3807us, 0.98x))
  scalar (f16/bf16):
    0 < n < 2**22            -> flat BLOCK=16384, nw=4
    n >= 2**22               -> flat BLOCK=262144, nw=4
                                (2**24: 24.6us = 1.05x, 2**30: 1080.7us = 1.00x)
  scalar (f32):
    0 < n < 2**19            -> flat BLOCK=min(8192, max(1024, np2(n))), nw=4
    2**19..2**22 / n >= 2**28 -> pointwise fallback (fastest arm in those bands)
    2**22..2**28             -> flat BLOCK=131072, nw=4 (2**24: 39.8us = 1.06x)

The pointwise fallback keeps the gcu400-local codegen (48-CTA, not the
12-CTA public default), so every non-fast-path corner (broadcast, mixed
dtype, non-contiguous, integer dtypes) keeps its previous behaviour.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_FP32_MID_N = 2**19
_MID_N = 2**22
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF
NUM_SIPS = 24


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def less_equal_func(x, y):
    return x.to(tl.float32) <= y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def less_equal_func_scalar(x, y):
    return x.to(tl.float32) <= y


@triton.jit
def _le_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x <= y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _le_cmp_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x <= y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _le_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x <= Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@libentry()
@triton.jit(do_not_specialize=["N", "Y_SCALAR"])
def less_equal_kernel(X, Y, Out, N, Y_SCALAR, HAS_Y: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if HAS_Y:
            y = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        else:
            y = Y_SCALAR
        res = x <= y
        tl.store(Out + off, res, mask=mask)


def _choose_gsl_block(N):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return min(triton.next_power_of_2(N), 8192)
    return 65536


def _launch_flat(A, B, out, n, block, nw):
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _le_cmp_kernel[(triton.cdiv(n, block),)](A, B, out, n, BLOCK=block, num_warps=nw)
    return True


def _launch_loop(A, B, out, n, cov, chunk, nw, st):
    if n > _INT32_SAFE_LIMIT - cov:
        return False
    _le_cmp_loop_kernel[(triton.cdiv(n, cov),)](
        A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
    )
    return True


def _launch_gsl_tensor(A, B, out):
    n = A.numel()
    block = _choose_gsl_block(n)
    grid_size = min(triton.cdiv(n, block), NUM_SIPS * 2)
    with torch_device_fn.device(A.device):
        less_equal_kernel[(grid_size,)](
            A,
            B,
            out,
            n,
            0.0,
            HAS_Y=True,
            BLOCK=block,
            num_warps=4,
        )
    return True


def _le_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if A.dtype == torch.float32:
        return _launch_gsl_tensor(A, B, out)
    if n < _SMALL_N:
        return False
    if n >= _BIG_N:
        return _launch_loop(A, B, out, n, 1048576, 131072, 4, 2)
    return _launch_flat(A, B, out, n, (131072 if n >= _MID_N else 16384), 4)


def less_equal(A, B):
    logger.debug("GEMS_ENFLAME LESS_EQUAL")
    if (
        isinstance(B, torch.Tensor)
        and A.device == B.device
        and A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _le_cmp_run(A, B, out):
            return out
    return less_equal_func(A, B)


def less_equal_scalar(A, B):
    logger.debug("GEMS_ENFLAME LESS_EQUAL_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n > 0:
            if A.dtype == torch.float32:
                if _FP32_MID_N <= n < _MID_N or n >= _BIG_N:
                    return less_equal_func_scalar(A, B)
                block = (
                    131072
                    if n >= _MID_N
                    else min(8192, max(1024, triton.next_power_of_2(n)))
                )
            else:
                block = 262144 if n >= _MID_N else 16384
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _le_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                    A, out, n, float(B), BLOCK=block, num_warps=4
                )
                return out
    return less_equal_func_scalar(A, B)
