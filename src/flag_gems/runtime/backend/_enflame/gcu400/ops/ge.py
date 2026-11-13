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

import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func(x, y):
    return x.to(tl.float32) >= y


# Large contiguous same-dtype float inputs are dispatched to dedicated flat 1D
# kernels instead of the ``pointwise_dynamic`` path (base bench core, L600:
# ge 0.771x / ge_scalar 0.752x vs native; weak spots 2**24 all dtypes 0.74-0.90x).
# The fast path is ported from the GCU400-validated ``greater_equal`` sibling
# (same ``>=`` semantics) and re-probed band-by-band on this board:
#
# * comparisons run in the native input dtype (exact for same-dtype floats);
#   the fp32 up-cast of the pointwise kernel only wastes ALU (-17-22% on large
#   shapes, greater_equal probe).
# * the scalar overload casts the Python scalar to ``A.dtype`` before
#   comparing, matching torch TensorIterator wrapped-number semantics.
# * per-dtype DSM cap (max_dsm = 917504 B, compile-time enforced) bounds the
#   tile size (flat double-load Required = 2 x itemsize x BLOCK); a
#   coverage/chunk loop kernel extends per-CTA coverage beyond the single-tile
#   DSM limit for the largest f16/bf16 shapes.
#
# Probe-validated tiers (interleaved A/B vs the pointwise fork and torch
# native, do_bench kernel median w1000/r100, L600):
#   tensor f16/bf16  n < 2**16        -> pointwise (launch-overhead floor)
#   tensor f16/bf16  2**16 .. 2**22   -> flat BLOCK=16384
#   tensor f16/bf16  2**22 .. 2**28   -> flat BLOCK=131072  (2**22: 16.1us vs native 18.6)
#   tensor f16/bf16  n >= 2**28       -> loop COV=524288 CHUNK=131072 nw4 st2
#   tensor f32       n < 2**16        -> pointwise
#   tensor f32       2**16 .. 2**20   -> flat BLOCK=16384
#   tensor f32       2**20 .. 2**22   -> flat BLOCK=65536
#   tensor f32       2**22 .. 2**28   -> loop COV=262144 CHUNK=65536 nw8  (2**24: 70.1 vs 72.5)
#   tensor f32       n >= 2**28       -> pointwise (beats every flat/loop variant at 2**30)
#   scalar  any      n < 2**14        -> pointwise
#   scalar  any      2**14 .. 2**20   -> flat BLOCK=16384
#   scalar  any      2**20 .. 2**22   -> flat BLOCK=65536   (2**21 f16: 7.3 vs 9.6)
#   scalar  f16/bf16 n >= 2**22       -> flat BLOCK=262144
#   scalar  f32      n >= 2**22       -> flat BLOCK=131072
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_SCALAR_SMALL_N = 2**14
_WIDE_N = 2**20
_MID_N = 2**22
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF

# The external triton_gcu toolchain SIGSEGVs on a fraction of cold compiles
# (compiler_problems/P02).  The crash is per-invocation probabilistic, leaves
# no Triton cache entry, and a fresh attempt compiles fine, so retry the
# launch; a deterministic failure is re-raised after the retries.
_TOOLCHAIN_COMPILE_RETRIES = 3


def _launch_with_toolchain_retry(launch):
    for attempt in range(_TOOLCHAIN_COMPILE_RETRIES):
        try:
            return launch()
        except Exception as error:
            if attempt == _TOOLCHAIN_COMPILE_RETRIES - 1:
                raise
            logger.debug(
                "ge: retrying Triton launch after toolchain failure (attempt %d): %r",
                attempt,
                error,
            )


@triton.jit
def _ge_flat_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x >= y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_loop_kernel(A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x >= y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _ge_flat_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x >= Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _ge_flat_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _BIG_N:
            return False
        if n >= _MID_N:
            cov, chunk, nw, st = 262144, 65536, 8, 1
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _launch_with_toolchain_retry(
                lambda: _ge_loop_kernel[(triton.cdiv(n, cov),)](
                    A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
                )
            )
            return True
        block = 65536 if n >= _WIDE_N else 16384
    else:
        if n >= _BIG_N:
            cov, chunk, nw, st = 524288, 131072, 4, 2
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _launch_with_toolchain_retry(
                lambda: _ge_loop_kernel[(triton.cdiv(n, cov),)](
                    A, B, out, n, COV=cov, CHUNK=chunk, num_warps=nw, num_stages=st
                )
            )
            return True
        block = 131072 if n >= _MID_N else 16384
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_toolchain_retry(
        lambda: _ge_flat_kernel[(triton.cdiv(n, block),)](
            A, B, out, n, BLOCK=block, num_warps=4
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
        if _ge_flat_run(A, B, out):
            return out
    return ge_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS_ENFLAME GE_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n >= _SCALAR_SMALL_N:
            if A.dtype == torch.float32:
                block = 131072 if n >= _MID_N else (65536 if n >= _WIDE_N else 16384)
            else:
                block = 262144 if n >= _MID_N else (65536 if n >= _WIDE_N else 16384)
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _launch_with_toolchain_retry(
                    lambda: _ge_flat_scalar_kernel[(triton.cdiv(n, block),)](
                        A, out, n, float(B), BLOCK=block, num_warps=4
                    )
                )
                return out
    return ge_func_scalar(A, B)
