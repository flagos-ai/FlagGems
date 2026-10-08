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

from flag_gems.ops.atan2_ import atan2_ as default_atan2_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


_BLOCK = 1024
_NUM_WARPS = 4
_MAX_DIMS = 8


@triton.jit
def _atan2_flat_kernel(
    A,
    B,
    n_elements,
    B_MOD: tl.constexpr,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    OFF32: tl.constexpr,
):
    if OFF32:
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    else:
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < n_elements
    a = tl.load(A + offs, mask=mask)
    if B_MOD > 0:
        b_off = offs % B_MOD
    else:
        b_off = offs
    b = tl.load(B + b_off, mask=mask)
    if UPCAST:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    b = b.to(a.dtype)
    r = tl.extra.musa.libdevice.atan2(a, b)
    r = r.to(A.dtype.element_ty)
    tl.store(A + offs, r, mask=mask)


@triton.jit
def _atan2_general_kernel(
    A,
    B,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    i = offs
    a_off = tl.zeros_like(offs)
    b_off = tl.zeros_like(offs)
    if D >= 1:
        idx = i % d0
        i = i // d0
        a_off += idx * s0
        b_off += idx * t0
    if D >= 2:
        idx = i % d1
        i = i // d1
        a_off += idx * s1
        b_off += idx * t1
    if D >= 3:
        idx = i % d2
        i = i // d2
        a_off += idx * s2
        b_off += idx * t2
    if D >= 4:
        idx = i % d3
        i = i // d3
        a_off += idx * s3
        b_off += idx * t3
    if D >= 5:
        idx = i % d4
        i = i // d4
        a_off += idx * s4
        b_off += idx * t4
    if D >= 6:
        idx = i % d5
        i = i // d5
        a_off += idx * s5
        b_off += idx * t5
    if D >= 7:
        idx = i % d6
        i = i // d6
        a_off += idx * s6
        b_off += idx * t6
    if D >= 8:
        idx = i % d7
        i = i // d7
        a_off += idx * s7
        b_off += idx * t7
    a = tl.load(A + a_off, mask=mask)
    b = tl.load(B + b_off, mask=mask)
    if UPCAST:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    b = b.to(a.dtype)
    r = tl.extra.musa.libdevice.atan2(a, b)
    r = r.to(A.dtype.element_ty)
    tl.store(A + a_off, r, mask=mask)


def _suffix_compatible(a_shape, b_shape):
    """True when B's non-1 dims align exactly to A's trailing dims, so flat
    index into B for flat index i into A equals i % B.numel()."""
    if len(b_shape) > len(a_shape):
        extra = len(b_shape) - len(a_shape)
        if any(b_shape[k] != 1 for k in range(extra)):
            return False
        b_shape = b_shape[extra:]
    k = len(a_shape) - len(b_shape)
    for i, bs in enumerate(b_shape):
        if bs != a_shape[k + i]:
            return False
    return True


def _pick_config(n, dtype):
    """Return (BLOCK, num_warps) for the flat kernels."""
    # MUSA llc segfaults in MTGPU DAG instruction selection for fp16 masked
    # kernels with BLOCK > 64 when the tail is not 32-lane aligned; use a
    # small block (no per-thread vectorization) in that case.
    if dtype == torch.float16 and n % 32 != 0:
        return 64, _NUM_WARPS
    # Small tensors: launch overhead dominates; fewer warps + small block
    # measured ~2.2x faster than BLOCK=1024/4 warps on 64x64 shapes.
    if n <= 16384:
        return 64, 2
    # fp32 medium/large: BLOCK=2048/8 warps measured ~1% (1G) to ~3% (16.7M)
    # faster than BLOCK=1024/4; fp16/bf16 show no gain, so keep them as-is.
    if dtype == torch.float32:
        return 2048, 8
    return _BLOCK, _NUM_WARPS


def _specialized_atan2_(A, B):
    n = A.numel()
    if n == 0:
        return A
    upcast = A.dtype in (torch.float16, torch.bfloat16)
    block, nw = _pick_config(n, A.dtype)
    # int32 offsets are ~2-3% faster for the fp16/bf16 upcast path; safe while
    # n < 2^31 (largest eval workload is 2^30), int64 fallback otherwise.
    off32 = n < (1 << 31)
    grid = (triton.cdiv(n, block),)

    if A.is_contiguous() and B.numel() == 1:
        # scalar B: offs % 1 folds to 0, giving a broadcast vector load.
        # A dedicated scalar-load kernel crashes the MUSA llc backend, so we
        # reuse the flat kernel with B_MOD=1 instead.
        _atan2_flat_kernel[grid](
            A, B, n, B_MOD=1, BLOCK=block, UPCAST=upcast, OFF32=off32, num_warps=nw
        )
        return A

    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        _atan2_flat_kernel[grid](
            A, B, n, B_MOD=0, BLOCK=block, UPCAST=upcast, OFF32=off32, num_warps=nw
        )
        return A

    if A.is_contiguous() and B.is_contiguous() and _suffix_compatible(A.shape, B.shape):
        _atan2_flat_kernel[grid](
            A,
            B,
            n,
            B_MOD=B.numel(),
            BLOCK=block,
            UPCAST=upcast,
            OFF32=off32,
            num_warps=nw,
        )
        return A

    d = A.dim()
    if d > _MAX_DIMS:
        raise ValueError(
            f"atan2_ does not support tensors with more than {_MAX_DIMS} dims"
        )
    dims = list(A.shape) + [1] * (_MAX_DIMS - d)
    a_strides = list(A.stride()) + [0] * (_MAX_DIMS - d)
    if B.dim() == 0:
        b_strides = [0] * _MAX_DIMS
    else:
        k = d - B.dim()
        b_strides = [0] * k + list(B.stride()) + [0] * (_MAX_DIMS - d)
        for i in range(d):
            if i >= k and B.shape[i - k] == 1:
                b_strides[i] = 0
    # The strided fallback crashed llc for fp16 even with 32-aligned tails, so
    # keep it on the smallest block for fp16; other dtypes use the default
    # config to avoid new codegen risk on a path the workloads never hit.
    gblock = 64 if A.dtype == torch.float16 else _BLOCK
    _atan2_general_kernel[(triton.cdiv(n, gblock),)](
        A,
        B,
        n,
        *dims,
        *a_strides,
        *b_strides,
        D=d,
        BLOCK=gblock,
        UPCAST=upcast,
        num_warps=_NUM_WARPS,
    )
    return A


def atan2_(A, B):
    logger.debug("GEMS_MTHREADS ATAN2_")
    if (
        isinstance(A, torch.Tensor)
        and A.device.type == "musa"
        and A.dtype in _SUPPORTED_DTYPES
        and isinstance(B, torch.Tensor)
        and B.device.type == "musa"
        and B.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_atan2_(A, B)
    return default_atan2_(A, B)
