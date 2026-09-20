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

from flag_gems.ops.atan2 import atan2 as default_atan2

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


_MAX_RANK = 8


@triton.jit
def _fast_atan2(y, x):
    # classic minimax atan approximation on [0,1] + quadrant fixups
    PI = 3.1415927410125732
    PI_2 = 1.5707963705062866
    ax = tl.abs(x)
    ay = tl.abs(y)
    a = tl.minimum(ax, ay) / tl.maximum(ax, ay)
    s = a * a
    r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a
    r = tl.where(ay > ax, PI_2 - r, r)
    r = tl.where(x < 0.0, PI - r, r)
    r = tl.where(y < 0.0, -r, r)
    r = tl.where((ax == 0.0) & (ay == 0.0), 0.0, r)
    return r


@triton.jit
def _atan2_flat_kernel(
    in_ptr,
    other_ptr,
    out_ptr,
    IS_FP64: tl.constexpr,
    FAST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.load(in_ptr + offs)
    x = tl.load(other_ptr + offs)
    if IS_FP64:
        r = tl.extra.libdevice.atan2(y.to(tl.float64), x.to(tl.float64))
        tl.store(out_ptr + offs, r)
    else:
        yf = y.to(tl.float32)
        xf = x.to(tl.float32)
        if FAST:
            r = _fast_atan2(yf, xf)
        else:
            r = tl.extra.libdevice.atan2(yf, xf)
        tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty))


@triton.jit
def _atan2_bcast_kernel(
    in_ptr,
    other_ptr,
    out_ptr,
    in_strides_ptr,
    other_strides_ptr,
    out_dims_ptr,
    IS_FP64: tl.constexpr,
    FAST: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rem = offs
    in_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    ot_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    # row-major decomposition: iterate dims from last (fastest) to first
    for rr in tl.static_range(RANK):
        r = RANK - 1 - rr
        d = tl.load(out_dims_ptr + r)
        c = rem % d
        rem = rem // d
        in_off += c * tl.load(in_strides_ptr + r)
        ot_off += c * tl.load(other_strides_ptr + r)
    y = tl.load(in_ptr + in_off)
    x = tl.load(other_ptr + ot_off)
    if IS_FP64:
        r = tl.extra.libdevice.atan2(y.to(tl.float64), x.to(tl.float64))
        tl.store(out_ptr + offs, r)
    else:
        yf = y.to(tl.float32)
        xf = x.to(tl.float32)
        if FAST:
            r = _fast_atan2(yf, xf)
        else:
            r = tl.extra.libdevice.atan2(yf, xf)
        tl.store(out_ptr + offs, r.to(out_ptr.dtype.element_ty))


def _pick_block(n):
    # largest power of two <= 1024 that divides n exactly, so the kernel needs
    # no boundary mask (masked fp16 kernels with mostly-inactive lanes crash
    # the MTT S5000 llc backend deterministically).
    v2 = (n & -n).bit_length() - 1
    return min(1024, 1 << v2)


def _specialized_atan2(input, other):
    out_dtype = torch.promote_types(input.dtype, other.dtype)
    is_fp64 = out_dtype == torch.float64
    fast = out_dtype in (torch.float16, torch.bfloat16)
    out_shape = torch.broadcast_shapes(input.shape, other.shape)
    out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    n = out.numel()
    if n == 0:
        return out
    use_flat = (
        input.shape == other.shape and input.is_contiguous() and other.is_contiguous()
    )
    block = _pick_block(n)
    num_warps = max(1, min(4, block // 32))
    grid = (n // block,)
    if use_flat:
        _atan2_flat_kernel[grid](
            input,
            other,
            out,
            IS_FP64=is_fp64,
            FAST=fast,
            BLOCK_SIZE=block,
            num_warps=num_warps,
        )
    else:
        rank = len(out_shape)
        rank_in = input.dim()
        rank_ot = other.dim()
        in_strides = []
        ot_strides = []
        out_dims = []
        for r in range(rank):
            out_dims.append(out_shape[r])
            if r >= rank - rank_in and input.shape[r - (rank - rank_in)] > 1:
                in_strides.append(input.stride()[r - (rank - rank_in)])
            else:
                in_strides.append(0)
            if r >= rank - rank_ot and other.shape[r - (rank - rank_ot)] > 1:
                ot_strides.append(other.stride()[r - (rank - rank_ot)])
            else:
                ot_strides.append(0)
        in_strides_t = torch.tensor(in_strides, dtype=torch.int64, device=input.device)
        ot_strides_t = torch.tensor(ot_strides, dtype=torch.int64, device=input.device)
        out_dims_t = torch.tensor(out_dims, dtype=torch.int64, device=input.device)
        _atan2_bcast_kernel[grid](
            input,
            other,
            out,
            in_strides_t,
            ot_strides_t,
            out_dims_t,
            IS_FP64=is_fp64,
            FAST=fast,
            RANK=rank,
            BLOCK_SIZE=block,
            num_warps=num_warps,
        )
    return out


def atan2(input, other):
    logger.debug("GEMS_MTHREADS ATAN2")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
        and isinstance(other, torch.Tensor)
        and other.device.type == "musa"
        and other.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_atan2(input, other)
    return default_atan2(input, other)
