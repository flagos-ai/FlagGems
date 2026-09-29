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

from flag_gems.ops.arctan2 import arctan2_ as default_arctan2_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def _atan2_manual(x, y):
    # Matches the generic implementation: libdevice.atan2(input, other) computed
    # in fp32 (libdevice on this backend implements the full-precision atan2,
    # including quadrant and inf/nan handling).
    return tl.extra.libdevice.atan2(x, y)


@triton.jit
def _arctan2_direct_um(
    x_ptr,
    y_ptr,
    UPCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Unmasked fast path: only launched when numel % BLOCK == 0, so every
    # element in the block is in-bounds; avoids mask compare/predicate work.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    if UPCAST:
        x = x.to(tl.float32)
        y = y.to(tl.float32)
    r = _atan2_manual(x, y)
    if UPCAST:
        r = r.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offs, r)


@triton.jit
def _arctan2_direct(
    x_ptr,
    y_ptr,
    numel,
    UPCAST: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if WIDE:
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    else:
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    if UPCAST:
        x = x.to(tl.float32)
        y = y.to(tl.float32)
    r = _atan2_manual(x, y)
    if UPCAST:
        r = r.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offs, r, mask=mask)


@triton.jit
def _arctan2_general(
    x_ptr,
    y_ptr,
    numel,
    shape_ptr,
    xs_ptr,
    ys_ptr,
    NDIM: tl.constexpr,
    UPCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    i = offs
    xoff = tl.zeros([BLOCK], dtype=tl.int64)
    yoff = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(NDIM):
        sh = tl.load(shape_ptr + d)
        xs = tl.load(xs_ptr + d)
        ys = tl.load(ys_ptr + d)
        idx = i % sh
        i = i // sh
        xoff += idx * xs
        yoff += idx * ys
    x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
    y = tl.load(y_ptr + yoff, mask=mask, other=0.0)
    if UPCAST:
        x = x.to(tl.float32)
        y = y.to(tl.float32)
    r = _atan2_manual(x, y)
    tl.store(x_ptr + xoff, r, mask=mask)


def _specialized_arctan2_(input, other):
    x = input
    y = other
    numel = x.numel()
    if numel == 0:
        return x
    upcast = x.dtype in (torch.float16, torch.bfloat16)
    if x.shape == y.shape and x.is_contiguous() and y.is_contiguous():
        if numel % 2048 == 0 and numel >= (1 << 20) and numel <= (1 << 30):
            grid = (numel // 2048,)
            _arctan2_direct_um[grid](x, y, UPCAST=upcast, BLOCK=2048, num_warps=8)
        else:
            grid = (triton.cdiv(numel, _BLOCK),)
            _arctan2_direct[grid](
                x,
                y,
                numel,
                UPCAST=upcast,
                WIDE=numel > (1 << 30),
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
    else:
        yb = torch.broadcast_to(y, x.shape)
        ndim = x.ndim
        shape_dev = torch.tensor(x.shape, dtype=torch.int64, device=x.device)
        xs_dev = torch.tensor(x.stride(), dtype=torch.int64, device=x.device)
        ys_dev = torch.tensor(yb.stride(), dtype=torch.int64, device=x.device)
        grid = (triton.cdiv(numel, _BLOCK),)
        _arctan2_general[grid](
            x,
            yb,
            numel,
            shape_dev,
            xs_dev,
            ys_dev,
            NDIM=ndim,
            UPCAST=upcast,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return x


def arctan2_(input, other):
    logger.debug("GEMS_MTHREADS ARCTAN2_")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
        and isinstance(other, torch.Tensor)
        and other.device.type == "musa"
        and other.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_arctan2_(input, other)
    return default_arctan2_(input, other)
