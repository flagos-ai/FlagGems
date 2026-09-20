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

from flag_gems.ops._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_ as default__amp_foreach_non_finite_check_and_unscale_,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def _amp_unscale_check_kernel(
    x_ptr,
    numel,
    base,
    inv_scale_ptr,
    found_inf_ptr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = base + pid * BLOCK + tl.arange(0, BLOCK)
    inv = tl.load(inv_scale_ptr)
    if EVEN:
        # Unmasked bulk path: full blocks, enables vectorized load/store codegen.
        x = tl.load(x_ptr + offs)
        y = x.to(tl.float32)
        bits = y.to(tl.int32, bitcast=True)
        non_finite = (bits & 0x7F800000) == 0x7F800000
        found = tl.max(non_finite.to(tl.int32), axis=0) > 0
        if found:
            tl.atomic_xchg(found_inf_ptr, 1.0)
        tl.store(x_ptr + offs, (y * inv).to(x.dtype))
    else:
        # Masked tail path: only the last partial block of a tensor.
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x.to(tl.float32)
        bits = y.to(tl.int32, bitcast=True)
        non_finite = (bits & 0x7F800000) == 0x7F800000
        found = tl.max(non_finite.to(tl.int32), axis=0) > 0
        if found:
            tl.atomic_xchg(found_inf_ptr, 1.0)
        tl.store(x_ptr + offs, (y * inv).to(x.dtype), mask=mask)


@libentry()
@triton.jit
def _amp_persistent_kernel(
    p0,
    p1,
    s1,
    total_blocks,
    inv_scale_ptr,
    found_inf_ptr,
    BLOCK: tl.constexpr,
    NUM_TENSORS: tl.constexpr,
    FP16: tl.constexpr,
):
    # Persistent grid: each program strides over blocks and accumulates the
    # per-element non-finite flags in registers, so the expensive cross-warp
    # reduce runs once per program instead of once per block.
    pid = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int32)
    inv = tl.load(inv_scale_ptr)
    for blk in range(pid, total_blocks, tl.num_programs(0)):
        if NUM_TENSORS == 2:
            if blk >= s1:
                ptr = p1
                offs = (blk - s1) * BLOCK + ar
            else:
                ptr = p0
                offs = blk * BLOCK + ar
        else:
            ptr = p0
            offs = blk * BLOCK + ar
        x = tl.load(ptr + offs)
        if FP16:
            # All-ones fp16 exponent (0x1F) marks inf or NaN.
            bits = x.to(tl.int16, bitcast=True).to(tl.int32)
            non_finite = (bits & 0x7C00) == 0x7C00
            out = x * inv.to(tl.float16)
        else:
            y = x.to(tl.float32)
            bits = y.to(tl.int32, bitcast=True)
            non_finite = (bits & 0x7F800000) == 0x7F800000
            out = (y * inv).to(x.dtype)
        acc |= non_finite.to(tl.int32)
        tl.store(ptr + offs, out)
    found = tl.max(acc, axis=0) > 0
    if found:
        tl.atomic_xchg(found_inf_ptr, 1.0)


_BLOCK = 4096
_NUM_WARPS = 4
_GRID = 256


def _use_triton_kernel(tensors, found_inf, inv_scale) -> bool:
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        return False
    if not isinstance(found_inf, torch.Tensor) or not isinstance(
        inv_scale, torch.Tensor
    ):
        return False
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            return False
        if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
            return False
        if not t.is_contiguous():
            return False
    return True


def _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale):
    logger.debug("GEMS_MTHREADS _AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_")
    if not _use_triton_kernel(tensors, found_inf, inv_scale):
        return default__amp_foreach_non_finite_check_and_unscale_(
            tensors, found_inf, inv_scale
        )
    n = len(tensors)
    if n == 0:
        return None
    fp16 = tensors[0].dtype == torch.float16
    with torch_device_fn.device(tensors[0].device):
        if n == 1:
            numel = tensors[0].numel()
            if numel % _BLOCK == 0 and numel > 0:
                total = numel // _BLOCK
                grid = min(total, _GRID)
                _amp_persistent_kernel[(grid,)](
                    tensors[0],
                    tensors[0],
                    total,
                    total,
                    inv_scale,
                    found_inf,
                    BLOCK=_BLOCK,
                    NUM_TENSORS=1,
                    FP16=fp16,
                    num_warps=_NUM_WARPS,
                )
                return None
        elif n == 2:
            n0 = tensors[0].numel()
            n1 = tensors[1].numel()
            if n0 % _BLOCK == 0 and n1 % _BLOCK == 0:
                s1 = n0 // _BLOCK
                total = s1 + n1 // _BLOCK
                grid = min(total, _GRID)
                _amp_persistent_kernel[(grid,)](
                    tensors[0],
                    tensors[1],
                    s1,
                    total,
                    inv_scale,
                    found_inf,
                    BLOCK=_BLOCK,
                    NUM_TENSORS=2,
                    FP16=fp16,
                    num_warps=_NUM_WARPS,
                )
                return None
        # Generic per-tensor path (masked tails or n > 2).
        for t in tensors:
            numel = t.numel()
            if numel == 0:
                continue
            full = numel // _BLOCK
            if full > 0:
                _amp_unscale_check_kernel[(full,)](
                    t,
                    numel,
                    0,
                    inv_scale,
                    found_inf,
                    BLOCK=_BLOCK,
                    EVEN=True,
                    num_warps=_NUM_WARPS,
                )
            tail = numel % _BLOCK
            if tail:
                base = full * _BLOCK
                _amp_unscale_check_kernel[(1,)](
                    t,
                    numel,
                    base,
                    inv_scale,
                    found_inf,
                    BLOCK=_BLOCK,
                    EVEN=False,
                    num_warps=_NUM_WARPS,
                )
    return None


__all__ = ["_amp_foreach_non_finite_check_and_unscale_"]
