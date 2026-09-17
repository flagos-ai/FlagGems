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

from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

_NP2 = triton.next_power_of_2
_NPROGS = 48
_BS = 32768


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS_ENFLAME EQ")
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq_scalar(A, B):
    logger.debug("GEMS_ENFLAME EQ_SCALAR")
    return eq_func_scalar(A, B)


@triton.jit
def _equal_small_k(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    all_eq = tl.full((), 1, dtype=tl.int32)
    for off in range(0, numel, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        eq_val = tl.min((x == y).to(tl.int32), axis=0)
        all_eq = all_eq & eq_val
    tl.store(out_ptr, all_eq)


@triton.jit
def _equal_fused_k(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    num_progs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    all_eq = tl.full((), 1, dtype=tl.int32)
    blk = pid
    while blk * BLOCK_SIZE < numel:
        offsets = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        eq_val = tl.min((x == y).to(tl.int32), axis=0)
        all_eq = all_eq & eq_val
        blk += num_progs
    tl.store(out_ptr + pid, all_eq)


@triton.jit
def _reduce_min_k(in_ptr, out_ptr, N: tl.constexpr):
    idx = tl.arange(0, N)
    v = tl.load(in_ptr + idx)
    result = tl.min(v, axis=0)
    tl.store(out_ptr, result)


def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    logger.debug("GEMS_ENFLAME EQUAL")
    if x.shape != y.shape:
        return False
    if x.numel() == 0:
        return True

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    numel = x_flat.numel()

    out = torch.empty(1, dtype=torch.int32, device=x.device)

    if numel <= _BS:
        _equal_small_k[(1,)](
            x_flat,
            y_flat,
            out,
            numel,
            BLOCK_SIZE=_NP2(numel),
            num_warps=1,
        )
    else:
        NP = _NP2(_NPROGS)
        out_t = torch.ones(NP, dtype=torch.int32, device=x.device)
        _equal_fused_k[(_NPROGS,)](
            x_flat,
            y_flat,
            out_t,
            numel,
            _NPROGS,
            BLOCK_SIZE=_BS,
            num_warps=1,
        )
        _reduce_min_k[(1,)](out_t, out, N=NP, num_warps=1)

    return bool(out.item() > 0)
