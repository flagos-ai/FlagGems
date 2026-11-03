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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _hinge_embedding_loss_elementwise_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    margin,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    # target == 1  -> x ; target == -1 -> max(0, margin - x)
    is_neg = yf < 0.0
    vals = tl.where(is_neg, tl.maximum(margin - xf, 0.0), xf)

    if NEED_MASK:
        tl.store(out_ptr + offsets, vals, mask=mask)
    else:
        tl.store(out_ptr + offsets, vals)


@libentry()
@triton.jit
def _hinge_embedding_loss_partial_sum_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    n_elements,
    margin,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    is_neg = yf < 0.0
    vals = tl.where(is_neg, tl.maximum(margin - xf, 0.0), xf)
    if NEED_MASK:
        vals = tl.where(mask, vals, 0.0)

    # aten computes the per-element loss in the input dtype and sums those
    # already-rounded values; for half/bfloat16 that rounding is observable in
    # the reduction result (e.g. bf16 0.560546875 -> 0.5625 before the sum), so
    # round each element before accumulating to match eager bit-for-bit.
    vals = vals.to(x_ptr.dtype.element_ty).to(tl.float32)

    tl.store(mid_ptr + pid, tl.sum(vals, axis=0))


@libentry()
@triton.jit
def _hinge_embedding_loss_final_reduce_kernel(
    mid_ptr,
    out_ptr,
    mid_size,
    n_elements,
    reduction: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    # Loop-accumulate into a [BLOCK_MID] fp32 tile, then a SINGLE tl.sum.
    # BLOCK_MID is capped at 8192 (XPU tl.sum only reduces the first 8192
    # lanes correctly), so when mid_size > 8192 we must stride over it in
    # chunks; the element-wise `acc +=` accumulation across iterations is
    # correct on XPU and a single final reduce stays within 8192.
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    n_iter = tl.cdiv(mid_size, BLOCK_MID)
    for i in range(n_iter):
        offset = i * BLOCK_MID + tl.arange(0, BLOCK_MID)
        mask = offset < mid_size
        acc += tl.load(mid_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(acc, axis=0)
    # Reduction.MEAN.value: 1, Reduction.SUM.value: 2
    if reduction == 1:
        total = total / n_elements
    tl.store(out_ptr, total.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def _hinge_embedding_loss_single_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    margin,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Single-program path for n <= 8192: one load, one tl.sum, one launch.
    offsets = tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    is_neg = yf < 0.0
    vals = tl.where(is_neg, tl.maximum(margin - xf, 0.0), xf)
    if NEED_MASK:
        vals = tl.where(mask, vals, 0.0)
    vals = vals.to(x_ptr.dtype.element_ty).to(tl.float32)

    total = tl.sum(vals, axis=0)
    if reduction == 1:
        total = total / n_elements
    tl.store(out_ptr, total.to(out_ptr.dtype.element_ty))


def _normalize_reduction(reduction):
    # Accept both string and enum/int forms: 0=none,1=mean,2=sum
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def hinge_embedding_loss(
    input: torch.Tensor, target: torch.Tensor, margin: float = 1.0, reduction="mean"
):
    logger.debug("GEMS_KUNLUNXIN HINGE_EMBEDDING_LOSS")
    if input.device.type != target.device.type:
        raise AssertionError(
            "hinge_embedding_loss: input and target must be on the same device."
        )
    red = _normalize_reduction(reduction)
    margin = float(margin)
    n_elements = input.numel()

    if red == 0:
        # reduction = 'none'
        out = torch.empty_like(input)
        if n_elements == 0:
            return out
        # Constant tuple grid (NOT a lambda): on the xpubin backend the grid
        # participates in the compilation cache key, so a fresh lambda per
        # call would force a full recompile on every launch.
        BLOCK_SIZE = 8192
        need_mask = n_elements % BLOCK_SIZE != 0
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
        with torch_device_fn.device(input.device):
            _hinge_embedding_loss_elementwise_kernel[grid](
                input,
                target,
                out,
                n_elements,
                margin,
                BLOCK_SIZE=BLOCK_SIZE,
                NEED_MASK=need_mask,
            )
        return out

    # reduction = 'mean' (1) or 'sum' (2)
    if n_elements == 0:
        # Follow PyTorch behavior: sum -> 0, mean -> NaN
        if red == 2:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        else:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    # Small shapes: a single-program kernel (one launch) beats the two-stage
    # partial/final pair on launch-bound tiny inputs.
    if n_elements <= 8192:
        block_size = triton.next_power_of_2(n_elements)
        out = torch.empty_strided((), (), dtype=input.dtype, device=input.device)
        with torch_device_fn.device(input.device):
            _hinge_embedding_loss_single_kernel[(1, 1, 1)](
                input,
                target,
                out,
                n_elements,
                margin,
                red,
                BLOCK_SIZE=block_size,
                NEED_MASK=n_elements != block_size,
            )
        return out

    # XPU tl.sum/large-tile rules (see soft_margin_loss):
    #  - without buffer_size_limit, tl.sum is only complete for BLOCK <= 8192;
    #  - with buffer_size_limit=2048, BLOCK == 32768 is complete.
    # So for N % 32768 == 0 we use BLOCK=32768 + buffer_size_limit=2048 (fewer
    # programs, smaller `mid`, less final-reduce work) and for everything else
    # BLOCK = 8192 with a masked tail.
    if n_elements % 32768 == 0:
        block_size = 32768
        buffer_limit = 2048
        need_mask = False
    else:
        block_size = 8192
        buffer_limit = None
        need_mask = n_elements % 8192 != 0
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = min(triton.next_power_of_2(mid_size), 8192)

    # empty_strided (NOT torch.empty): sketch buffers only, native allocator.
    mid = torch.empty_strided(
        (mid_size,), (1,), dtype=torch.float32, device=input.device
    )
    out = torch.empty_strided((), (), dtype=input.dtype, device=input.device)

    with torch_device_fn.device(input.device):
        kw = {}
        if buffer_limit is not None:
            kw["buffer_size_limit"] = buffer_limit
        _hinge_embedding_loss_partial_sum_kernel[(mid_size, 1, 1)](
            input,
            target,
            mid,
            n_elements,
            margin,
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
            **kw,
        )
        _hinge_embedding_loss_final_reduce_kernel[(1, 1, 1)](
            mid,
            out,
            mid_size,
            n_elements,
            red,
            BLOCK_MID=block_mid,
            **kw,
        )

    return out
