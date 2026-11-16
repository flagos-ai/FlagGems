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

"""Kunlunxin XPU3 override for ``aten::l1_loss``.

Generic ``src/flag_gems/ops/l1_loss.py`` launches the stage-2 reduction with a
32768-lane ``tl.sum`` tile but *without* ``buffer_size_limit``; on XPU3 that
single-tile sum is silently incomplete (on ``[200, 40999, 3]`` it returned
~0 / garbage instead of the true sum), which is the functional failure this
override fixes.  It also used a 1024-lane masked stage-1 tile for reduction,
which is ~20x slower than torch on the large shapes.

The reduction path mirrors the proven ``_kunlunxin/ops/mse_loss.py`` design:
unmasked full-lane stage-1 tiles on the block-DMA path, a single in-bounds
window program for the residue, and an fp32 stage-2 ``tl.sum`` launched with
``buffer_size_limit=2048`` (the largest exact ``tl.sum`` tile on this backend).
The elementwise (``reduction=none``) path reuses the generic pointwise kernel,
which already passes and is not on the perf-critical reduction path.
"""

import logging
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.ops.l1_loss import _l1_loss_elementwise_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _l1_partial_kernel(
    inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr
):
    # Full lane-blocks only (grid = M // BLOCK_SIZE): every lane is in-bounds,
    # so both loads are plain/unconditional.  Masked stage-1 tiles at BLOCK >=
    # 32768 run at a fraction of the unmasked bandwidth on XPU3 (masked memory
    # is a non-coalesced slow path), and a masked load does not reliably keep
    # out-of-range lanes out of the reduction -- hence the unmasked form here
    # and the residue handled separately by _l1_tail_kernel.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + offset).to(tl.float32)
    target_val = tl.load(target + offset).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    # Reduction.MEAN.value == 1, Reduction.SUM.value == 2
    if reduction == 1:
        acc = tl.sum(diff) / M
    else:
        acc = tl.sum(diff)
    tl.store(mid + pid, acc)


@libentry()
@triton.jit
def _l1_tail_kernel(
    inp,
    target,
    mid,
    M,
    base,
    n_keep,
    g,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
):
    # Reduces the in-bounds window [M - BLOCK_SIZE, M).  Its trailing `n_keep`
    # lanes are the elements past the full blocks; the leading
    # (BLOCK_SIZE - n_keep) lanes were already counted by _l1_partial_kernel and
    # are zeroed BY VALUE (tl.where selects values, never addresses).  Exact for
    # any tensor and never reads out of range.
    lane = tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + base + lane).to(tl.float32)
    target_val = tl.load(target + base + lane).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    diff = tl.where(lane >= BLOCK_SIZE - n_keep, diff, 0.0)
    if reduction == 1:
        acc = tl.sum(diff) / M
    else:
        acc = tl.sum(diff)
    tl.store(mid + g, acc)


@libentry()
@triton.jit
def _l1_final_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    # `mid` is allocated BLOCK_MID slots but only the first `mid_size` hold
    # partials; padding lanes are dropped BY VALUE (never an address mask).
    offset = tl.arange(0, BLOCK_MID)
    mid_val = tl.load(mid + offset).to(tl.float32)
    mid_val = tl.where(offset < mid_size, mid_val, 0.0)
    tl.store(out, tl.sum(mid_val))


@libentry()
@triton.jit
def _l1_small_kernel(
    inp,
    target,
    out,
    M,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    reduction: tl.constexpr,
):
    # Single-program whole-tensor reduction for small numel.  When M is a power
    # of two the load is unmasked; otherwise a value-predicated masked load
    # (single `other`, then tl.where) is used -- safe on this backend and the
    # launch/per-program cost dominates for tiny tensors.
    off = tl.arange(0, BLOCK)
    if MASKED:
        mask = off < M
        inp_val = tl.load(inp + off, mask=mask, other=0.0).to(tl.float32)
        target_val = tl.load(target + off, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, tl.abs(inp_val - target_val), 0.0)
    else:
        inp_val = tl.load(inp + off).to(tl.float32)
        target_val = tl.load(target + off).to(tl.float32)
        diff = tl.abs(inp_val - target_val)
    if reduction == 1:
        acc = tl.sum(diff) / M
    else:
        acc = tl.sum(diff)
    tl.store(out, acc)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# Largest unmasked stage-1 tile verified exact on XPU3 with
# buffer_size_limit=2048 for the two-load + tl.sum pattern (see the same
# constants in _kunlunxin/ops/mse_loss.py).  Fully divisible tensors get it.
_FULL_BLOCK = 131072
# Secondary exact tile: 2 programs of 32768 lanes beat 1 program of 65536 lanes
# on 65536-element shapes (cluster scheduling).
_V2_BLOCK = 32768
# Full-block size for non-divisible big shapes (fastest remainder pattern).
_LEGACY_BLOCK = 65536
# Stage-2 must stay inside the 32768-lane exact tl.sum ceiling.
_MAX_MID = 32768
# buffer_size_limit for every launch: the exact tl.sum tile ceiling here.
_BUFFER_SIZE_LIMIT = 2048


def _normalize_reduction(reduction):
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


def l1_loss(input: torch.Tensor, target: torch.Tensor, reduction=Reduction.MEAN.value):
    logger.debug("GEMS_KUNLUNXIN L1_LOSS")
    reduction = _normalize_reduction(reduction)
    if input.device != target.device:
        raise AssertionError("l1_loss: input and target must be on the same device.")

    input, target = torch.broadcast_tensors(input, target)
    input = input.contiguous()
    target = target.contiguous()

    if reduction == Reduction.NONE.value:
        # The generic pointwise kernel already passes on XPU3 (broadcast /
        # strided handled by pointwise_dynamic) and reduction=none is not on the
        # benchmark's reduction path.
        return _l1_loss_elementwise_kernel(input, target)

    M = input.numel()
    dtype = input.dtype

    if M == 0:
        # Follow PyTorch behavior: sum -> 0, mean -> NaN.
        if reduction == Reduction.SUM.value:
            return torch.zeros((), device=input.device, dtype=dtype)
        return torch.full((), float("nan"), device=input.device, dtype=dtype)

    out = torch.empty((), dtype=dtype, device=input.device)

    if M <= _FULL_BLOCK:
        # One program covers the whole tensor.
        block = triton.next_power_of_2(M)
        masked = block != M
        with torch_device_fn.device(input.device):
            _l1_small_kernel[(1,)](
                input,
                target,
                out,
                M,
                BLOCK=block,
                MASKED=masked,
                reduction=reduction,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        return out

    # Large tensor: pick an unmasked stage-1 tile.
    if M % _FULL_BLOCK == 0 and (M // _FULL_BLOCK) <= _MAX_MID:
        block = _FULL_BLOCK
    elif M % _V2_BLOCK == 0 and (M // _V2_BLOCK) <= _MAX_MID:
        block = _V2_BLOCK
    else:
        block = get_block_size_1d(M, input.element_size() * 2)
        if M > _LEGACY_BLOCK:
            block = _LEGACY_BLOCK
        if triton.cdiv(M, block) > _MAX_MID:
            block = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))

    n_full = M // block
    rem = M - n_full * block
    mid_size = n_full + (1 if rem else 0)
    block_mid = triton.next_power_of_2(mid_size)

    # BLOCK_MID slots: stage 2 reduces the whole tile unmasked and drops the
    # padding lanes by value.
    mid = torch.empty((block_mid,), dtype=torch.float32, device=input.device)

    with torch_device_fn.device(input.device):
        _l1_partial_kernel[(n_full,)](
            input,
            target,
            mid,
            M,
            BLOCK_SIZE=block,
            reduction=reduction,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
        if rem:
            _l1_tail_kernel[(1,)](
                input,
                target,
                mid,
                M,
                M - block,
                rem,
                n_full,
                BLOCK_SIZE=block,
                reduction=reduction,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        if mid_size == 1:
            return mid.reshape([]).to(dtype)
        _l1_final_kernel[(1,)](
            mid,
            out,
            mid_size,
            BLOCK_MID=block_mid,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
    return out
