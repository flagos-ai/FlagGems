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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Unmasked stage-1 tile: largest `tl.sum` tile that is complete on this target
# with buffer_size_limit=2048 (same ceiling used by the other reduction kernels).
_FULL_BLOCK = 32768
# Masked stage-1 tile for sizes that do not divide _FULL_BLOCK.
_MASKED_BLOCK = 2048
# Tensors up to this size use the single-launch mean/sum kernel.
_SMALL_MAX = 4096
# Lane width of the compensated cross-partial final reduction.
_FINAL_BLOCK = 32


@libentry()
@triton.jit
def _l1_loss_kernel(
    input,
    target,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    loss = tl.abs(x - y)
    tl.store(out + offsets, loss.to(out.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _l1_loss_partial_sum_unmasked_kernel(
    input,
    target,
    partial_sums,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Unmasked full-block program over the leading floor(n/BLOCK_SIZE)*BLOCK_SIZE
    # elements. Masked `tl.sum` reduces with extra rounding on this target, so
    # the bulk is always taken by this unmasked path and only the small tail
    # (< BLOCK_SIZE elements) goes through the masked tail kernel.
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + offsets).to(tl.float32)
    y = tl.load(target + offsets).to(tl.float32)
    tl.store(partial_sums + pid, tl.sum(tl.abs(x - y), axis=0))


@libentry()
@triton.jit
def _l1_loss_tail_kernel(
    input,
    target,
    partial_sums,
    base,
    r,
    off,
    BLOCK_SIZE: tl.constexpr,
):
    # Masked reduction over the tail remainder (r < BLOCK_SIZE * n_programs),
    # writing its partials after the unmasked partials.
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < r
    x = tl.load(input + base + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + base + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        partial_sums + off + pid,
        tl.sum(tl.where(mask, tl.abs(x - y), 0.0), axis=0),
    )


@libentry()
@triton.jit
def _l1_loss_small_kernel(
    input,
    target,
    out,
    n_elements,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single-launch mean/sum for small tensors, where the two-launch
    # partial+final path is launch-bound.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, tl.abs(x - y), 0.0), axis=0)
    if REDUCTION == 1:
        total = total / n_elements
    tl.store(out, total)


@libentry()
@triton.jit
def _l1_loss_final_kernel(
    partial_sums,
    out,
    n_partials,
    n_elements,
    REDUCTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Cross-partial combine with Kahan (compensated) summation in fp32. A plain
    # fp32 tree over ~1e4 partials accumulates ~1e-5 relative error on the large
    # reductions, which exceeds the fp32 test tolerance; the compensation drops
    # it to roughly machine epsilon.
    offs = tl.arange(0, BLOCK)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    c = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_partials, BLOCK):
        idx = start + offs
        v = tl.load(partial_sums + idx, mask=idx < n_partials, other=0.0)
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    total = tl.sum(s, axis=0)
    if REDUCTION == 1:
        total = total / n_elements
    tl.store(out, total)


_REDUCTION_MAP = {"none": 0, "mean": 1, "sum": 2}


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r not in _REDUCTION_MAP:
            raise ValueError(f"Invalid reduction: {reduction}")
        return _REDUCTION_MAP[r]
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _broadcast_shape(a, b):
    """Elementwise broadcast shape, computed from the sizes alone."""
    rank = max(len(a), len(b))
    pa = [1] * (rank - len(a)) + list(a)
    pb = [1] * (rank - len(b)) + list(b)
    out = []
    for da, db in zip(pa, pb):
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise RuntimeError(
                "The size of tensor a (%d) must match the size of tensor b (%d) "
                "at non-singleton dimension" % (da, db)
            )
    return tuple(out)


def l1_loss(input: torch.Tensor, target: torch.Tensor, reduction=1) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN L1_LOSS")
    red = _normalize_reduction(reduction)
    if input.device != target.device:
        raise AssertionError("l1_loss: input and target must be on the same device.")

    shape = _broadcast_shape(input.shape, target.shape)
    if tuple(input.shape) != shape:
        input = input.expand(shape)
    if tuple(target.shape) != shape:
        target = target.expand(shape)
    input = input.contiguous()
    target = target.contiguous()
    n_elements = input.numel()
    dtype = torch.result_type(input, target)

    if red == 0:
        out = torch.empty(input.shape, dtype=dtype, device=input.device)
        if n_elements == 0:
            return out
        grid = (triton.cdiv(n_elements, _MASKED_BLOCK),)
        with torch_device_fn.device(input.device):
            _l1_loss_kernel[grid](
                input, target, out, n_elements, BLOCK_SIZE=_MASKED_BLOCK
            )
        return out

    if n_elements == 0:
        # Follow ATen: sum -> 0, mean -> NaN
        if red == 2:
            return torch.zeros((), dtype=dtype, device=input.device)
        return torch.full((), float("nan"), dtype=dtype, device=input.device)

    out = torch.empty((), dtype=dtype, device=input.device)

    if n_elements <= _SMALL_MAX:
        block = max(triton.next_power_of_2(n_elements), 1)
        with torch_device_fn.device(input.device):
            _l1_loss_small_kernel[(1,)](
                input,
                target,
                out,
                n_elements,
                REDUCTION=red,
                BLOCK_SIZE=block,
            )
        return out

    blk = _FULL_BLOCK
    n_main = n_elements // blk
    rem = n_elements - n_main * blk
    n_tail = triton.cdiv(rem, _MASKED_BLOCK) if rem > 0 else 0
    n_partials = n_main + n_tail

    partial_sums = torch.empty((n_partials,), dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        if n_main > 0:
            _l1_loss_partial_sum_unmasked_kernel[(n_main,)](
                input,
                target,
                partial_sums,
                n_elements,
                BLOCK_SIZE=blk,
                buffer_size_limit=2048,
            )
        if n_tail > 0:
            _l1_loss_tail_kernel[(n_tail,)](
                input,
                target,
                partial_sums,
                n_main * blk,
                rem,
                n_main,
                BLOCK_SIZE=_MASKED_BLOCK,
            )
        _l1_loss_final_kernel[(1,)](
            partial_sums,
            out,
            n_partials,
            n_elements,
            REDUCTION=red,
            BLOCK=_FINAL_BLOCK,
        )
    return out
