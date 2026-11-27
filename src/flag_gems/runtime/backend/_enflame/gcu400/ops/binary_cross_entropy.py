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

logger = logging.getLogger(__name__)

# GCU400's grid.x hardware limit is 65535; a fixed 48-way program set covers the
# 4 DIE x 6 SIP x 8 sub-thread topology, and the kernel strides over all tiles.
GRID_DIM_X = 48
# Element count of a single tile; tuned tier by tier across iterations.
BLOCK_SIZE = 4096
NUM_WARPS = 4
# The tl.arange length in the finalize kernel must be a power of two and no less than the program count.
PARTIAL_BLOCK = triton.next_power_of_2(GRID_DIM_X)


@triton.jit
def _binary_cross_entropy_elementwise_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_DIM_X: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    # Fixed grid + stride loop: program count is decoupled from problem size, avoiding the grid.x limit on large shapes.
    for tile_id in tl.range(pid, num_tiles, GRID_DIM_X):
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

        xf = x.to(tl.float32)
        yf = y.to(tl.float32)

        one_minus_y = 1.0 - yf

        # PyTorch clamps the log values to a lower bound of -100 (not the input),
        # so input==0/1 with the opposite target yields exactly 100.
        log_x = tl.maximum(tl.log(xf), -100.0)
        log_one_minus_x = tl.maximum(tl.log(1.0 - xf), -100.0)

        # BCE: -(target * log(input) + (1 - target) * log(1 - input))
        term1 = yf * log_x
        term2 = one_minus_y * log_one_minus_x
        vals = -(term1 + term2)

        tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _binary_cross_entropy_weighted_elementwise_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_DIM_X: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in tl.range(pid, num_tiles, GRID_DIM_X):
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        wf = w.to(tl.float32)

        one_minus_y = 1.0 - yf

        # PyTorch clamps the log values to a lower bound of -100 (not the input),
        # so input==0/1 with the opposite target yields exactly 100.
        log_x = tl.maximum(tl.log(xf), -100.0)
        log_one_minus_x = tl.maximum(tl.log(1.0 - xf), -100.0)

        # BCE: -(target * log(input) + (1 - target) * log(1 - input))
        term1 = yf * log_x
        term2 = one_minus_y * log_one_minus_x
        vals = -(term1 + term2) * wf

        tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _binary_cross_entropy_sum_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_DIM_X: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    # Each program performs one atomic_add after its loop, compressing the original per-tile
    # global atomic contention into GRID_DIM_X operations.
    acc = tl.full((), 0.0, tl.float32)

    for tile_id in tl.range(pid, num_tiles, GRID_DIM_X):
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

        xf = x.to(tl.float32)
        yf = y.to(tl.float32)

        one_minus_y = 1.0 - yf

        # PyTorch clamps the log values to a lower bound of -100 (not the input),
        # so input==0/1 with the opposite target yields exactly 100.
        log_x = tl.maximum(tl.log(xf), -100.0)
        log_one_minus_x = tl.maximum(tl.log(1.0 - xf), -100.0)

        # BCE: -(target * log(input) + (1 - target) * log(1 - input))
        term1 = yf * log_x
        term2 = one_minus_y * log_one_minus_x
        vals = -(term1 + term2)
        vals = tl.where(mask, vals, 0.0)

        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + pid, acc)


@triton.jit
def _binary_cross_entropy_weighted_sum_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_DIM_X: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    acc = tl.full((), 0.0, tl.float32)

    for tile_id in tl.range(pid, num_tiles, GRID_DIM_X):
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

        xf = x.to(tl.float32)
        yf = y.to(tl.float32)
        wf = w.to(tl.float32)

        one_minus_y = 1.0 - yf

        # PyTorch clamps the log values to a lower bound of -100 (not the input),
        # so input==0/1 with the opposite target yields exactly 100.
        log_x = tl.maximum(tl.log(xf), -100.0)
        log_one_minus_x = tl.maximum(tl.log(1.0 - xf), -100.0)

        # BCE: -(target * log(input) + (1 - target) * log(1 - input))
        term1 = yf * log_x
        term2 = one_minus_y * log_one_minus_x
        vals = -(term1 + term2) * wf
        vals = tl.where(mask, vals, 0.0)

        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + pid, acc)


@triton.jit
def _binary_cross_entropy_finalize_kernel(
    partial_ptr,
    out_ptr,
    n_elements,
    num_partials,
    REDUCTION: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    # One program merges all programs' partial sums and completes
    # mean normalization and output dtype conversion in the same store, avoiding an extra div / cast kernel.
    partial_offsets = tl.arange(0, BLOCK_PARTIALS)
    partial_sums = tl.load(
        partial_ptr + partial_offsets,
        mask=partial_offsets < num_partials,
        other=0.0,
    )
    total = tl.sum(partial_sums, axis=0)
    if REDUCTION == 1:
        total = total / n_elements
    tl.store(out_ptr, total)


def _normalize_reduction(reduction):
    # Accept both string and enum/int forms: 0=none, 1=mean, 2=sum
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


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    if not (input.is_cuda and target.is_cuda):
        raise AssertionError(
            "binary_cross_entropy: input and target must be CUDA tensors for Triton kernel."
        )
    if input.device != target.device:
        raise AssertionError(
            "binary_cross_entropy: input and target must be on the same device."
        )
    if input.numel() != target.numel():
        raise AssertionError(
            "binary_cross_entropy: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def _prepare_weight(weight, input, n_elements):
    # Returns a contiguous, per-element weight tensor (or None).
    if weight is None:
        return None
    weight = weight.contiguous()
    if weight.numel() == 1:
        # Scalar weight: broadcast to all elements.
        return torch.full(
            (n_elements,),
            weight.item(),
            device=input.device,
            dtype=torch.float32,
        )
    if weight.numel() != n_elements:
        raise AssertionError(
            "binary_cross_entropy: weight must have same number "
            "of elements as input and target, or be a scalar."
        )
    return weight


def _bce_grid(n_elements, block_size):
    # Program count has a fixed upper bound while guaranteeing each program gets at least one tile.
    return (min(GRID_DIM_X, triton.cdiv(n_elements, block_size)),)


def _bce_elementwise(input, target, weight, n_elements, out=None):
    # The out variant writes elementwise results directly into the caller's buffer, saving a temporary tensor and an
    # extra copy.
    result = torch.empty_like(input) if out is None else out
    if n_elements == 0:
        return result
    grid = _bce_grid(n_elements, BLOCK_SIZE)
    if weight is None:
        _binary_cross_entropy_elementwise_kernel[grid](
            input,
            target,
            result,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            GRID_DIM_X=GRID_DIM_X,
            num_warps=NUM_WARPS,
        )
    else:
        _binary_cross_entropy_weighted_elementwise_kernel[grid](
            input,
            target,
            weight,
            result,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            GRID_DIM_X=GRID_DIM_X,
            num_warps=NUM_WARPS,
        )
    return result


def _bce_reduced(input, target, weight, red, n_elements, out=None):
    # Computes the sum/mean BCE and returns a scalar tensor.
    if n_elements == 0:
        # Follow PyTorch behavior: sum -> 0, mean -> NaN
        empty_value = 0.0 if red == 2 else float("nan")
        if out is None:
            return torch.full((), empty_value, device=input.device, dtype=input.dtype)
        out.fill_(empty_value)
        return out

    grid = _bce_grid(n_elements, BLOCK_SIZE)
    num_partials = grid[0]
    # The uninitialized partial buffer is filled with valid prefixes per program by the BCE kernel,
    # so a separate zeros kernel is unnecessary.
    partial_sums = torch.empty(
        (PARTIAL_BLOCK,), device=input.device, dtype=torch.float32
    )
    result = (
        torch.empty((), device=input.device, dtype=input.dtype) if out is None else out
    )

    if weight is None:
        _binary_cross_entropy_sum_kernel[grid](
            input,
            target,
            partial_sums,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            GRID_DIM_X=GRID_DIM_X,
            num_warps=NUM_WARPS,
        )
    else:
        _binary_cross_entropy_weighted_sum_kernel[grid](
            input,
            target,
            weight,
            partial_sums,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            GRID_DIM_X=GRID_DIM_X,
            num_warps=NUM_WARPS,
        )

    _binary_cross_entropy_finalize_kernel[(1,)](
        partial_sums,
        result,
        n_elements,
        num_partials,
        REDUCTION=red,
        BLOCK_PARTIALS=PARTIAL_BLOCK,
        num_warps=4,
    )
    return result


def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: str = "mean",
):
    logger.debug("GEMS_ENFLAME BINARY_CROSS_ENTROPY")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()
    weight = _prepare_weight(weight, input, n_elements)

    if red == 0:
        return _bce_elementwise(input, target, weight, n_elements)
    return _bce_reduced(input, target, weight, red, n_elements)


def binary_cross_entropy_out(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: str = "mean",
    out: torch.Tensor = None,
):
    logger.debug("GEMS_ENFLAME BINARY_CROSS_ENTROPY_OUT")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()
    weight = _prepare_weight(weight, input, n_elements)

    if out is None:
        # Allocate output based on reduction.
        if red == 0:
            out = torch.empty_like(input)
        else:
            out = torch.empty((), device=input.device, dtype=input.dtype)
    else:
        if not out.is_cuda:
            raise AssertionError("binary_cross_entropy_out: out must be a CUDA tensor.")
        if red == 0:
            if out.numel() != n_elements:
                raise AssertionError(
                    "binary_cross_entropy_out: for reduction='none', out must match input shape."
                )
        else:
            if out.numel() != 1:
                raise AssertionError(
                    "binary_cross_entropy_out: for reduction='sum' or 'mean', out must be a scalar tensor."
                )
        if out.device != input.device:
            raise AssertionError(
                "binary_cross_entropy_out: out must be on the same device as input."
            )

    # The kernel writes directly into the caller's out, preserving the out variant's "return the same object" semantics
    # while saving a temporary result tensor and an extra copy_ kernel.
    if red == 0:
        _bce_elementwise(input, target, weight, n_elements, out=out)
    else:
        _bce_reduced(input, target, weight, red, n_elements, out=out)
    return out
