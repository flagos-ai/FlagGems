import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _select_params(n):
    if n <= 256:
        return 256, 2
    if n <= 1024:
        return 256, 4
    if n <= 4096:
        return 512, 4
    return 1024, 4


def _estimate_output_size(n, minlength):
    estimate = max(8192, n * 4, minlength)
    estimate = min(estimate, 65536)
    return max(estimate, minlength)


def _select_max_block_size(n):
    return triton.next_power_of_2(max(1, math.ceil(math.sqrt(n))))


def _select_bins_block(output_size):
    return min(128, triton.next_power_of_2(max(1, output_size)))


@triton.jit
def fused_max_bincount_kernel(
    input_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)

    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)

    safe_mask = mask & (vals < output_size)
    tl.atomic_add(output_ptr + vals, 1, mask=safe_mask)


@triton.jit
def bincount_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(output_ptr + vals, 1, mask=mask)


@triton.jit
def fused_max_bincount_weights_fp32_kernel(
    input_ptr,
    weights_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp32 = w.to(tl.float32)

    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)

    safe_mask = mask & (vals < output_size)
    tl.atomic_add(output_ptr + vals, w_fp32, mask=safe_mask)


@triton.jit
def fused_max_bincount_weights_fp64_kernel(
    input_ptr,
    weights_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp64 = w.to(tl.float64)

    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)

    safe_mask = mask & (vals < output_size)
    tl.atomic_add(output_ptr + vals, w_fp64, mask=safe_mask)


@triton.jit
def bincount_max_kernel_1(
    input_ptr,
    mid_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    local_max = tl.max(vals, axis=0)
    tl.store(mid_ptr + pid, local_max)


@triton.jit
def bincount_max_kernel_2(
    mid_ptr,
    max_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size
    mid_vals = tl.load(mid_ptr + offsets, mask=mask, other=0)
    max_val = tl.max(mid_vals, axis=0)
    tl.store(max_ptr, max_val)


@triton.jit
def bincount_weights_fp32_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp32 = w.to(tl.float32)
    tl.atomic_add(output_ptr + vals, w_fp32, mask=mask)


@triton.jit
def bincount_weights_fp64_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp64 = w.to(tl.float64)
    tl.atomic_add(output_ptr + vals, w_fp64, mask=mask)


@triton.jit
def bincount_partial_int64_kernel(
    input_ptr,
    partial_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
    TILE_INPUT: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_bin = tl.program_id(1)

    block_start = pid_block * BLOCK_SIZE
    bin_offsets = pid_bin * BLOCK_BINS + tl.arange(0, BLOCK_BINS)
    bin_mask = bin_offsets < output_size
    acc = tl.zeros([BLOCK_BINS], dtype=tl.int32)

    for tile_start in range(0, BLOCK_SIZE, TILE_INPUT):
        input_offsets = block_start + tile_start + tl.arange(0, TILE_INPUT)
        input_mask = input_offsets < n_elements
        vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0)
        bins = bin_offsets.to(vals.dtype)
        matches = (
            bin_mask[:, None] & input_mask[None, :] & (bins[:, None] == vals[None, :])
        )
        acc += tl.sum(matches.to(tl.int32), axis=1)

    partial_offsets = pid_block * output_size + bin_offsets
    tl.store(partial_ptr + partial_offsets, acc.to(tl.int64), mask=bin_mask)


@triton.jit
def bincount_partial_weights_fp64_kernel(
    input_ptr,
    weights_ptr,
    partial_ptr,
    n_elements,
    output_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
    TILE_INPUT: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_bin = tl.program_id(1)

    block_start = pid_block * BLOCK_SIZE
    bin_offsets = pid_bin * BLOCK_BINS + tl.arange(0, BLOCK_BINS)
    bin_mask = bin_offsets < output_size
    acc = tl.zeros([BLOCK_BINS], dtype=tl.float64)

    for tile_start in range(0, BLOCK_SIZE, TILE_INPUT):
        input_offsets = block_start + tile_start + tl.arange(0, TILE_INPUT)
        input_mask = input_offsets < n_elements
        vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0)
        w = tl.load(weights_ptr + input_offsets, mask=input_mask, other=0.0).to(
            tl.float64
        )
        bins = bin_offsets.to(vals.dtype)
        matches = (
            bin_mask[:, None] & input_mask[None, :] & (bins[:, None] == vals[None, :])
        )
        acc += tl.sum(tl.where(matches, w[None, :], 0.0), axis=1)

    partial_offsets = pid_block * output_size + bin_offsets
    tl.store(partial_ptr + partial_offsets, acc, mask=bin_mask)


@triton.jit
def bincount_reduce_partial_kernel(
    partial_ptr,
    output_ptr,
    num_partials,
    output_size,
    BLOCK_PARTIAL: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
):
    pid_bin = tl.program_id(0)
    bin_offsets = pid_bin * BLOCK_BINS + tl.arange(0, BLOCK_BINS)
    bin_mask = bin_offsets < output_size
    acc = tl.zeros([BLOCK_BINS], dtype=output_ptr.dtype.element_ty)

    for partial_start in range(0, num_partials, BLOCK_PARTIAL):
        partial_rows = partial_start + tl.arange(0, BLOCK_PARTIAL)
        partial_ptrs = (
            partial_ptr + partial_rows[:, None] * output_size + bin_offsets[None, :]
        )
        partial_mask = (partial_rows[:, None] < num_partials) & bin_mask[None, :]
        partial_vals = tl.load(partial_ptrs, mask=partial_mask, other=0)
        acc += tl.sum(partial_vals, axis=0)

    tl.store(output_ptr + bin_offsets, acc, mask=bin_mask)


def _compute_output_size(input_contig, n, minlength):
    max_block_size = _select_max_block_size(n)
    mid_size = triton.cdiv(n, max_block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=input_contig.dtype, device=input_contig.device)
    max_tensor = torch.empty([], dtype=input_contig.dtype, device=input_contig.device)

    with torch_device_fn.device(input_contig.device):
        bincount_max_kernel_1[(mid_size, 1, 1)](
            input_contig,
            mid,
            n,
            BLOCK_SIZE=max_block_size,
        )
        bincount_max_kernel_2[(1, 1, 1)](
            mid,
            max_tensor,
            mid_size,
            BLOCK_MID=block_mid,
        )

    return max(int(max_tensor.item()) + 1, minlength)


def _bincount_atomic_launch(
    input_contig,
    weights_contig,
    n,
    output_size,
    BLOCK_SIZE,
    num_warps,
):
    output = torch.zeros(output_size, dtype=torch.float32, device=input_contig.device)
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    with torch_device_fn.device(input_contig.device):
        bincount_weights_fp32_kernel[grid](
            input_contig,
            weights_contig,
            output,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return output


def _fused_bincount_atomic_launch(
    input_contig,
    weights_contig,
    n,
    pre_size,
    minlength,
    out_dtype,
    grid,
    BLOCK_SIZE,
    num_warps,
):
    max_tensor = torch.zeros(1, dtype=input_contig.dtype, device=input_contig.device)
    is_fp64 = out_dtype == torch.float64
    compute_dtype = torch.float64 if is_fp64 else torch.float32
    output = torch.zeros(pre_size, dtype=compute_dtype, device=input_contig.device)

    with torch_device_fn.device(input_contig.device):
        if is_fp64:
            fused_max_bincount_weights_fp64_kernel[grid](
                input_contig,
                weights_contig,
                max_tensor,
                output,
                n,
                pre_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            fused_max_bincount_weights_fp32_kernel[grid](
                input_contig,
                weights_contig,
                max_tensor,
                output,
                n,
                pre_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

    max_val = int(max_tensor.item())
    needed_size = max(max_val + 1, minlength)

    if needed_size <= pre_size:
        return output[:needed_size]

    if is_fp64:
        output = torch.zeros(
            needed_size, dtype=torch.float64, device=input_contig.device
        )
    else:
        output = torch.zeros(
            needed_size, dtype=torch.float32, device=input_contig.device
        )

    with torch_device_fn.device(input_contig.device):
        if is_fp64:
            bincount_weights_fp64_kernel[grid](
                input_contig,
                weights_contig,
                output,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            bincount_weights_fp32_kernel[grid](
                input_contig,
                weights_contig,
                output,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

    return output


def _bincount_no_atomic_launch(
    input_contig,
    weights_contig,
    n,
    output_size,
    out_dtype,
    BLOCK_SIZE,
    num_warps,
):
    block_bins = _select_bins_block(output_size)
    tile_input = min(64, BLOCK_SIZE)
    num_partials = triton.cdiv(n, BLOCK_SIZE)
    grid = (num_partials, triton.cdiv(output_size, block_bins))

    partial = torch.empty(
        (num_partials, output_size), dtype=out_dtype, device=input_contig.device
    )
    output = torch.empty(output_size, dtype=out_dtype, device=input_contig.device)

    with torch_device_fn.device(input_contig.device):
        if weights_contig is None:
            bincount_partial_int64_kernel[grid](
                input_contig,
                partial,
                n,
                output_size,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_BINS=block_bins,
                TILE_INPUT=tile_input,
                num_warps=num_warps,
            )
        else:
            bincount_partial_weights_fp64_kernel[grid](
                input_contig,
                weights_contig,
                partial,
                n,
                output_size,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_BINS=block_bins,
                TILE_INPUT=tile_input,
                num_warps=num_warps,
            )

        bincount_reduce_partial_kernel[(triton.cdiv(output_size, block_bins), 1, 1)](
            partial,
            output,
            num_partials,
            output_size,
            BLOCK_PARTIAL=8,
            BLOCK_BINS=block_bins,
            num_warps=4,
        )

    return output


def _supports_atomic_accumulate(out_dtype):
    return out_dtype not in (torch.int64, torch.float64)


def _supports_fused_atomic(input_dtype, out_dtype):
    return _supports_atomic_accumulate(out_dtype) and input_dtype == torch.int32


def bincount(input, weights=None, minlength=0):
    logger.debug("GEMS BINCOUNT")

    assert input.dim() == 1, "input must be a 1-D tensor"
    assert minlength >= 0, "minlength must be non-negative"

    if weights is not None:
        assert weights.shape == input.shape, "weights must have the same shape as input"

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    input_contig = input.contiguous()
    weights_contig = weights.contiguous() if weights is not None else None

    if weights is not None and weights.dtype == torch.float64:
        return torch.bincount(
            input_contig.cpu(),
            weights=weights_contig.cpu(),
            minlength=minlength,
        ).to(input.device)

    BLOCK_SIZE, num_warps = _select_params(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    out_dtype = weights.dtype if weights is not None else torch.int64

    if _supports_fused_atomic(input_contig.dtype, out_dtype):
        pre_size = _estimate_output_size(n, minlength)
        output = _fused_bincount_atomic_launch(
            input_contig,
            weights_contig,
            n,
            pre_size,
            minlength,
            out_dtype,
            grid,
            BLOCK_SIZE,
            num_warps,
        )
    elif _supports_atomic_accumulate(out_dtype):
        output_size = _compute_output_size(input_contig, n, minlength)
        output = _bincount_atomic_launch(
            input_contig,
            weights_contig,
            n,
            output_size,
            BLOCK_SIZE,
            num_warps,
        )
    else:
        output_size = _compute_output_size(input_contig, n, minlength)
        output = _bincount_no_atomic_launch(
            input_contig,
            weights_contig,
            n,
            output_size,
            out_dtype,
            BLOCK_SIZE,
            num_warps,
        )

    if (
        weights is not None
        and weights.dtype != torch.float64
        and weights.dtype != torch.float32
    ):
        output = output.to(dtype=weights.dtype)

    return output
