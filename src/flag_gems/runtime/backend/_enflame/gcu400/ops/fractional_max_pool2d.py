import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# Hardware limit of grid dimension x on GCU400
MAX_GRID_X = 65535

# Curated autotune configuration set (replacing the default 3x3x3x3=81 grid):
# Output planes are generally <= 32x32, so BLOCK covering the whole plane suffices; num_warps is determined empirically
# -
# forward is best at nw2, while backward needs more warps to advance because of atomic_add.
# The compiler must generate a separate version for each (BLOCK_H, BLOCK_W, num_warps, num_stages); converging the
# configuration
# count can reduce the number of compilations for a single shape from ~81 to ~9.
_FWD_CONFIGS = [
    triton.Config({"BLOCK_H": block_h, "BLOCK_W": block_w}, num_warps=2, num_stages=3)
    for block_h in (8, 16, 32)
    for block_w in (8, 16, 32)
] + [
    # After the kernel shape changed (mask reduction, int32 indices), give autotune two more higher-parallelism
    # candidates: on large planes nw4 has a chance of beating nw2.
    triton.Config({"BLOCK_H": block_h, "BLOCK_W": block_w}, num_warps=4, num_stages=3)
    for block_h, block_w in ((16, 16), (32, 16), (32, 32))
]
_BWD_CONFIGS = [
    triton.Config(
        {"BLOCK_H": block_h, "BLOCK_W": block_w}, num_warps=warps, num_stages=4
    )
    for block_h, block_w, warps in (
        (8, 8, 8),
        (8, 16, 8),
        (16, 16, 2),
        (16, 16, 4),
        (16, 16, 8),
        (32, 16, 4),
        (32, 32, 2),
        (32, 32, 4),
        (32, 32, 8),
    )
]


@libentry()
@triton.autotune(
    configs=_FWD_CONFIGS,
    key=["out_h", "out_w", "kernel_h", "kernel_w"],
)
@triton.jit
def fractional_max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Total number of (n, c) planes (in_n * in_c)
    num_nc,
    # Precomputed alpha values for interval generation
    alpha_h,
    alpha_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Forward kernel for fractional max pool 2d.

    Computes pooling region start positions inline from random_samples,
    using the formula: start[i] = int((i + sample) * alpha) - int(sample * alpha).
    This eliminates precomputed h_starts/w_starts arrays and their Python overhead.

    Grid axis 0 covers the (n, c) planes and is capped at MAX_GRID_X; when
    num_nc exceeds the grid size, each program strides over multiple planes.
    """
    pid_nc_start = tl.program_id(0)
    grid_nc = tl.num_programs(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)

    h_mask = h_out_offsets < out_h
    w_mask = w_out_offsets < out_w
    tile_mask = h_mask[:, None] & w_mask[None, :]

    for pid_nc in range(pid_nc_start, num_nc, grid_nc):
        n_idx = pid_nc // in_c
        c_idx = pid_nc % in_c

        input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

        # Load random samples for this (n, c) plane
        # random_samples layout: [N*C, 2] contiguous
        sample_h = tl.load(random_samples_ptr + pid_nc * 2).to(tl.float32)
        sample_w = tl.load(random_samples_ptr + pid_nc * 2 + 1).to(tl.float32)

        # Compute start positions inline
        sample_alpha_h = (sample_h * alpha_h).to(tl.int32)
        sample_alpha_w = (sample_w * alpha_w).to(tl.int32)

        h_starts = ((h_out_offsets.to(tl.float32) + sample_h) * alpha_h).to(
            tl.int32
        ) - sample_alpha_h
        w_starts = ((w_out_offsets.to(tl.float32) + sample_w) * alpha_w).to(
            tl.int32
        ) - sample_alpha_w

        # Override last position: start[output_size - 1] = input_size - pool_size
        h_starts = tl.where(h_out_offsets == out_h - 1, in_h - kernel_h, h_starts)
        w_starts = tl.where(w_out_offsets == out_w - 1, in_w - kernel_w, w_starts)

        max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
        # Indices are carried only as int32 inside the kernel (saving the i64 vector ALU); at the final store
        # convert to int64 in one shot, avoiding another cast kernel on the host side.
        max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int32)

        h_base = h_starts[:, None]
        w_base = w_starts[None, :]

        # Strength reduction: compute the address/index base within a tile only once, each kernel tap does only a scalar
        # add
        offset_base = h_base * in_stride_h + w_base * in_stride_w
        index_base = h_base * in_w + w_base

        # Use the 0th tap directly as the initial value, saving one compare + two selects;
        # Only record the index when the value is strictly greater than the dtype minimum, keeping semantics exactly the
        # same as "initialize with min_val,
        # update only on strictly greater" (the index remains -1 for extreme-value inputs).
        first_mask = h_mask[:, None] & w_mask[None, :] & (h_base >= 0) & (h_base < in_h)
        first_mask = first_mask & (w_base >= 0) & (w_base < in_w)
        first_val = tl.load(
            input_base_ptr + offset_base, mask=first_mask, other=min_val
        )
        max_val_acc = first_val
        max_idx_acc = tl.where(first_mask & (first_val > min_val), index_base, -1)

        for tap in range(1, kernel_h * kernel_w):
            kh = tap // kernel_w
            kw = tap % kernel_w
            # Compute row-direction validity only once on (BLOCK_H, 1), saving the 2D comparison chain for each tap
            h_in = h_base + kh
            h_valid = h_mask[:, None] & (h_in >= 0) & (h_in < in_h)
            w_in = w_base + kw
            in_mask = h_valid & w_mask[None, :] & (w_in >= 0) & (w_in < in_w)

            tap_offset = kh * in_stride_h + kw * in_stride_w
            current_val = tl.load(
                input_base_ptr + (offset_base + tap_offset),
                mask=in_mask,
                other=min_val,
            )
            current_idx = index_base + (kh * in_w + kw)

            update_mask = current_val > max_val_acc
            max_val_acc = tl.where(update_mask, current_val, max_val_acc)
            max_idx_acc = tl.where(update_mask, current_idx, max_idx_acc)

        # Write output and indices
        out_nc_offset = pid_nc * out_h * out_w
        out_offsets = (
            out_nc_offset + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
        )
        tl.store(output_ptr + out_offsets, max_val_acc, mask=tile_mask)
        tl.store(indices_ptr + out_offsets, max_idx_acc.to(tl.int64), mask=tile_mask)


@libentry()
@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["out_h", "out_w"],
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def fractional_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Output strides (element-based, contiguous)
    out_stride_nc,
    out_stride_h,
    out_stride_w,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    # True when windows do not overlap (alpha >= kernel): each input position is written at most once,
    # so a plain store can replace atomic_add.
    USE_ATOMIC: tl.constexpr = True,
):
    """Backward kernel for fractional max pool 2d.

    Scatters gradients back to input positions using the indices from forward.
    Uses atomic_add since multiple output positions may map to the same input.

    Grid axis 0 covers the (n, c) planes and is capped at MAX_GRID_X; when
    num_nc exceeds the grid size, each program strides over multiple planes.
    """
    pid_nc_start = tl.program_id(0)
    grid_nc = tl.num_programs(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    h_mask = h_out_offsets < out_h
    w_mask = w_out_offsets < out_w
    out_mask = h_mask[:, None] & w_mask[None, :]

    num_nc = in_n * in_c
    for pid_nc in range(pid_nc_start, num_nc, grid_nc):
        # Load grad_output and indices for this tile
        out_base = pid_nc * out_stride_nc
        out_offsets = (
            out_base
            + h_out_offsets[:, None] * out_stride_h
            + w_out_offsets[None, :] * out_stride_w
        )

        grad_vals = tl.load(grad_output_ptr + out_offsets, mask=out_mask, other=0.0)
        idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)

        # Scatter gradients to input using indices
        in_nc_offset = pid_nc * in_h * in_w
        in_offsets = in_nc_offset + idx_vals

        valid_mask = out_mask & (idx_vals >= 0)
        if USE_ATOMIC:
            tl.atomic_add(
                grad_input_ptr + in_offsets, grad_vals.to(tl.float32), mask=valid_mask
            )
        else:
            # Windows do not overlap: within the same launch each grad_input element is hit by at most one output
            # position, so a plain store is equivalent to atomic_add with no atomics overhead.
            tl.store(
                grad_input_ptr + in_offsets, grad_vals.to(tl.float32), mask=valid_mask
            )


def fractional_max_pool2d(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = True,
    _random_samples=None,
):
    """Fractional max pooling 2d operation.

    Applies fractional max pooling over a 2D input signal using stochastic
    pooling regions determined by random_samples.

    Args:
        input: Input tensor of shape (N, C, H, W).
        kernel_size: Size of the pooling kernel.
        output_size: Target output spatial size (H_out, W_out).
        output_ratio: Alternative to output_size, specifies output as ratio of input.
        return_indices: Whether to return the max indices.
        _random_samples: Optional tensor of shape (N, C, 2) with values in [0, 1).
    """
    logger.debug("GEMS_ENFLAME FRACTIONAL_MAX_POOL2D")
    assert input.dim() == 4, f"Expected 4D input, got {input.dim()}D"

    in_n, in_c, in_h, in_w = input.shape
    input = input.contiguous()

    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if output_size is not None:
        if isinstance(output_size, int):
            out_h = out_w = output_size
        else:
            out_h, out_w = output_size
    elif output_ratio is not None:
        if isinstance(output_ratio, float):
            out_h = int(in_h * output_ratio)
            out_w = int(in_w * output_ratio)
        else:
            out_h = int(in_h * output_ratio[0])
            out_w = int(in_w * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")

    assert out_h + kernel_h - 1 <= in_h, (
        f"fractional_max_pool2d: kernel_h ({kernel_h}) + output_h ({out_h}) - 1 "
        f"exceeds input_h ({in_h})"
    )
    assert out_w + kernel_w - 1 <= in_w, (
        f"fractional_max_pool2d: kernel_w ({kernel_w}) + output_w ({out_w}) - 1 "
        f"exceeds input_w ({in_w})"
    )

    # Generate or validate random samples
    if _random_samples is None:
        _random_samples = torch.rand(in_n, in_c, 2, device=input.device)
    else:
        assert _random_samples.shape == (in_n, in_c, 2), (
            f"_random_samples must have shape ({in_n}, {in_c}, 2), "
            f"got {_random_samples.shape}"
        )

    # Reshape to [N*C, 2] contiguous for kernel access
    random_samples_flat = _random_samples.reshape(in_n * in_c, 2).contiguous()

    # Precompute alpha values for interval generation
    alpha_h = (in_h - kernel_h) / (out_h - 1) if out_h > 1 else 0.0
    alpha_w = (in_w - kernel_w) / (out_w - 1) if out_w > 1 else 0.0

    output = torch.empty(
        in_n, in_c, out_h, out_w, dtype=input.dtype, device=input.device
    )
    indices = torch.empty(
        in_n, in_c, out_h, out_w, dtype=torch.int64, device=input.device
    )

    if output.numel() == 0:
        return (output, indices) if return_indices else output

    grid = lambda meta: (
        min(in_n * in_c, MAX_GRID_X),
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    fractional_max_pool2d_forward_kernel[grid](
        input,
        output,
        indices,
        random_samples_flat,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        in_n * in_c,
        alpha_h,
        alpha_w,
        kernel_h,
        kernel_w,
    )

    if return_indices:
        return output, indices
    return output


def fractional_max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    output_size,
    indices: torch.Tensor,
):
    """Backward pass for fractional_max_pool2d.

    Scatters gradients back to input positions using the indices from forward.
    """
    logger.debug("GEMS_ENFLAME FRACTIONAL_MAX_POOL2D_BACKWARD")
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    in_n, in_c, in_h, in_w = input.shape

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    # Determine whether windows may overlap: the increment of the (i+1)-th window start relative to the i-th
    # floor((i+1+s)*alpha) - floor((i+s)*alpha) >= floor(alpha),
    # so when alpha >= kernel the increment is always >= kernel, windows are pairwise non-overlapping (at most
    # adjacent),
    # each input position is hit by at most one output position, so atomics can be avoided.
    alpha_h = (in_h - kernel_h) / (out_h - 1) if out_h > 1 else 0.0
    alpha_w = (in_w - kernel_w) / (out_w - 1) if out_w > 1 else 0.0
    windows_disjoint = alpha_h >= kernel_h and alpha_w >= kernel_w
    # When the output grid exactly tiles the input (out * kernel == in) and windows do not overlap, each input element
    # is guaranteed to be written once, so the fill kernel for zeros initialization can be omitted.
    fully_covered = (
        windows_disjoint and in_h == out_h * kernel_h and in_w == out_w * kernel_w
    )

    if fully_covered:
        grad_input = torch.empty_like(input, dtype=torch.float32)
    else:
        grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        min(in_n * in_c, MAX_GRID_X),
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    out_stride_nc = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    fractional_max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        out_stride_nc,
        out_stride_h,
        out_stride_w,
        USE_ATOMIC=not windows_disjoint,
    )

    return grad_input.to(grad_output.dtype)
