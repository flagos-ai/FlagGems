import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry


@libentry()
@triton.jit
def unique_consecutive_phase1_kernel(
    input_ptr: tl.tensor,
    ne_output_ptr: tl.tensor,
    numel: tl.constexpr,
    tile_size: tl.constexpr,
):
    """
    Phase 1: Compute the 'not equal' mask indicating the start of new consecutive groups.
    Optimized by removing unnecessary tl.max checks.
    """
    block_start = tle.program_id(axis=0) * tile_size
    offsets = block_start + tl.arange(0, tile_size)
    mask = offsets < numel

    # Load current element
    current_elem = tl.load(input_ptr + offsets, mask=mask)

    # --- Compute 'not equal' mask (True if element is the start of a new group) ---
    ne_result = tl.zeros([tile_size], dtype=tl.int1)

    # Mark the very first element globally as unique
    first_elem_global_mask = (offsets == 0) & mask
    ne_result = tl.where(first_elem_global_mask, True, ne_result)

    has_local_predecessor = (offsets > 0) & mask
    prev_offsets = offsets - 1
    prev_elem = tl.load(input_ptr + prev_offsets, mask=has_local_predecessor)
    local_ne = (current_elem != prev_elem) & has_local_predecessor
    ne_result = tl.where(has_local_predecessor, local_ne, ne_result)

    # --- Handle cross-block boundary ---
    if block_start > 0:
        first_elem_in_tile_mask = (offsets == block_start) & mask
        if tl.sum(first_elem_in_tile_mask.to(tl.int32)) > 0:
            prev_block_last_elem = tl.load(input_ptr + block_start - 1)
            current_first_elem = tl.load(input_ptr + block_start)
            is_first_in_tile_unique_scalar = current_first_elem != prev_block_last_elem
            ne_result = tl.where(
                first_elem_in_tile_mask, is_first_in_tile_unique_scalar, ne_result
            )

    tl.store(ne_output_ptr + offsets, ne_result, mask=mask)


@libentry()
@triton.jit
def unique_consecutive_phase2_kernel(
    input_ptr: tl.tensor,
    input_indices_ptr: tl.tensor,
    ne_input_ptr: tl.tensor,
    cumsum_indices_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    unique_size_ptr: tl.tensor,
    numel: tl.constexpr,
    tile_size: tl.constexpr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    """
    Phase 3: Use precomputed masks and indices to scatter data, compute inverse, prepare counts.
    """
    block_start = tle.program_id(axis=0) * tile_size
    offsets = block_start + tl.arange(0, tile_size)
    mask = offsets < numel

    # Load necessary inputs
    current_elem = tl.load(input_ptr + offsets, mask=mask)
    ne_result = tl.load(ne_input_ptr + offsets, mask=mask)
    cumsum_indices = tl.load(cumsum_indices_ptr + offsets, mask=mask)

    # --- Update global unique count atomically ---
    ne_result_int = ne_result.to(tl.int32)
    local_unique_count = tl.sum(ne_result_int, axis=0).to(tl.int64)
    tl.atomic_add(unique_size_ptr, local_unique_count, sem="relaxed")

    # --- Scatter Output Data ---
    scatter_mask = ne_result & mask
    data_to_write = tl.where(scatter_mask, current_elem, 0.0)
    tl.store(data_out_ptr + cumsum_indices, data_to_write, mask=scatter_mask)

    # --- Compute Inverse Indices ---
    if return_inverse:
        orig_indices = tl.load(input_indices_ptr + offsets, mask=mask)
        tl.store(inverse_indices_ptr + orig_indices, cumsum_indices, mask=mask)

    # --- Prepare for Counts (if needed) ---
    if return_counts:
        tl.store(idx_ptr + cumsum_indices, offsets, mask=scatter_mask)


# --- Helper Kernel for Count Calculation ---
@triton.jit
def calculate_counts_from_indices_kernel(
    idx_ptr: tl.tensor,
    numel_input: tl.constexpr,
    counts_ptr: tl.tensor,
    unique_size: tl.constexpr,
    tile_size: tl.constexpr,
):
    """Calculates counts from the stored indices."""
    block_start = tle.program_id(axis=0) * tile_size
    offsets = block_start + tl.arange(0, tile_size)
    mask = offsets < unique_size

    current_idx = tl.load(idx_ptr + offsets, mask=mask)

    next_offsets = offsets + 1
    next_mask = next_offsets < unique_size
    next_idx = tl.load(idx_ptr + next_offsets, mask=next_mask, other=numel_input)

    count = tl.where(next_mask, next_idx - current_idx, numel_input - current_idx)

    tl.store(counts_ptr + offsets, count, mask=mask)


def _unique_consecutive_dim(input_tensor, return_inverse, return_counts, dim):
    """Handle unique_consecutive when dim is specified."""
    n = input_tensor.dim()

    # Special handling for invalid dim in 1D tensors
    if n == 1 and dim == 1:
        dim = 0

    # Check if dim is in valid range [-n, n-1]
    if dim < -n or dim >= n:
        raise RuntimeError(
            f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {dim})"
        )

    # Normalize dim to [0, n-1]
    if dim < 0:
        dim += n

    # Get shape information
    shape = input_tensor.shape
    size_along_dim = shape[dim]
    numel = input_tensor.numel()

    # Handle empty tensor
    if numel == 0:
        # Output shape: same as input but with size 0 along dim
        output_shape = list(shape)
        output_shape[dim] = 0
        empty_out = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        empty_inv = (
            None
            if not return_inverse
            else torch.empty(shape[dim], dtype=torch.int64, device=input_tensor.device)
        )
        empty_counts = (
            None
            if not return_counts
            else torch.empty(0, dtype=torch.int64, device=input_tensor.device)
        )
        return empty_out, empty_inv, empty_counts

    # make target dimension first
    permute_dims = [dim] + [i for i in range(n) if i != dim]
    input_permuted = input_tensor.permute(permute_dims).contiguous()

    # Reshape to 2D: [size_along_dim, other_dims_prod]
    other_dims = input_permuted.shape[1:]
    other_dims_prod = math.prod(other_dims) if other_dims else 1
    input_reshaped = input_permuted.reshape(size_along_dim, other_dims_prod)

    # Compute mask for new groups
    starts = torch.ones(size_along_dim, dtype=torch.bool, device=input_tensor.device)
    if size_along_dim > 1:
        diff = input_reshaped[1:] != input_reshaped[:-1]
        ne_mask = torch.any(diff, dim=1)
        starts[1:] = ne_mask

    # Get unique indices
    unique_indices = torch.where(starts)[0]
    unique_size = unique_indices.size(0)

    # Extract unique slices
    unique_slices = input_reshaped[unique_indices]

    # Reshape back to original dimensions (with reduced size_along_dim)
    output_shape = (unique_size,) + other_dims
    output_permuted = unique_slices.reshape(output_shape)

    # Inverse permutation to restore original dimension order
    inv_permute_dims = [0] * n
    inv_permute_dims[dim] = 0
    current_dim = 1
    for i in range(n):
        if i != dim:
            inv_permute_dims[i] = current_dim
            current_dim += 1
    output = output_permuted.permute(*inv_permute_dims).contiguous()

    # Prepare inverse indices if needed
    inverse_indices = None
    if return_inverse:
        group_ids = torch.cumsum(starts, dim=0) - 1
        inverse_indices = group_ids

    # Prepare counts if needed
    counts = None
    if return_counts and unique_size > 0:
        positions = unique_indices
        counts_vals = torch.empty(
            unique_size, dtype=torch.int64, device=input_tensor.device
        )
        if positions.numel() > 0:
            if positions.numel() > 1:
                counts_vals[:-1] = positions[1:] - positions[:-1]
            counts_vals[-1] = size_along_dim - positions[-1]
            counts = counts_vals

    return output, inverse_indices, counts


def unique_consecutive(
    input_tensor: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    """
    Implements torch.unique_consecutive using Triton kernels for flattened case and PyTorch for dim case.
    """
    if dim is not None:
        return _unique_consecutive_dim(input_tensor, return_inverse, return_counts, dim)

    # Flatten the input tensor (dim=None case)
    flattened_input = input_tensor.flatten()
    numel = flattened_input.numel()

    if numel == 0:
        empty_out = torch.empty(
            0, dtype=flattened_input.dtype, device=flattened_input.device
        )
        empty_inv = (
            None
            if not return_inverse
            else torch.empty(0, dtype=torch.int64, device=flattened_input.device)
        )
        empty_counts = (
            None
            if not return_counts
            else torch.empty(0, dtype=torch.int64, device=flattened_input.device)
        )
        return empty_out, empty_inv, empty_counts

    # --- Grid and Block Configuration  ---
    MAX_TILE_SIZE = 4096
    TILE_SIZE = min(MAX_TILE_SIZE, triton.next_power_of_2(max(numel, 1)))
    NUM_BLOCKS = triton.cdiv(numel, TILE_SIZE)

    WARP_SIZE = 32
    NUM_WARPS = max(1, min(8, (TILE_SIZE + WARP_SIZE - 1) // WARP_SIZE))

    # --- Allocate Intermediate and Output ---
    ne_buffer = torch.empty(numel, dtype=torch.bool, device=flattened_input.device)
    cumsum_indices_buffer = torch.empty(
        numel, dtype=torch.int32, device=flattened_input.device
    )

    data_out_buffer = torch.empty_like(flattened_input)
    inverse_indices_out = None
    if return_inverse:
        inverse_indices_out = torch.empty(
            numel, dtype=torch.int64, device=flattened_input.device
        )
    idx_buffer = None
    if return_counts:
        idx_buffer = torch.empty(
            numel, dtype=torch.int64, device=flattened_input.device
        )

    unique_size_buffer = torch.zeros(
        (), dtype=torch.int64, device=flattened_input.device
    )

    input_indices_buffer = torch.arange(
        numel, dtype=torch.int64, device=flattened_input.device
    )

    grid = (NUM_BLOCKS, 1, 1)

    # --- Compute ne_result (Start of new groups) ---
    with torch_device_fn.device(flattened_input.device.index):
        unique_consecutive_phase1_kernel[grid](
            flattened_input, ne_buffer, numel, TILE_SIZE, num_warps=NUM_WARPS
        )

    # --- Global Cumsum (Highly optimized native operation) ---
    cumsum_indices_buffer = (
        torch.cumsum(ne_buffer.int(), dim=0, dtype=cumsum_indices_buffer.dtype) - 1
    )

    # --- Scatter, Inverse, Counts Prep ---
    with torch_device_fn.device(flattened_input.device.index):
        unique_consecutive_phase2_kernel[grid](
            flattened_input,
            input_indices_buffer,
            ne_buffer,
            cumsum_indices_buffer,
            data_out_buffer,
            inverse_indices_out,
            idx_buffer,
            unique_size_buffer,
            numel,
            TILE_SIZE,
            return_inverse,
            return_counts,
            num_warps=NUM_WARPS,
        )

    torch_device_fn.synchronize()
    unique_size_computed = unique_size_buffer.item()

    # --- Prepare Final Outputs ---
    data_out_final = data_out_buffer[:unique_size_computed]

    inverse_indices_final = inverse_indices_out if return_inverse else None
    if return_inverse and inverse_indices_final is not None:
        inverse_indices_final = inverse_indices_final.view_as(input_tensor)

    counts_final = None
    if return_counts and idx_buffer is not None and unique_size_computed > 0:
        counts_buffer = torch.empty(
            unique_size_computed, dtype=torch.int64, device=flattened_input.device
        )
        counts_grid = (triton.cdiv(unique_size_computed, TILE_SIZE), 1, 1)
        calculate_counts_from_indices_kernel[counts_grid](
            idx_buffer,
            numel,
            counts_buffer,
            unique_size_computed,
            TILE_SIZE,
            num_warps=NUM_WARPS,
        )
        counts_final = counts_buffer

    return data_out_final, inverse_indices_final, counts_final
