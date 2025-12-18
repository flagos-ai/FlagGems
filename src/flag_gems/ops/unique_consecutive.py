import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry


@libentry()
@triton.jit
def unique_consecutive_flat_kernel(
    input_ptr: tl.tensor,           # Pointer to the input tensor data
    input_indices_ptr: tl.tensor,   # Pointer to indices [0, 1, 2, ... N-1] (for inverse)
    data_out_ptr: tl.tensor,        # Pointer to the output unique data tensor
    inverse_indices_ptr: tl.tensor, # Pointer to the output inverse indices tensor
    idx_ptr: tl.tensor,             # Pointer to the index buffer for counts
    unique_size_ptr: tl.tensor,     # Pointer to store the final size of unique elements
    numel: tl.constexpr,            # Total number of elements in the input
    tile_size: tl.constexpr,        # Tile size for processing
    return_inverse: tl.constexpr,   # Flag to compute inverse indices
    return_counts: tl.constexpr,    # Flag to compute counts
):
    """
    Kernel to find unique consecutive elements and optionally compute inverse indices/counts.
    """
    # Create block indices
    block_start = tle.program_id(axis=0) * tile_size
    offsets = block_start + tl.arange(0, tile_size)
    mask = offsets < numel

    # Load current element
    current_elem = tl.load(input_ptr + offsets, mask=mask)

    # --- Compute 'not equal' mask ---
    ne_result = tl.full([tile_size], 1, dtype=tl.int1)

    has_local_predecessor = (offsets > 0) & mask
    if tl.max(has_local_predecessor.to(tl.int32)) > 0:
        prev_offsets = offsets - 1
        prev_elem = tl.load(input_ptr + prev_offsets, mask=has_local_predecessor)
        local_ne = (current_elem != prev_elem) & has_local_predecessor
        ne_result = tl.where(has_local_predecessor, local_ne, ne_result)

    # --- Handle cross-block boundary ---
    if block_start > 0:
         first_elem_mask = (offsets == block_start) & mask
         if tl.max(first_elem_mask.to(tl.int32)) > 0:
              current_first_elem = tl.load(input_ptr + block_start, mask=first_elem_mask)
              prev_block_last_elem = tl.load(input_ptr + block_start - 1)
              is_first_unique = current_first_elem != prev_block_last_elem
              ne_result = tl.where(first_elem_mask, is_first_unique, ne_result)

    # --- Convert bool to int for cumsum ---
    ne_result_int = ne_result.to(tl.int32)
    cumsum_indices = tl.cumsum(ne_result_int, axis=0)

    is_last_position_in_grid = (tle.program_id(axis=0) == tle.num_programs(axis=0) - 1)
    is_last_valid_offset = offsets == (numel - 1)
    last_cumsum_mask = is_last_position_in_grid & is_last_valid_offset & mask
    if tl.max(last_cumsum_mask.to(tl.int32)) > 0:
        last_cumsum_val = tl.load(cumsum_indices + offsets, mask=last_cumsum_mask)
        tl.atomic_max(unique_size_ptr, last_cumsum_val)

    # --- Scatter Output Data ---
    scatter_mask = ne_result & mask
    output_indices = cumsum_indices
    tl.store(data_out_ptr + output_indices, current_elem, mask=scatter_mask)

    # --- Compute Inverse Indices ---
    if return_inverse:
        orig_indices = tl.load(input_indices_ptr + offsets, mask=mask)
        tl.store(inverse_indices_ptr + orig_indices, cumsum_indices, mask=mask)

    # --- Prepare for Counts (if needed) ---
    if return_counts:
        store_count_idx_mask = ne_result & mask
        tl.store(idx_ptr + output_indices, offsets, mask=store_count_idx_mask)


# --- Helper Kernel for Count Calculation (Reused) ---
@triton.jit
def calculate_counts_from_indices_kernel(
    idx_ptr: tl.tensor,       # Buffer containing start indices of unique runs
    numel_input: int,         # Original input size (N)
    counts_ptr: tl.tensor,    # Output buffer for counts
    unique_size: int,         # Number of unique elements (M)
    tile_size: tl.constexpr,
):
    """Calculates counts from the stored indices."""
    block_start = tle.program_id(axis=0) * tile_size
    offsets = block_start + tl.arange(0, tile_size)
    mask = offsets < unique_size

    # Load current index (start of run)
    current_idx = tl.load(idx_ptr + offsets, mask=mask)

    # Load next index (start of next run)
    next_offsets = offsets + 1
    next_mask = next_offsets < unique_size
    next_idx = tl.load(idx_ptr + next_offsets, mask=next_mask)

    # Calculate count for current unique element
    count = tl.where(next_mask,
                     next_idx - current_idx,
                     numel_input - current_idx)

    # Store the calculated count
    tl.store(counts_ptr + offsets, count, mask=mask)


def unique_consecutive(
    input_tensor: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None # Assuming None (flatten) for this implementation
):
    """
    Implements torch.unique_consecutive using Triton kernels.
    Assumes dim=None (flattens input).
    """
    if dim is not None:
        raise NotImplementedError("unique_consecutive with dim != None is not implemented.")

    # Flatten the input tensor
    flattened_input = input_tensor.flatten()
    numel = flattened_input.numel()

    if numel == 0:
        empty_out = torch.empty(0, dtype=flattened_input.dtype, device=flattened_input.device)
        empty_inv = torch.empty(0, dtype=torch.int64, device=flattened_input.device) if return_inverse else None
        empty_counts = torch.empty(0, dtype=torch.int64, device=flattened_input.device) if return_counts else None
        return empty_out, empty_inv, empty_counts

    # --- Grid and Block Configuration ---
    # Similar heuristic to simple_unique_flat, adjusted for potentially longer sequences
    TILE_SIZE = min(8192, triton.next_power_of_2(numel))
    NUM_BLOCKS = triton.cdiv(numel, TILE_SIZE)
    NUM_WARPS = 4 if NUM_BLOCKS == 1 else 8 # Adjusted defaults

    # --- Allocate Output Tensors ---
    data_out_buffer = torch.empty_like(flattened_input) # Buffer, might be larger than needed
    inverse_indices_out = None
    if return_inverse:
        inverse_indices_out = torch.empty_like(flattened_input, dtype=torch.int64)
    counts_out = None
    idx_buffer = None
    if return_counts:
        idx_buffer = torch.empty_like(flattened_input, dtype=torch.int64) # Buffer for run start indices
    unique_size_buffer = torch.zeros((), dtype=torch.int64, device=flattened_input.device) # Scalar tensor for size

    # Buffer for input indices [0, 1, ..., N-1]
    input_indices_buffer = torch.arange(numel, dtype=torch.int64, device=flattened_input.device)


    # --- Launch Main Kernel ---
    grid = (NUM_BLOCKS, 1, 1)
    with torch_device_fn.device(flattened_input.device.index):
        unique_consecutive_flat_kernel[grid](
            flattened_input,              # input_ptr
            input_indices_buffer,         # input_indices_ptr
            data_out_buffer,              # data_out_ptr
            inverse_indices_out,          # inverse_indices_ptr
            idx_buffer,                   # idx_ptr
            unique_size_buffer,           # unique_size_ptr
            numel,                        # numel
            TILE_SIZE,                    # tile_size
            return_inverse,               # return_inverse
            return_counts,                # return_counts
            num_warps=NUM_WARPS
        )

    # Get the final unique size
    unique_size_computed = unique_size_buffer.item() + 1

    if unique_size_computed <= 0:
         unique_size_computed = 1 if numel > 0 else 0

    data_out_final = data_out_buffer[:unique_size_computed]
    inverse_indices_final = inverse_indices_out if not return_inverse else inverse_indices_out
    counts_final = None
    if return_counts and idx_buffer is not None:
        counts_buffer = torch.empty((unique_size_computed,), dtype=torch.int64, device=flattened_input.device)
        counts_grid = (triton.cdiv(unique_size_computed, TILE_SIZE), 1, 1)
        calculate_counts_from_indices_kernel[counts_grid](
            idx_buffer,
            numel,
            counts_buffer,
            unique_size_computed,
            TILE_SIZE,
            num_warps=NUM_WARPS
        )
        counts_final = counts_buffer

    # Reshape inverse indices to match original input shape if needed
    if return_inverse and inverse_indices_final is not None:
        inverse_indices_final = inverse_indices_final.view_as(input_tensor)

    return data_out_final, inverse_indices_final, counts_final

