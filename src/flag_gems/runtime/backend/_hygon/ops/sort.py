import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    # Explicitly handle bool to avoid ambiguity
    if x.dtype == tl.int1:
        out = uint_to_uint(x, descending)
    elif x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    else:
        out = uint_to_uint(x, descending)
    return out


@triton.jit
def count_kernel(
    arr_ptr,
    count_ptr,  # Output: (Grid, 2**k_bits)
    m,
    N,
    grid_n,  # [FIX] Explicitly pass grid_n
    k_bits: tl.constexpr,
    bit_offset: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    # Use explicitly passed grid_n to avoid inconsistency
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N

    # [FIX] Use int64 for pointer arithmetic to be safe with large m
    val = tl.load(arr_ptr + pid_m.to(tl.int64) * N + n_offset, mask=mask, other=0)
    val_u = convert_to_uint_preverse_order(val, descending)

    bfe_mask: tl.constexpr = (1 << k_bits) - 1
    key = (val_u >> bit_offset) & bfe_mask

    # Cast key to int32 to match atomic_add pointer arithmetic requirements
    key = key.to(tl.int32)

    NUM_BINS: tl.constexpr = 1 << k_bits
    off_base = pid * NUM_BINS
    tl.atomic_add(count_ptr + off_base + key, 1, mask=mask)


@triton.jit
def scatter_kernel(
    arr_ptr,
    arr_out_ptr,
    idx_ptr,  # Optional: input indices
    idx_out_ptr,  # Optional: output indices
    global_offsets_ptr,  # Input: (Grid, 2**k_bits) - Precomputed prefix sum
    m,
    N,
    grid_n,  # [FIX] Explicitly pass grid_n
    k_bits: tl.constexpr,
    bit_offset: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    # Use explicitly passed grid_n
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    NUM_BINS: tl.constexpr = 1 << k_bits
    bfe_mask: tl.constexpr = NUM_BINS - 1

    # Base destination index for this block (ptr to the start of bins for this block)
    off_base_ptr = global_offsets_ptr + pid * NUM_BINS

    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N

    # 1. Load Data
    # [FIX] Use int64 for pointer arithmetic
    val = tl.load(arr_ptr + pid_m.to(tl.int64) * N + n_offset, mask=mask, other=0)
    val_u = convert_to_uint_preverse_order(val, descending)
    key = (val_u >> bit_offset) & bfe_mask
    key = key.to(tl.int32)

    # 2. Load Index (Pre-load OUTSIDE the loop)
    # The index belongs to the thread's element, it is invariant of the bin loop.
    # Loading it once here ensures stability and correctness.
    src_idx = tl.zeros((BLOCK_N,), dtype=tl.int64)
    if idx_ptr is not None:
        src_idx = tl.load(
            idx_ptr + pid_m.to(tl.int64) * N + n_offset, mask=mask, other=0
        )

    # 3. Calculate Local Rank and Scatter
    for b in range(0, NUM_BINS):
        # Load the scalar offset for the specific bin
        base_offset = tl.load(off_base_ptr + b)

        is_bin = key == b

        # Compute local prefix sum for stability
        local_cumsum = tl.cumsum(is_bin.to(tl.int32), axis=0)
        local_rank = local_cumsum - 1

        dest_idx = base_offset + local_rank
        write_mask = mask & is_bin

        # Store Data
        tl.store(arr_out_ptr + pid_m.to(tl.int64) * N + dest_idx, val, mask=write_mask)

        # Store Index (using the pre-loaded value)
        if idx_ptr is not None:
            tl.store(
                idx_out_ptr + pid_m.to(tl.int64) * N + dest_idx,
                src_idx,
                mask=write_mask,
            )


def radix_sort(arr, k_bits=4, descending=False):
    # Determine dimensions
    n = arr.shape[-1]
    m = arr.numel() // n
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    # Tuning parameters
    # Increase k_bits to 8 for speed if compilation allows.
    # BLOCK_N needs to balance register usage.
    BLOCK_N = 512 if k_bits >= 8 else 1024

    grid_n = triton.cdiv(n, BLOCK_N)
    num_bins = 1 << k_bits
    n_passes = triton.cdiv(num_bits, k_bits)

    # Double buffering
    # TODO: If we can modify inplace, we can arr_in = arr
    arr_in = arr.clone()
    arr_out = torch.empty_like(arr)

    # Indices double buffering
    indices_in = (
        torch.arange(0, n, dtype=torch.int64, device=arr.device)
        .broadcast_to(arr.shape)
        .contiguous()
    )
    indices_out = torch.empty_like(indices_in)

    # Count Buffer: (Total_Blocks, num_bins)
    counts = torch.zeros((m * grid_n, num_bins), dtype=torch.int32, device=arr.device)

    with torch_device_fn.device(arr.device):
        for i in range(n_passes):
            bit_offset = i * k_bits

            # Step 1: Count
            counts.zero_()
            grid_total = m * grid_n

            count_kernel[(grid_total,)](
                arr_in,
                counts,
                m,
                n,
                grid_n,  # Pass grid_n explicitly
                k_bits,
                bit_offset,
                BLOCK_N,
                descending,
            )

            # Step 2: Scan (Host Side with PyTorch)
            # Calculate global offsets for Scatter

            # View counts as (m, grid_n, bins)
            cnt_view = counts.view(m, grid_n, num_bins)

            # Total count per bin for each row m
            # .sum() on int32 produces int64 in PyTorch
            total_per_bin = cnt_view.sum(dim=1)  # (m, bins)

            # Global start position of each bin (Exclusive Scan over bins)
            start_per_bin = torch.cumsum(total_per_bin, dim=1) - total_per_bin

            # Offset of each block within its bin (Exclusive Scan over grid)
            offset_in_bin = torch.cumsum(cnt_view, dim=1) - cnt_view

            # Final Offsets = Bin_Start + Block_Offset_In_Bin
            final_offsets = start_per_bin.unsqueeze(1) + offset_in_bin
            final_offsets = final_offsets.view(m * grid_n, num_bins).contiguous()

            # Force offsets to int32 to match kernel pointer expectations
            final_offsets = final_offsets.to(torch.int32)

            # Step 3: Scatter
            scatter_kernel[(grid_total,)](
                arr_in,
                arr_out,
                indices_in,
                indices_out,
                final_offsets,
                m,
                n,
                grid_n,  # Pass grid_n explicitly
                k_bits,
                bit_offset,
                BLOCK_N,
                descending,
            )

            # Swap buffers for next pass
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)

    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    # We only implement stable radix sort here
    logger.debug("GEMS SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS SORT.STABLE")
    # We only implement stable radix sort here
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    # Ensure memory is contiguous even if dim was already last
    # This fixes issues with non-contiguous inputs like slices or transposed tensors
    if not inp.is_contiguous():
        inp = inp.contiguous()

    dtype = inp.dtype
    # NOTE: You can increase this to 8 for higher performance on large arrays,
    # but 4 is safer for compilation/resource limits.
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
