import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_select import index_select as default_index_select
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@libentry()
@triton.jit
def index_select_dim0_1d_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    inp_row_stride,
    out_row_stride,
    row_size,
    num_indices,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for dim=0 index_select - each program handles one row."""
    pid = ext.program_id(axis=0)

    # Load the index for this row
    row_index = tl.load(index_ptr + pid)

    # Calculate input and output row offsets
    inp_row_offset = row_index * inp_row_stride
    out_row_offset = pid * out_row_stride

    # Process row in chunks
    for offset in range(0, row_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < row_size

        # Load from input and store to output
        data = tl.load(inp_ptr + inp_row_offset + cols, mask=mask, other=0.0)
        tl.store(out_ptr + out_row_offset + cols, data, mask=mask)


@libentry()
@triton.jit
def index_select_dim0_split_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    inp_row_stride,
    out_row_stride,
    row_size,
    num_indices,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for dim=0 index_select - 2D grid for large row_size.
    First dimension: indices, Second dimension: column chunks.
    """
    pid_idx = ext.program_id(axis=0)
    pid_col = ext.program_id(axis=1)

    # Load the index for this row
    row_index = tl.load(index_ptr + pid_idx)

    # Calculate input and output row offsets
    inp_row_offset = row_index * inp_row_stride
    out_row_offset = pid_idx * out_row_stride

    # Calculate column offset for this program
    col_offset = pid_col * BLOCK_SIZE
    cols = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = cols < row_size

    # Load from input and store to output
    data = tl.load(inp_ptr + inp_row_offset + cols, mask=mask, other=0.0)
    tl.store(out_ptr + out_row_offset + cols, data, mask=mask)


@libentry()
@triton.jit
def index_select_dim1_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    num_rows,
    inp_row_stride,
    out_row_stride,
    num_indices,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Optimized kernel for dim=1 index_select on 2D tensors.
    Each program handles a tile of rows x indices.
    """
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    row_start = pid_m * BLOCK_M
    idx_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)[:, None]
    idx_offsets = idx_start + tl.arange(0, BLOCK_N)[None, :]

    rows_mask = rows < num_rows
    idx_mask = idx_offsets < num_indices
    mask = rows_mask & idx_mask

    # Load indices
    indices = tl.load(index_ptr + idx_offsets, mask=idx_mask, other=0)

    # Calculate offsets
    inp_offsets = rows * inp_row_stride + indices
    out_offsets = rows * out_row_stride + idx_offsets

    # Load and store
    data = tl.load(inp_ptr + inp_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, data, mask=mask)


def _get_num_warps(total_elements):
    """Get optimal num_warps based on workload size."""
    if total_elements < 1024:
        return 2
    elif total_elements < 4096:
        return 4
    elif total_elements < 16384:
        return 8
    else:
        return 16


def index_select(inp, dim, index):
    logger.debug("GEMS_MTHREADS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)

    dim = dim % inp.ndim
    index_len = index.numel()

    # Create output shape
    out_shape = list(inp.shape)
    out_shape[dim] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if inp.numel() == 0 or index_len == 0:
        return out

    # Optimized path for 2D tensors with dim=0
    if inp.ndim == 2 and dim == 0 and inp.is_contiguous():
        num_rows, row_size = inp.shape
        inp_row_stride = inp.stride(0)
        out_row_stride = out.stride(0)

        # For large row_size, use 2D grid (indices x column_chunks) for more parallelism
        if row_size >= 16384:
            BLOCK_SIZE = 1024
            num_col_chunks = triton.cdiv(row_size, BLOCK_SIZE)
            grid = (index_len, num_col_chunks)
            num_warps = _get_num_warps(BLOCK_SIZE)

            with torch_device_fn.device(inp.device):
                index_select_dim0_split_kernel[grid](
                    inp,
                    out,
                    index,
                    inp_row_stride,
                    out_row_stride,
                    row_size,
                    index_len,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                )
            return out
        else:
            # Use 1D kernel - each program handles one complete row
            BLOCK_SIZE = min(triton.next_power_of_2(row_size), 2048)
            num_warps = _get_num_warps(BLOCK_SIZE)

            with torch_device_fn.device(inp.device):
                index_select_dim0_1d_kernel[(index_len,)](
                    inp,
                    out,
                    index,
                    inp_row_stride,
                    out_row_stride,
                    row_size,
                    index_len,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                )
            return out

    # Optimized path for 2D tensors with dim=1
    if inp.ndim == 2 and dim == 1 and inp.is_contiguous():
        num_rows, num_cols = inp.shape
        inp_row_stride = inp.stride(0)
        out_row_stride = out.stride(0)

        BLOCK_M = min(triton.next_power_of_2(num_rows), 64)
        BLOCK_N = min(triton.next_power_of_2(index_len), 128)

        grid = (triton.cdiv(num_rows, BLOCK_M), triton.cdiv(index_len, BLOCK_N))
        num_warps = _get_num_warps(BLOCK_M * BLOCK_N)

        with torch_device_fn.device(inp.device):
            index_select_dim1_kernel[grid](
                inp,
                out,
                index,
                num_rows,
                inp_row_stride,
                out_row_stride,
                index_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
            )
        return out

    # Fall back to default implementation for other cases
    return default_index_select(inp, dim, index)
