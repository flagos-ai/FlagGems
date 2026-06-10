import logging

import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def index_add_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    src_ptr,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel for index_add operation with autotune.

    After dim_compress, tensors are reshaped so that:
    - inp has shape (M, inp_len) where inp_len is the size of target dimension
    - src has shape (M, N) where N is the size of index

    For each row m and each index position n:
        out[m, index[n]] += alpha * src[m, n]
    """
    pid_m = tle.program_id(axis=0)
    pid_n = tle.program_id(axis=1)

    # Calculate row and column offsets
    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Create masks
    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    # Load indices for this block of columns
    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    # Calculate offsets into inp/out (which has shape M x inp_len)
    inp_off = rows_offset * inp_len + cur_indices

    # Load current values from input
    cur_inp = tl.load(inp_ptr + inp_off, mask=block_mask, other=0.0)

    # Calculate offsets into src (which has shape M x N)
    src_off = rows_offset * N + cols_offset

    # Load source values
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    # Compute: out = inp + alpha * src
    result = cur_inp + alpha * cur_src

    # Store result
    tl.store(out_ptr + inp_off, result, mask=block_mask)


def index_add(inp, dim, index, src, alpha=1):
    """
    Optimized index_add for mthreads backend.

    self.index_add_(dim, index, source, alpha=1) -> Tensor

    For a 3-D tensor the output is:
        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    """
    logger.debug("GEMS_MTHREADS INDEX ADD")

    # Make inputs contiguous
    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    # Normalize dimension
    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Move target dim to last position for coalesced memory access
    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    # Clone input for output
    out = inp.clone()

    # Calculate grid with autotune
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](inp, out, index, src, M, N, alpha, inp_len)

    # Restore original dimension order if needed
    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    """
    In-place version of index_add.
    """
    logger.debug("GEMS_MTHREADS INDEX ADD_")

    # Make index and src contiguous
    index = index.contiguous()
    src = src.contiguous()

    # Normalize dimension
    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Move target dim to last position
    final_dim = inp.ndim - 1

    if dim != final_dim:
        # Need to work on a permuted copy
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](
                inp_work, inp_work, index, src_work, M, N, alpha, inp_len
            )

        # Restore original dimension order and copy back
        order = list(range(inp_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = inp_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        # Can work directly on input if already contiguous
        inp_contig = inp.contiguous()

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](
                inp_contig, inp_contig, index, src, M, N, alpha, inp_len
            )

        # Copy back if input wasn't contiguous
        if not inp.is_contiguous():
            inp.copy_(inp_contig)

    return inp
