import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.shape_utils import c_contiguous_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)

# repeat_interleave.self_{int,Tensor} are CompositeImplicitAutograd;
# Direct coverage will cause the gradient to break;
# Redispatch to this keyset to run the decomposed forward (and backward)
# when gradients may be needed.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


@pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_MTHREADS REPEAT_INTERLEAVE_SELF_INT")
    if torch.is_grad_enabled():
        return torch.ops.aten.repeat_interleave.self_int.redispatch(
            _FALLBACK_KEYSET, inp, repeats, dim, output_size=output_size
        )
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    out_view_stride = c_contiguous_stride(out_view_shape)

    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(output, out_view_shape, out_view_stride)
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask, other=0)
    out_offset = cumsum - repeats

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS_MTHREADS REPEAT_INTERLEAVE_TENSOR")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"

    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()

    assert result_size >= 0, "repeats can not be negative"

    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    grid = (size,)
    BLOCK_SIZE = 32
    with torch_device_fn.device(repeats.device):
        repeat_interleave_tensor_kernel[grid](
            repeats,
            cumsum,
            out,
            size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
        )
    return out


@libentry()
@triton.jit
def fused_repeat_interleave_dim0_kernel(
    inp_ptr,
    out_ptr,
    cumsum_ptr,
    num_input_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for repeat_interleave with dim=0.
    Each program handles one input row and copies to all its repeated output positions.
    """
    pid = ext.program_id(0)

    if pid >= num_input_rows:
        return

    # Get output row range for this input row
    row_idx_mask = pid > 0
    start_row_idx = tl.load(cumsum_ptr + pid - 1, mask=row_idx_mask, other=0)
    end_row_idx = tl.load(cumsum_ptr + pid)

    num_of_rows = end_row_idx - start_row_idx
    if num_of_rows == 0:
        return

    # Calculate input row offset
    inp_row_offset = pid * row_size

    # Process columns in blocks
    for col_block in range(0, tl.cdiv(row_size, BLOCK_SIZE)):
        col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < row_size

        # Load from input
        cur_inp = tl.load(
            inp_ptr + inp_row_offset + col_offsets, mask=col_mask, other=0.0
        )

        # Store to each output row
        for cur_row in range(0, num_of_rows):
            output_row_index = start_row_idx + cur_row
            output_row_offsets = output_row_index * row_size + col_offsets
            tl.store(out_ptr + output_row_offsets, cur_inp, mask=col_mask)


@libentry()
@triton.jit
def fused_repeat_interleave_output_centric_kernel(
    inp_ptr,
    out_ptr,
    cumsum_ptr,
    num_input_rows,
    num_output_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Output-centric kernel for repeat_interleave with dim=0.
    Uses 2D grid: (num_output_rows, num_col_chunks).
    Uses binary search to find input row.
    """
    out_row_idx = ext.program_id(0)
    col_chunk_idx = ext.program_id(1)

    if out_row_idx >= num_output_rows:
        return

    # Binary search to find input row index
    # Find the smallest i such that cumsum[i] > out_row_idx
    low = 0
    high = num_input_rows
    while low < high:
        mid = (low + high) // 2
        cumsum_mid = tl.load(cumsum_ptr + mid)
        if cumsum_mid <= out_row_idx:
            low = mid + 1
        else:
            high = mid

    inp_row_idx = low

    # Calculate column offsets for this chunk
    col_offset = col_chunk_idx * BLOCK_SIZE
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < row_size

    # Load from input
    inp_row_offset = inp_row_idx * row_size
    cur_inp = tl.load(inp_ptr + inp_row_offset + col_offsets, mask=col_mask, other=0.0)

    # Store to output
    out_row_offset = out_row_idx * row_size
    tl.store(out_ptr + out_row_offset + col_offsets, cur_inp, mask=col_mask)


@libentry()
@triton.jit
def fused_repeat_interleave_1d_bsearch_kernel(
    inp_ptr,
    out_ptr,
    cumsum_ptr,
    num_input_rows,
    num_output_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    """1D output-centric kernel with binary search.
    Each program handles one complete output row.
    Better for large row sizes.
    """
    out_row_idx = ext.program_id(0)

    if out_row_idx >= num_output_rows:
        return

    # Binary search to find input row index
    low = 0
    high = num_input_rows
    while low < high:
        mid = (low + high) // 2
        cumsum_mid = tl.load(cumsum_ptr + mid)
        if cumsum_mid <= out_row_idx:
            low = mid + 1
        else:
            high = mid

    inp_row_idx = low

    # Calculate row offsets
    inp_row_offset = inp_row_idx * row_size
    out_row_offset = out_row_idx * row_size

    # Process all columns in blocks
    for col_offset in range(0, row_size, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < row_size

        cur_inp = tl.load(
            inp_ptr + inp_row_offset + col_offsets, mask=col_mask, other=0.0
        )
        tl.store(out_ptr + out_row_offset + col_offsets, cur_inp, mask=col_mask)


@libentry()
@triton.jit
def fused_repeat_interleave_with_indices_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    num_output_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Output-centric kernel using precomputed index mapping.
    Uses 2D grid: (num_output_rows, num_col_chunks).
    """
    out_row_idx = ext.program_id(0)
    col_chunk_idx = ext.program_id(1)

    if out_row_idx >= num_output_rows:
        return

    # Load precomputed input row index
    inp_row_idx = tl.load(index_ptr + out_row_idx)

    # Calculate column offsets for this chunk
    col_offset = col_chunk_idx * BLOCK_SIZE
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < row_size

    # Load from input
    inp_row_offset = inp_row_idx * row_size
    cur_inp = tl.load(inp_ptr + inp_row_offset + col_offsets, mask=col_mask, other=0.0)

    # Store to output
    out_row_offset = out_row_idx * row_size
    tl.store(out_ptr + out_row_offset + col_offsets, cur_inp, mask=col_mask)


@libentry()
@triton.jit
def fused_repeat_interleave_large_row_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    num_output_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for large row sizes.
    Each program handles one output row and processes all columns.
    """
    out_row_idx = ext.program_id(0)

    if out_row_idx >= num_output_rows:
        return

    # Load precomputed input row index
    inp_row_idx = tl.load(index_ptr + out_row_idx)

    # Calculate row offsets
    inp_row_offset = inp_row_idx * row_size
    out_row_offset = out_row_idx * row_size

    # Process all columns in blocks
    for col_offset in range(0, row_size, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < row_size

        # Load from input and store to output
        cur_inp = tl.load(
            inp_ptr + inp_row_offset + col_offsets, mask=col_mask, other=0.0
        )
        tl.store(out_ptr + out_row_offset + col_offsets, cur_inp, mask=col_mask)


def fused_repeat_interleave_dim0(inp, repeats, dim):
    """Fused repeat_interleave for dim=0 case.
    Works with any tensor dimension, handles dim=0 efficiently.
    """
    logger.debug("GEMS_MTHREADS FUSED_REPEAT_INTERLEAVE_DIM0")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"

    # Compute cumsum of repeats
    cumsum = repeats.cumsum(axis=0)
    total_output_rows = cumsum[-1].item()

    if total_output_rows == 0:
        out_shape = list(inp.shape)
        out_shape[dim] = 0
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Setup output tensor
    out_shape = list(inp.shape)
    out_shape[dim] = total_output_rows
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Flatten non-dim dimensions for easier indexing
    num_input_rows = inp.shape[dim]
    row_size = inp.numel() // num_input_rows

    # Make input contiguous for efficient access
    inp_contig = inp.contiguous()

    # Strategy selection:
    # 1. Small tensors: input-centric kernel
    # 2. Medium row sizes: output-centric 2D grid with binary search
    # 3. Large row sizes: output-centric 1D grid with binary search

    if row_size < 512 and total_output_rows < 512:
        # Small tensor: use input-centric kernel
        BLOCK_SIZE = min(triton.next_power_of_2(row_size), 4096)

        if BLOCK_SIZE <= 256:
            num_warps = 2
        elif BLOCK_SIZE <= 512:
            num_warps = 4
        else:
            num_warps = 8

        grid = (num_input_rows,)

        with torch_device_fn.device(inp.device):
            fused_repeat_interleave_dim0_kernel[grid](
                inp_contig,
                out,
                cumsum,
                num_input_rows,
                row_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
    elif row_size >= 16384:
        # Large row size: use 1D grid with binary search
        # This reduces total number of programs and amortizes binary search cost
        BLOCK_SIZE = 2048
        num_warps = 16

        grid = (total_output_rows,)

        with torch_device_fn.device(inp.device):
            fused_repeat_interleave_1d_bsearch_kernel[grid](
                inp_contig,
                out,
                cumsum,
                num_input_rows,
                total_output_rows,
                row_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
    else:
        # Medium row size: use 2D grid with binary search
        BLOCK_SIZE = min(triton.next_power_of_2(row_size), 1024)
        num_col_chunks = triton.cdiv(row_size, BLOCK_SIZE)

        if BLOCK_SIZE <= 256:
            num_warps = 2
        elif BLOCK_SIZE <= 512:
            num_warps = 4
        else:
            num_warps = 8

        grid = (total_output_rows, num_col_chunks)

        with torch_device_fn.device(inp.device):
            fused_repeat_interleave_output_centric_kernel[grid](
                inp_contig,
                out,
                cumsum,
                num_input_rows,
                total_output_rows,
                row_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

    return out


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_MTHREADS REPEAT_INTERLEAVE_SELF_TENSOR")
    if torch.is_grad_enabled():
        return torch.ops.aten.repeat_interleave.self_Tensor.redispatch(
            _FALLBACK_KEYSET, inp, repeats, dim, output_size=output_size
        )

    if repeats.numel() == 0:
        return inp.clone()

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim, but got \
                repeats.size(0) = {} and input.size({}) = {}".format(
                repeats.size(0), dim, inp_shape[dim]
            )
        )

    # Use fused kernel for dim=0
    if dim == 0:
        return fused_repeat_interleave_dim0(inp, repeats, dim)

    # For other dimensions, use the fallback implementation
    indices = repeat_interleave_tensor(repeats)
    res = torch.index_select(inp, dim, indices)

    return res
