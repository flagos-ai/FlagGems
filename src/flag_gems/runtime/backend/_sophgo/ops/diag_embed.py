import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


@triton.jit
def diag_copy_kernel(
    x_ptr,
    y_ptr,
    B,
    D_local,
    x_row_stride,
    y_batch_stride,
    diag_stride,
):
    """Copy a [B, D_local] tile from x to y_diag (non-contiguous diagonal view).

    Each program handles one diagonal column within the current tile.
    B and D_local are the tile sizes — both small enough that
    B * y_batch_stride * elem_size and D_local * diag_stride * elem_size
    both fit within int32, avoiding overflow in the compiled kernel.

    Uses element-by-element scalar loads/stores via tl.arange(0, 1) to keep
    DMA w_stride=1, avoiding the sophgo TPU hardware limit of w_stride <= 32.
    """
    d = tle.program_id(axis=0)
    off = tl.arange(0, 1)

    for b in range(B):
        val = tl.load(x_ptr + (b * x_row_stride + d) + off)
        tl.store(y_ptr + (b * y_batch_stride + d * diag_stride) + off, val)


def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    logger.debug("GEMS DIAG_EMBED")

    rank = x.ndim + 1

    assert dim1 >= -rank and dim1 < rank, f"Invalid dim1: {dim1}"
    assert dim2 >= -rank and dim2 < rank, f"Invalid dim2: {dim2}"

    # Convert from negative dims
    dim1 = dim1 % rank
    dim2 = dim2 % rank

    assert dim1 != dim2, "diagonal dimensions cannot be identical"

    # As per the docs, exchanging dims is equivalent to changing the sign of offset
    if dim1 > dim2:
        offset = -offset
        dim1, dim2 = dim2, dim1

    # As per the docs, the size of last dim is placed at dim1 and dim2
    last_dim = x.size(-1) + abs(offset)

    y_shape = list(x.shape)
    y_shape.pop()
    y_shape.insert(dim1, last_dim)
    y_shape.insert(dim2, last_dim)

    # Initialize output tensor with zeros using chunked processing
    chunk_size = 128
    y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

    for i in range(0, y_shape[0], chunk_size):
        sub_size = (min(chunk_size, y_shape[0] - i), *y_shape[1:])
        total_size = volume(sub_size)
        grid_fn = lambda meta: (triton.cdiv(total_size, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(device):
            zeros_kernel[grid_fn](
                y[i:i + sub_size[0]],
                total_size,
                BLOCK_SIZE=1024
            )

    y_diag = torch.diagonal(y, offset, dim1, dim2)

    # Flatten x to 2D and copy element-by-element to avoid w_stride too large
    # on sophgo TPU DMA (the diagonal view stride would exceed hardware limit).
    x_flat = x.reshape(-1, x.shape[-1])
    if y_diag.ndim == 1:
        y_diag = y_diag.unsqueeze(0)

    B = x_flat.shape[0]                       # number of batch rows
    D = x.shape[-1]                           # diagonal length
    y_batch_stride = y_diag.stride(-2)         # stride between adjacent batch rows in y_diag
    diag_stride = y_diag.stride(-1)            # stride between adjacent diagonal elements in y_diag
    x_row_stride = x_flat.stride(0)            # row stride in x_flat (= D for contiguous input)

    # Chunk both B and D to keep all element-offset computations within int32
    # byte range. Without D chunking, d * diag_stride * elem_size can overflow
    # when diag_stride is large (e.g. dim1=0,dim2=-1 for [1024,1024] gives
    # diag_stride = 1024*1024+1 = 1048577 → 1023*1048577*4 = 4.29e9 > INT32_MAX).
    B_chunk = 256
    D_chunk = 256
    for b_start in range(0, B, B_chunk):
        b_end = min(b_start + B_chunk, B)
        b_size = b_end - b_start
        for d_start in range(0, D, D_chunk):
            d_end = min(d_start + D_chunk, D)
            d_size = d_end - d_start
            x_tile = x_flat[b_start:b_end, d_start:d_end]
            y_tile = y_diag[b_start:b_end, d_start:d_end]
            with torch_device_fn.device(device):
                diag_copy_kernel[(d_size,)](
                    x_tile, y_tile,
                    b_size, d_size,
                    x_row_stride,
                    y_batch_stride, diag_stride,
                )

    return y
