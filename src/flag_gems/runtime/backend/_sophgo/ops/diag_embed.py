import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


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

    # Calculate total number of elements
    N = 1
    for y_dim in y_shape:
        N = N * y_dim

    # Initialize output tensor with zeros using chunked processing
    chunk_size = [128, 64]
    y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

    for i in range(0, y_shape[0], chunk_size[0]):
        sub_size = (min(chunk_size[0], y_shape[0] - i), *y_shape[1:])
        total_size = volume(sub_size)
        grid_fn = lambda meta: (triton.cdiv(total_size, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(device):
            zeros_kernel[grid_fn](
                y[i:i + sub_size[0]],
                total_size,
                BLOCK_SIZE=1024
            )

    y_diag = torch.diagonal(y, offset, dim1, dim2)

    # Copy diagonal elements based on tensor dimensions
    if x.ndim > 1 and dim1 == 1 and dim2 == 2:
        for i in range(0, x.shape[0]):
            copy_func.instantiate(x.ndim - 1)(x[i], out0=y_diag[i])
    else:
        D = y_diag.shape[-1]
        for start in range(0, D, chunk_size[1]):
            end = min(start + chunk_size[1], D)
            y_chunk = y_diag[..., start:end]  # [B, chunk_len]
            x_chunk = x[..., start:end]
            copy_func.instantiate(x.ndim)(x_chunk, out0=y_chunk)

    return y
