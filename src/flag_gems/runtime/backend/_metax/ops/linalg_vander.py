import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def vander_kernel_metax(
    x_ptr,
    out_ptr,
    N,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # offsets map to flat output: out_flat[k*N + j] = x_flat[k] ** j
    col = offsets % N
    row = offsets // N

    x_val = tl.load(x_ptr + row, mask=mask)
    result = tl_extra_shim.pow(x_val.to(tl.float32), col.to(tl.float32))

    tl.store(out_ptr + offsets, result, mask=mask)


def linalg_vander(x, N=None):
    logger.debug("GEMS_METAX LINALG_VANDER")

    # fmt: off
    assert x.dtype.is_floating_point or x.dtype.is_complex, f"Unsupported dtype {x.dtype}"
    # fmt: on

    # Handle N parameter
    if N is None:
        N = x.shape[-1]

    # Get input shape info
    batch_dims = x.shape[:-1]
    n = x.shape[-1]

    # Flatten batch dims
    x_flat = x.reshape(-1)

    # Output shape: (*, n, N)
    final_shape = batch_dims + (n, N)

    # Run triton kernel
    total_elements = x_flat.numel() * N
    out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    vander_kernel_metax[grid](x_flat, out, N, total_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
