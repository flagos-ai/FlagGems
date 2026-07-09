import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def vecdot_kernel_multi(
    x_ptr,
    y_ptr,
    output_ptr,
    batch_size,
    vec_dim,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    tid = tl.arange(0, BLOCK_SIZE)
    offsets = tid
    mask = tid < vec_dim

    for b in range(BLOCK_BATCH):
        batch_id = pid * BLOCK_BATCH + b
        if batch_id < batch_size:
            batch_offset = batch_id * vec_dim
            x = tl.load(x_ptr + batch_offset + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + batch_offset + offsets, mask=mask, other=0.0)
            result = tl.sum(x * y, axis=0)
            tl.store(output_ptr + batch_id, result)


def linalg_vecdot(x, y, dim=-1):
    logger.debug("GEMS LINALG_VECDOT")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be tensors")

    x = x.contiguous()
    y = y.contiguous()

    if x.shape != y.shape:
        raise ValueError("Input shapes must match")

    if dim < 0:
        dim = x.dim() + dim

    vec_dim = x.shape[dim]
    batch_shape = list(x.shape)
    batch_shape.pop(dim)

    x_flat = x.movedim(dim, -1).reshape(-1, vec_dim)
    y_flat = y.movedim(dim, -1).reshape(-1, vec_dim)

    batch_size = x_flat.shape[0]
    output = torch.empty(batch_size, dtype=x.dtype, device=x.device)

    if vec_dim < 8:
        BLOCK_SIZE = 16
    elif vec_dim < 16:
        BLOCK_SIZE = 32
    elif vec_dim < 32:
        BLOCK_SIZE = 64
    elif vec_dim < 64:
        BLOCK_SIZE = 128
    elif vec_dim < 128:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512

    if batch_size < 16:
        BLOCK_BATCH = 1
    elif batch_size < 64:
        BLOCK_BATCH = 2
    elif batch_size < 128:
        BLOCK_BATCH = 4
    else:
        BLOCK_BATCH = 8

    grid = (triton.cdiv(batch_size, BLOCK_BATCH),)

    vecdot_kernel_multi[grid](
        x_flat,
        y_flat,
        output,
        batch_size,
        vec_dim,
        BLOCK_BATCH,
        BLOCK_SIZE,
    )

    return output.view(batch_shape)
