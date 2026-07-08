import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


def index_select_dim0_block_index(index_len, c_dim):
    if c_dim <= 256:
        return 8 if index_len >= 512 else 4
    if c_dim <= 1024:
        return 4 if index_len >= 256 else 2
    return 1


def index_select_dim0_block_c(c_dim):
    if c_dim <= 128:
        return 128
    if c_dim <= 512:
        return 256
    return 128


@libentry()
@triton.jit
def index_select_dim0_kernel(
    inp,
    out,
    index,
    index_len,
    c_dim,
    BLOCK_INDEX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_i = pid_i * BLOCK_INDEX + tl.arange(0, BLOCK_INDEX)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    row_mask = offs_i < index_len
    col_mask = offs_c < c_dim
    mask = row_mask[:, None] & col_mask[None, :]

    rows = tl.load(index + offs_i, mask=row_mask, other=0).to(tl.int64)
    inp_ptrs = inp + rows[:, None] * c_dim + offs_c[None, :]
    out_ptrs = out + offs_i[:, None] * c_dim + offs_c[None, :]

    values = tl.load(inp_ptrs, mask=mask, other=0)
    tl.store(out_ptrs, values, mask=mask)


def _generic_index_select(inp, dim, index):
    from flag_gems.ops.index_select import index_select as generic_index_select

    return generic_index_select(inp, dim, index)


def index_select(inp, dim, index):
    logging.debug("GEMS SOPHGO INDEX_SELECT")

    if torch.is_grad_enabled() and inp.requires_grad:
        return _generic_index_select(inp, dim, index)

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim

    if dim != 0:
        return _generic_index_select(inp, dim, index)

    # Preserve the narrow dim0 fast path for the historical 2D hotspot, but
    # avoid applying it to higher-rank shapes where the migrated implementation
    # is no longer numerically stable under the new pytest coverage.
    if inp.ndim != 2 or not inp.is_contiguous():
        return _generic_index_select(inp, dim, index)

    index_len = index.numel()
    if index_len == 0:
        out_shape = list(inp.shape)
        out_shape[0] = 0
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    inp = inp.contiguous()
    index = index.contiguous()
    c_dim = math.prod(inp.shape[1:])

    inp_2d = inp.reshape(inp.shape[0], c_dim)
    out_2d = torch.empty((index_len, c_dim), dtype=inp.dtype, device=inp.device)

    block_index = index_select_dim0_block_index(index_len, c_dim)
    block_c = index_select_dim0_block_c(c_dim)
    grid = (triton.cdiv(index_len, block_index), triton.cdiv(c_dim, block_c))

    with torch_device_fn.device(inp.device):
        index_select_dim0_kernel[grid](
            inp_2d,
            out_2d,
            index,
            index_len,
            c_dim,
            BLOCK_INDEX=block_index,
            BLOCK_C=block_c,
        )

    return out_2d.reshape(index_len, *inp.shape[1:])
