import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask and (cols_offsets < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)


@libentry()
@triton.jit
def _index_select_dim0_kernel(
    inp,
    out,
    M,
    index,
    index_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """index_select on a 2D (N, M) contiguous tensor where dim=0.

    Equivalent to: out[i, j] = inp[index[i], j]
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < index_len
    idx = tl.load(index + row_offsets, mask=row_mask, other=0)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < M

    inp_off = idx[:, None] * M + col_offsets[None, :]
    out_off = row_offsets[:, None] * M + col_offsets[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    val = tl.load(inp + inp_off, mask=mask, other=0.0)
    tl.store(out + out_off, val, mask=mask)


@libentry()
@triton.jit
def _index_select_inner_2d_kernel(
    inp,
    out,
    before_size,
    N,
    after_size,
    index,
    index_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """index_select on dim=1 of a (B, N, A) tensor flattened to (B*N, A).

    Each output row k = b * index_len + i maps to input row b * N + index[i].
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < (before_size * index_len)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < after_size

    b = row_offsets // index_len
    i = row_offsets % index_len
    idx = tl.load(index + i, mask=row_mask, other=0)
    inp_row = b * N + idx

    inp_off = inp_row[:, None] * after_size + col_offsets[None, :]
    out_off = row_offsets[:, None] * after_size + col_offsets[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    val = tl.load(inp + inp_off, mask=mask, other=0.0)
    tl.store(out + out_off, val, mask=mask)


def index_select(inp, dim, index):
    logger.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()
    N = inp_shape[dim]
    M = inp.numel() // N

    if dim == 0:
        inp_2d = inp.reshape(N, M)
        out_2d = torch.empty(
            (index_len, M), dtype=inp.dtype, device=inp.device
        )
        grid = lambda meta: (
            triton.cdiv(index_len, meta["BLOCK_M"]),
            triton.cdiv(M, meta["BLOCK_N"]),
        )
        _index_select_dim0_kernel[grid](
            inp_2d, out_2d, M, index, index_len, BLOCK_M=32, BLOCK_N=64,
        )
        out_shape = [index_len] + inp_shape[1:]
        return out_2d.reshape(out_shape)
    elif dim == inp.ndim - 1:
        inp_2d = inp.reshape(M, N)
        out_2d = torch.empty(
            (M, index_len), dtype=inp.dtype, device=inp.device
        )
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(index_len, meta["BLOCK_N"]),
        )
        index_select_kernel[grid](inp_2d, out_2d, M, N, index, index_len)
        out_shape = inp_shape[:-1] + [index_len]
        return out_2d.reshape(out_shape)
    else:
        before_size = 1
        for d in range(0, dim):
            before_size *= inp_shape[d]
        after_size = 1
        for d in range(dim + 1, inp.ndim):
            after_size *= inp_shape[d]
        inp_2d = inp.reshape(before_size * N, after_size)
        out_rows = before_size * index_len
        out_2d = torch.empty(
            (out_rows, after_size), dtype=inp.dtype, device=inp.device
        )
        grid = lambda meta: (
            triton.cdiv(out_rows, meta["BLOCK_M"]),
            triton.cdiv(after_size, meta["BLOCK_N"]),
        )
        _index_select_inner_2d_kernel[grid](
            inp_2d, out_2d, before_size, N, after_size,
            index, index_len, BLOCK_M=32, BLOCK_N=64,
        )
        out_shape = inp_shape[:dim] + [index_len] + inp_shape[dim + 1:]
        return out_2d.reshape(out_shape)