import logging

import torch
import triton
import triton.language as tl

import flag_gems.runtime as runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


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
def index_select_2d_opt_kernel(inp, out, M, N, index, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(axis=0)

    row_index = tl.load(index + pid)
    row_offset = row_index * M
    rows_mask = row_index < N

    for m in range(0, tl.cdiv(M, BLOCK_SIZE)):
        cols_offsets = m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols_mask = cols_offsets < M
        block_mask = rows_mask and cols_mask
        cur_inp = tl.load(inp + row_offset + cols_offsets, mask=block_mask, other=0.0)
        out_offset = pid * M + cols_offsets
        tl.store(out + out_offset, cur_inp, mask=block_mask)


# Swap two dimensions of a 2D tensor to for better memory access pattern
def dim_transpose(inp):
    return torch.transpose(inp, 0, 1).contiguous()


def index_select(inp, dim, index):
    logger.debug("METAX GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    index_len = index.numel()
    inp_shape = list(inp.shape)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[dim] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if inp.ndim == 2 and dim == 0:
        BLOCK_SIZE = min(triton.next_power_of_2(M), 4096)
        if dim == 0:
            index_select_2d_opt_kernel[(index_len,)](
                inp, out, M, N, index, BLOCK_SIZE=BLOCK_SIZE
            )

            return out
    else:
        # with dim_compress
        inp = dim_compress(inp, dim)
        N = inp_shape[dim]
        M = inp.numel() // N
        out_shape = list(inp.shape)
        out_shape[inp.ndim - 1] = index_len
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(index_len, meta["BLOCK_N"]),
        )
        index_select_kernel[grid](inp, out, M, N, index, index_len)
        if dim != out.ndim - 1:
            order = [i for i in range(out.ndim - 1)]
            order.insert(dim, out.ndim - 1)
            return out.permute(order).contiguous()
        else:
            return out
