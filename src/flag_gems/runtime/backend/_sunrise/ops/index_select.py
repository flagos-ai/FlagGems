import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = ext.program_id(axis=0)
    pid_y = ext.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask and (cols_offsets < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound

    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    final_mask = out_mask & index_valid_mask
    selected = tl.load(inp + inp_off, mask=final_mask, other=0.0)
    tl.store(out + out_off, selected, mask=final_mask)


def index_select_heur_block_m(args):
    N = args["N"]
    if N >= 8192:
        return 2048
    elif N >= 4096:
        return 1024
    elif N >= 1024:
        return 512
    return 256


@libentry()
@triton.heuristics({"BLOCK_N": index_select_heur_block_m})
@triton.jit
def index_select_dim0_kernel(inp, out, N, index, BLOCK_N: tl.constexpr):
    pid_x = ext.program_id(axis=0)
    pid_y = ext.program_id(axis=1)
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols_offsets < N
    indices = tl.load(index + pid_x)
    in_offset = indices * N + cols_offsets
    selected = tl.load(inp + in_offset, mask=mask, other=0.0)
    out_offset = pid_x * N + cols_offsets
    tl.store(out + out_offset, selected, mask=mask)


def _dim_compress(inp, dim):
    batch_dim = [dim]
    reduction_dim = [i for i in range(inp.ndim) if i != dim]
    order = batch_dim + reduction_dim
    return inp.permute(order).contiguous()


def index_select_dim0(inp, dim, index):
    # inp_shape = list(inp.shape)
    inp = _dim_compress(inp, dim)
    out_shape = list(inp.shape)
    index_len = index.numel()
    out_shape[0] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    N = out.numel() // index_len
    grid = lambda meta: (
        index_len,
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    index_select_dim0_kernel[grid](inp, out, N, index)
    if dim != 0:
        order = [i for i in range(1, out.ndim)]
        order.insert(dim, 0)
        return out.permute(order).contiguous()
    else:
        return out


def index_select(inp, dim, index):
    logger.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()

    if index_len > 0 and (inp.ndim == 2 or inp.ndim == 3):
        return index_select_dim0(inp, dim, index)

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
        out = out.permute(order).contiguous()
        return out.reshape(out.shape)
    else:
        return out
