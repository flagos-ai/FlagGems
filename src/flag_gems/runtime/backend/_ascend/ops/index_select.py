import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    num_pid_x = tle.num_programs(axis=0)
    loop_count = tl.cdiv(M, num_pid_x)
    for loop in range(0, loop_count):
        rows_offsets = (pid_x * loop_count + loop) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        rows_mask = rows_offsets < M
        cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

        out_mask = rows_mask and (cols_offsets < index_len)

        indices = tl.load(
            index + cols_offsets, mask=(cols_offsets < index_len), other=0
        )
        inp_off = rows_offsets * N + indices[None, :]
        out_off = rows_offsets * index_len + cols_offsets[None, :]

        selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
        tl.store(out + out_off, selected, mask=out_mask)


def index_select(inp, dim, index):
    logger.debug("GEMS_ASCEND INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()

    # with dim_compress
    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    def grid(meta):
        dim0 = triton.cdiv(M, meta["BLOCK_M"])
        dim1 = triton.cdiv(index_len, meta["BLOCK_N"]) if index_len > 0 else 1
        while dim0 * dim1 >= 65536:
            dim0 = triton.cdiv(dim0, 2)
        return (
            dim0,
            dim1,
        )

    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order).contiguous()
    else:
        return out
