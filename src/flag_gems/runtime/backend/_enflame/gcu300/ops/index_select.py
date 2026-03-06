import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def index_select_kernel(
    inp,
    out,
    M,
    N,
    index,
    index_len,
    NUM_TILES_BLOCK_M: tl.constexpr,
    NUM_TILES_PER_BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_x = pid_x * NUM_TILES_PER_BLOCK_M
    pid_y = tl.program_id(axis=1)
    for start_m in range(0, NUM_TILES_PER_BLOCK_M):
        if pid_x < NUM_TILES_BLOCK_M:
            rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
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
        pid_x = pid_x + 1


def make_contiguous_with_correct_stride(tensor):
    """确保 contiguous 后的 tensor 具有正确的 stride"""
    if tensor.numel() == 0:
        # 对于空 tensor，手动创建具有正确 stride 的版本
        new_tensor = tensor.flatten().view(tensor.shape)
        return new_tensor.contiguous()
    return tensor.contiguous()


def index_select(inp, dim, index):
    logger.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if index.dtype == torch.int64:
        index = index.to(torch.int32)

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

    BLOCK_M = min(4, triton.next_power_of_2(triton.cdiv(256, N)))
    m = min(triton.next_power_of_2(triton.cdiv(N, 16)), 512)
    BLOCK_N = max(m, 16)
    grid_m = num_tiles_block_m = triton.cdiv(M, BLOCK_M)
    if grid_m > MAX_GRID_DIM:
        num_tiles_per_block_m = triton.cdiv(grid_m, MAX_GRID_DIM)
        grid_m = MAX_GRID_DIM
    else:
        num_tiles_per_block_m = 1
    grid_n = triton.cdiv(index_len, BLOCK_N)
    index_select_kernel[(grid_m, grid_n)](
        inp,
        out,
        M,
        N,
        index,
        index_len,
        NUM_TILES_BLOCK_M=num_tiles_block_m,
        NUM_TILES_PER_BLOCK_M=num_tiles_per_block_m,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        res = make_contiguous_with_correct_stride(out.permute(order))
        return res.reshape(res.shape)
    else:
        return out
