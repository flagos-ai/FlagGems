import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .sort import sort as gems_sort

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

ModeOut = namedtuple("mode", ["values", "indices"])

MODE_BLOCK_M = 16
MODE_BLOCK_N = 128
MODE_NUM_WARPS = 4
MODE_NUM_STAGES = 1


@libentry()
@triton.jit
def mode_kernel(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    row_mask = rows < M

    first_offset = rows * N
    cur_value = tl.load(sorted_inp + first_offset, mask=row_mask, other=0.0)
    cur_index = tl.load(sorted_indices + first_offset, mask=row_mask, other=0)
    cur_count = tl.full([BLOCK_M], 1, dtype=tl.int32)
    best_value = cur_value
    best_index = cur_index
    best_count = tl.full([BLOCK_M], 1, dtype=tl.int32)

    for start_n in range(1, N, BLOCK_N):
        for n_delta in tl.static_range(0, BLOCK_N):
            n_offset = start_n + n_delta
            valid = row_mask & (n_offset < N)
            offset = rows * N + n_offset
            value = tl.load(sorted_inp + offset, mask=valid, other=0.0)
            index = tl.load(sorted_indices + offset, mask=valid, other=0)

            same = valid & (value == cur_value)
            next_count = tl.where(same, cur_count + 1, 1)
            new_run = valid & ~same

            cur_value = tl.where(new_run, value, cur_value)
            cur_index = tl.where(valid, index, cur_index)
            cur_count = tl.where(valid, next_count, cur_count)

            better = valid & (next_count > best_count)
            best_value = tl.where(better, cur_value, best_value)
            best_index = tl.where(better, cur_index, best_index)
            best_count = tl.where(better, next_count, best_count)

    tl.store(out_value + rows, best_value, mask=row_mask)
    tl.store(out_index + rows, best_index, mask=row_mask)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_MTHREADS MODE")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    M = inp.numel() // N

    sorted_inp, sorted_indices = gems_sort(inp, dim=dim)

    sorted_inp = torch.movedim(sorted_inp, dim, -1).contiguous()
    sorted_indices = torch.movedim(sorted_indices, dim, -1).contiguous()

    sorted_flat = sorted_inp.reshape(M, N)
    indices_flat = sorted_indices.reshape(M, N)

    out_value = torch.empty(M, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(M, dtype=torch.int64, device=inp.device)

    grid = (triton.cdiv(M, MODE_BLOCK_M),)
    with torch_device_fn.device(inp.device):
        mode_kernel[grid](
            sorted_flat,
            indices_flat,
            out_value,
            out_index,
            M,
            N,
            MODE_BLOCK_M,
            MODE_BLOCK_N,
            num_warps=MODE_NUM_WARPS,
            num_stages=MODE_NUM_STAGES,
        )

    out_shape = shape.copy()
    out_shape[dim] = 1
    out_value = out_value.reshape(out_shape)
    out_index = out_index.reshape(out_shape)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    return ModeOut(values=out_value, indices=out_index)


__all__ = ["mode"]
