import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry, offsetCalculator, restride_dim


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def slice_scatter_kernel(
    inp,
    inp_indices,
    src,
    src_offsets,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M

    for off in range(0, N, BLOCK_N):
        cols_offsets = off + tl.arange(0, BLOCK_N)[None, :]
        cols_mask = cols_offsets < N

        offsets = rows_offsets * N + cols_offsets
        mask = rows_mask and cols_mask

        indices = tl.load(inp_indices + offsets, mask=mask, other=0)
        src_indices = tl.load(src_offsets + offsets, mask=mask, other=0)
        cur_src = tl.load(src + src_indices, mask=mask, other=0)

        tl.store(inp + indices, cur_src, mask=mask)


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logging.debug("GEMS SLICE_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim
    out = inp.clone().contiguous()
    src = src.contiguous()
    size_dim = inp.size(dim)

    if start is None:
        start = 0
    if end is None:
        end = size_dim

    range = end - start
    if end < start:
        range = 0
    elif (end - start) > size_dim:
        range = size_dim
        start = 0
        end = size_dim

    if range == 0:
        return out

    valid_shape = list(inp.shape)
    valid_shape[dim] = (range + (step - 1)) // step
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    storage_offset = out.storage_offset() + start * out.stride(dim)
    out_strided = restride_dim(out, dim, valid_shape, step, storage_offset)
    idx = torch.arange(0, src.numel(), device=inp.device).reshape(valid_shape)
    strides = list(out.stride())
    strides[dim] *= step
    indices = (
        offsetCalculator(out_strided, idx, strides, dim, isInp=False) + storage_offset
    )
    src_offsets = offsetCalculator(src, idx, src.stride(), dim, isInp=False)

    N = valid_shape[src.ndim - 1]
    M = src.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    slice_scatter_kernel[grid](out, indices, src, src_offsets, M, N)

    return out
