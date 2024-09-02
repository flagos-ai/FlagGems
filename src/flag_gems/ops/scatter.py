import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry, offset_calculator, restride_dim


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def scatter_kernel(
    inp_offsets,
    src,
    index,
    idx_offsets,
    out,
    M,
    N,
    stride_dim,
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

        inp_indices = tl.load(inp_offsets + offsets, mask=mask, other=0)
        idx_indices = tl.load(idx_offsets + offsets, mask=mask, other=0)

        cur_src = tl.load(src + idx_indices, mask=mask, other=0)
        cur_index = tl.load(index + idx_indices, mask=mask, other=0)

        inp_indices += cur_index * stride_dim
        tl.store(out + inp_indices, cur_src, mask=mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def scatter_add_kernel(
    inp,
    inp_offsets,
    src,
    index,
    idx_offsets,
    out,
    M,
    N,
    stride_dim,
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

        inp_indices = tl.load(inp_offsets + offsets, mask=mask, other=0)
        idx_indices = tl.load(idx_offsets + offsets, mask=mask, other=0)

        cur_src = tl.load(src + idx_indices, mask=mask, other=0).to(tl.float32)
        cur_index = tl.load(index + idx_indices, mask=mask, other=0)

        inp_indices += cur_index * stride_dim
        cur_inp = tl.load(inp + inp_indices, mask=mask, other=0).to(tl.float32)
        res = cur_inp + cur_src
        tl.store(out + inp_indices, res, mask=mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def scatter_mul_kernel(
    inp,
    inp_offsets,
    src,
    index,
    idx_offsets,
    out,
    M,
    N,
    stride_dim,
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

        inp_indices = tl.load(inp_offsets + offsets, mask=mask, other=0)
        idx_indices = tl.load(idx_offsets + offsets, mask=mask, other=0)

        cur_src = tl.load(src + idx_indices, mask=mask, other=0).to(tl.float32)
        cur_index = tl.load(index + idx_indices, mask=mask, other=0)

        inp_indices += cur_index * stride_dim
        cur_inp = tl.load(inp + inp_indices, mask=mask, other=0).to(tl.float32)
        res = cur_inp * cur_src
        tl.store(out + inp_indices, res, mask=mask)


def scatter(inp, dim, index, src, reduction=None):
    assert (
        inp.ndim == index.ndim and inp.ndim == src.ndim
    ), "self, index and src (if it is a Tensor) should all have the same number of dimensions"
    assert (
        (0 <= index.size(i) and index.size(i) <= src.size(i))
        for i in range(0, index.ndim)
    ), "index.size(d) <= src.size(d) for all dimensions d"
    assert (
        ((0 <= index.size(i) and index.size(i) <= inp.size(i)) or i == dim)
        for i in range(0, index.ndim)
    ), "index.size(d) <= self.size(d) for all dimensions d != dim"
    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()
    out = inp.clone()

    src_strided = src.as_strided(index.shape, src.stride()).contiguous()
    inp_strided = restride_dim(inp, dim, index.shape)
    # FIXME: Are there any other way to get the "flatten offset" of a tensor?
    idx = torch.arange(0, index.numel(), device=inp.device).reshape(index.shape)
    # Temporarily call offsetCalculator() outside the block(although it can actually proceed in parallel),
    # because the triton jit.function cannot accept Tuple as input in version 2.2.0(in 3.0.0, it's available),
    # and we do need **the whole stride[]** to accomplish this calculation!
    # FIXME: If stride[] can be wholely passed to triton jit.function, we can do this calculation in the kernel
    # so that the offset calculation can proceed in parallel
    inp_offsets = offset_calculator(inp_strided, idx, inp.stride(), dim, isInp=True)
    idx_offsets = offset_calculator(index, idx, index.stride(), dim, isInp=False)
    N = list(index.shape)[index.ndim - 1]
    M = index.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    if reduction is None:
        scatter_kernel[grid](
            inp_offsets,
            src_strided,
            index,
            idx_offsets,
            out,
            M,
            N,
            inp.stride(dim),
        )
    elif reduction == "add":
        scatter_add_kernel[grid](
            inp,
            inp_offsets,
            src_strided,
            index,
            idx_offsets,
            out,
            M,
            N,
            inp.stride(dim),
        )
    elif reduction == "multiply":
        scatter_mul_kernel[grid](
            inp,
            inp_offsets,
            src_strided,
            index,
            idx_offsets,
            out,
            M,
            N,
            inp.stride(dim),
        )
    return out


def scatter_src(inp, dim, index, src):
    logging.debug("GEMS SCATTER SRC")
    return scatter(inp, dim, index, src)


def scatter_reduce(inp, dim, index, src, reduce):
    logging.debug("GEMS SCATTER REDUCE")
    # TODO: As is shown in PyTorch's document(torch.Tensor.scatter_reduce_),
    # this function is still **in beta** and may change in the near future.
    # So for now, we're just going to stick with the original "add" and "multiply" parameters.
    # Maybe we can add reduction options like "sum", "prod", "mean", "amax" and "amin" in the future.
    if reduce == "add":
        return scatter(inp, dim, index, src, reduction="add")
    elif reduce == "multiply":
        return scatter(inp, dim, index, src, reduction="multiply")
