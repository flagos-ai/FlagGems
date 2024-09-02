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
def gather_kernel(
    inp,
    inp_offsets,
    out,
    index,
    idx_offsets,
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

        cur_index = tl.load(index + idx_indices, mask=mask, other=0)
        inp_indices += cur_index * stride_dim
        cur_inp = tl.load(inp + inp_indices, mask=mask, other=0)

        tl.store(out + idx_indices, cur_inp, mask=mask)


def gather(inp, dim, index, sparse_grad=False):
    logging.debug("GEMS GATHER")
    assert (
        inp.ndim == index.ndim
    ), "self and index should all have the same number of dimensions"
    assert (
        ((0 <= index.size(i) and index.size(i) <= inp.size(i)) or i == dim)
        for i in range(0, index.ndim)
    ), "index.size(d) <= self.size(d) for all dimensions d != dim"
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    inp = inp.contiguous()
    index = index.contiguous()
    out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

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
    gather_kernel[grid](
        inp, inp_offsets, out, index, idx_offsets, M, N, inp.stride(dim)
    )
    return out


def gather_out(inp, dim, index, sparse_grad=False, out=None):
    logging.debug("GEMS GATHER OUT")
    assert (
        inp.ndim == index.ndim and inp.ndim == out.ndim
    ), "self, index and out (if it is a Tensor) should all have the same number of dimensions"
    assert (
        (0 <= index.size(i) and index.size(i) <= out.size(i))
        for i in range(0, index.ndim)
    ), "index.size(d) <= out.size(d) for all dimensions d"
    assert (
        ((0 <= index.size(i) and index.size(i) <= inp.size(i)) or i == dim)
        for i in range(0, index.ndim)
    ), "index.size(d) <= self.size(d) for all dimensions d != dim"
    assert index.shape == out.shape, "out will have the same shape as index"
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    inp = inp.contiguous()
    index = index.contiguous()
    out = out.contiguous()

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
    gather_kernel[grid](
        inp, inp_offsets, out, index, idx_offsets, M, N, inp.stride(dim)
    )
    return out
