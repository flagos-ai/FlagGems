import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def cfggen():
    block_m = [1, 2, 4, 8]
    block_n = [128, 1024, 2048, 4096]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=1)
        for m in block_m
        for n in block_n
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def index_add_kernel(
    inp,
    out,
    index,
    src,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    rows_mask = rows_offsets < M
    index_mask = cols_offsets < N
    block_mask = rows_mask and index_mask

    cur_indices = tl.load(index + cols_offsets, mask=index_mask, other=0)
    inp_off = rows_offsets * inp_len + cur_indices[None, :]
    cur_inp = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    src_off = rows_offsets * N + cols_offsets[None, :]
    cur_src = tl.load(src + src_off, mask=block_mask, other=0.0)
    cur_inp += alpha * cur_src

    tl.store(out + inp_off, cur_inp, mask=block_mask)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_CAMBRICON INDEX ADD")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device="cuda")
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N
    fine_dim = inp.ndim - 1
    if dim != fine_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)
    out = inp.clone()

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    index_add_kernel[grid](inp, out, index, src, M, N, alpha, inp_len)
    if dim != fine_dim:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, fine_dim)
        return out.permute(order).contiguous()
    else:
        return out
