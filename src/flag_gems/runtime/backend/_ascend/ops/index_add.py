import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def index_add_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    src_ptr,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(axis=0)
    pid_n = tle.program_id(axis=1)

    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    inp_off = rows_offset * inp_len + cur_indices
    cur_inp = tl.load(inp_ptr + inp_off, mask=block_mask, other=0.0)

    src_off = rows_offset * N + cols_offset
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    result = cur_inp + alpha * cur_src
    tl.store(out_ptr + inp_off, result, mask=block_mask)


def _get_block_config(M, N):
    BLOCK_M = 4 if M < 4096 else 8
    BLOCK_N = max(4, min(512, triton.next_power_of_2(N)))
    return BLOCK_M, BLOCK_N


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ASCEND INDEX ADD")

    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    out = inp.clone()

    BLOCK_M, BLOCK_N = _get_block_config(M, N)
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](
            inp,
            out,
            index,
            src,
            M,
            N,
            alpha,
            inp_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ASCEND INDEX ADD_")

    index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    final_dim = inp.ndim - 1

    if dim != final_dim:
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)
        out_work = inp_work.clone()

        BLOCK_M, BLOCK_N = _get_block_config(M, N)
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](
                inp_work,
                out_work,
                index,
                src_work,
                M,
                N,
                alpha,
                inp_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        order = list(range(out_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = out_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        inp_contig = inp.contiguous()
        out_contig = inp_contig.clone()

        BLOCK_M, BLOCK_N = _get_block_config(M, N)
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](
                inp_contig,
                out_contig,
                index,
                src,
                M,
                N,
                alpha,
                inp_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        if inp.is_contiguous():
            inp.copy_(out_contig)
        else:
            inp.copy_(out_contig)

    return inp
