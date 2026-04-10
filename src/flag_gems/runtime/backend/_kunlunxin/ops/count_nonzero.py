import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

cluster_num = 12
core_num = 64
buf_len_per_core = 2048


def heur_m_block_size(args):
    return triton.next_power_of_2(min(triton.cdiv(args["M"], cluster_num), core_num))


def heur_n_block_size(args):
    return triton.next_power_of_2(min(args["N"], 512))


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def count_nonzero_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Use int32 for faster intermediate counting
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0)
        _count += (a != 0).to(tl.int32)

    count = tl.sum(_count, axis=1).to(tl.int64)
    tl.store(out, count[:, None], row_mask)


@libentry()
@triton.jit
def count_nonzero_kernel_1d_parallel(
    inp,
    partial_out,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pids = tle.num_programs(0)

    # Use int32 for faster intermediate counting
    _count = tl.zeros([BLOCK_N], dtype=tl.int32)
    for off in range(pid * BLOCK_N, N, num_pids * BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        a = tl.load(inp + cols, col_mask, other=0)
        _count += (a != 0).to(tl.int32)

    count = tl.sum(_count, axis=0).to(tl.int64)
    tl.store(partial_out + pid, count)


@libentry()
@triton.jit
def reduce_partial_counts(
    partial_in,
    out,
    num_partials,
    BLOCK: tl.constexpr,
):
    _sum = tl.zeros([BLOCK], dtype=tl.int64)
    for off in range(0, num_partials, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < num_partials
        vals = tl.load(partial_in + idx, mask, other=0)
        _sum += vals

    total = tl.sum(_sum, axis=0)
    tl.store(out, total)


def count_nonzero(x, dim=None):
    logger.debug("GEMS COUNT NONZERO")

    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = x.shape
        numel = x.numel()
        # permute
        x = dim_compress(x, dim)
        x = x.contiguous().flatten()
        # 2D count_nonzero
        out_shape = list(shape)
        del out_shape[dim]
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)
        N = shape[dim]
        M = triton.cdiv(numel, shape[dim])

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            count_nonzero_kernel_dim[grid](x, out, M, N, buffer_size_limit=buf_len_per_core)
        return out
    else:
        # 1D count_nonzero with parallel reduction
        x = x.contiguous().flatten()
        numel = x.numel()
        out = torch.zeros(1, dtype=torch.int64, device=x.device)

        # Use larger block size for better memory throughput
        BLOCK_N = 2048
        # Use fewer blocks to reduce kernel launch and reduction overhead
        num_blocks = min(cluster_num, triton.cdiv(numel, BLOCK_N))
        num_blocks = max(1, num_blocks)

        with torch_device_fn.device(x.device):
            if num_blocks == 1:
                # Small tensor: single block
                count_nonzero_kernel_1d_parallel[(1,)](
                    x, out, numel, BLOCK_N=BLOCK_N, buffer_size_limit=buf_len_per_core
                )
            else:
                # Large tensor: parallel reduction
                partial = torch.zeros(num_blocks, dtype=torch.int64, device=x.device)
                count_nonzero_kernel_1d_parallel[(num_blocks,)](
                    x, partial, numel, BLOCK_N=BLOCK_N, buffer_size_limit=buf_len_per_core
                )
                REDUCE_BLOCK = triton.next_power_of_2(num_blocks)
                reduce_partial_counts[(1,)](
                    partial, out, num_blocks, BLOCK=REDUCE_BLOCK, buffer_size_limit=buf_len_per_core
                )

        return out[0]