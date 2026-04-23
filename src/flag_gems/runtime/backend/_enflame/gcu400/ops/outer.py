import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit
def outer_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    NUM_COL_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_N)

    for row_base in tl.range(pid * ROWS_PER_GROUP, M, num_pids * ROWS_PER_GROUP):
        for cb in tl.range(0, NUM_COL_BLOCKS):
            cols = cb * BLOCK_N + col_offsets
            col_mask = cols < N
            b_vals = tl.load(b_ptr + cols, mask=col_mask)

            for r in tl.range(0, ROWS_PER_GROUP):
                row = row_base + r
                if row < M:
                    a_val = tl.load(a_ptr + row)
                    tl.store(out_ptr + row * N + cols, a_val * b_vals, mask=col_mask)


def _next_po2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def outer(inp, weight):
    logger.debug("GEMS OUTER")
    assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"

    M = inp.shape[0]
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=inp.dtype, device=inp.device)

    bpe = inp.element_size()
    max_block = 4096 if bpe > 2 else 16384
    BLOCK_N = min(_next_po2(N), max_block)
    BLOCK_N = max(BLOCK_N, 2048)

    ROWS_PER_GROUP = 32 if M * N > 1024 * 1024 else 16
    NUM_COL_BLOCKS = triton.cdiv(N, BLOCK_N)
    num_row_groups = triton.cdiv(M, ROWS_PER_GROUP)
    grid_size = min(num_row_groups, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        outer_kernel[(grid_size,)](
            inp, weight, out, M,
            N=N, NUM_COL_BLOCKS=NUM_COL_BLOCKS,
            BLOCK_N=BLOCK_N, ROWS_PER_GROUP=ROWS_PER_GROUP,
            num_stages=1,
        )

    return out
