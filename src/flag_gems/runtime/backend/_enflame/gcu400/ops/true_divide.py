# Copyright 2026, The FlagOS Contributors.
# GCU400-specialized true division: contiguous same-shape tensors go through a hand-written Triton kernel, everything
# else falls back to the generic implementation.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.div import true_divide as _generic_true_divide
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# GCU caps grid.x at 65535, so the host widens BLOCK until the grid fits and the
# kernel keeps a grid-stride loop so an oversized problem still runs instead of
# failing to launch.
_MAX_GRID = 65535
# Wide tiles are what reach memory bandwidth on GCU400: measured 1.2 TB/s at
# BLOCK=65536 for fp32 versus 0.5 TB/s at 8192.  The fp32 tile is the largest that
# keeps the three staged tiles inside the 917504-byte dsm budget, so 16-bit
# element types double it.
_TARGET_TILE_BYTES = 65536 * 4


@libentry()
@triton.jit
def _tdiv_kernel(a_ptr, b_ptr, out_ptr, n_elements, n_iter, BLOCK: tl.constexpr):
    block_base = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    block_step = tl.num_programs(0) * BLOCK
    for block_index in range(n_iter):
        offsets = block_base + block_index * block_step
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask, other=0)
        b = tl.load(b_ptr + offsets, mask=mask, other=1)
        tl.store(out_ptr + offsets, a / b, mask=mask)


def _launch(a, b, out, block, warps):
    n = out.numel()
    if n == 0:
        return out
    target_block = _TARGET_TILE_BYTES // a.element_size()
    block = max(block, min(target_block, triton.next_power_of_2(n)))
    while triton.cdiv(n, block) > _MAX_GRID:
        block *= 2
    grid = triton.cdiv(n, block)
    n_iter = triton.cdiv(n, grid * block)
    _tdiv_kernel[(grid,)](a, b, out, n, n_iter, BLOCK=block, num_warps=warps)
    return out


def _eligible(a, b):
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape == b.shape
        and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def true_divide_tensor(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE")
    if _eligible(A, B):
        out = torch.empty_like(A)
        return _launch(A, B, out, 8192, 4)
    return _generic_true_divide(A, B)
