import logging

import torch
import triton

from flag_gems.ops.sort import sort_kernel, sort_stable

logger = logging.getLogger(__name__)


def argsort(inp, dim=-1, descending=False):
    """Returns the indices that sort a tensor along a given dimension.

    This is equivalent to calling torch.sort and returning only the indices.

    Performance Notes:
    - For small N (≤4096): Uses bitonic sort (single kernel launch)
      Speedup: ~2.5-3x for N=64-256
    - For large N (>4096): Uses radix sort (multiple passes)
      Current performance: 0.1-0.3x slower than PyTorch for N≥1024
      TODO: Optimize radix sort for large N scenarios
      Benchmark results (N=1024-262144):
        - [1024, 1024]: 0.4-0.7x speedup
        - [4096, 4096]: 0.1-0.2x speedup
        - [1024, 65536]: 0.2-0.3x speedup
    """
    logger.debug("GEMS ARGSORT")

    # For small N, use bitonic sort (single kernel launch)
    # For large N, use radix sort (multiple passes)
    if dim < 0:
        dim = dim + inp.ndim

    sort_elem_cnt = inp.shape[dim]

    # Use bitonic sort for small sizes (faster single-kernel approach)
    if sort_elem_cnt <= 4096:
        if dim != inp.ndim - 1:
            inp = torch.movedim(inp, dim, -1).contiguous()
        else:
            inp = inp.contiguous()

        N = inp.shape[-1]
        M = inp.numel() // N

        out = torch.empty_like(inp)
        out_index = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)

        BLOCK_SIZE = triton.next_power_of_2(N)
        IS_FLOAT = inp.dtype.is_floating_point

        grid = lambda meta: (M,)
        sort_kernel[grid](
            inp,
            out,
            out_index,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            DESCENDING=descending,
            IS_FLOAT=IS_FLOAT,
        )

        if dim != inp.ndim - 1:
            out_index = torch.movedim(out_index, -1, dim)

        return out_index
    else:
        # Use radix sort for large sizes
        _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
        return indices
