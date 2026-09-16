import logging

import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# absolute is an alias of abs; reuse abs's tuned recipe (memory async enabled).
_absolute_config = CodeGenConfig(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,  # Enable memory async for better overlap
)

# Small tensors (<= 8192 elements) go faster on a direct masked BLOCK-1024
# kernel (the vectorized 1D-tile path's coarser grid doubles launch count;
# same hybrid as _kunlunxin/ops/sign.py).
_SMALL_NUMEL = 8192


@triton.jit
def _absolute_small_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, tl.abs(x), mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=_absolute_config)
@triton.jit
def absolute_func(x):
    return tl.abs(x)


def absolute(A):
    logger.debug("GEMS_KUNLUNXIN ABSOLUTE")
    return absolute_func(A)


def absolute_(A):
    logger.debug("GEMS_KUNLUNXIN ABSOLUTE_")
    if A.numel() == 0:
        return A
    if A.numel() <= _SMALL_NUMEL and A.is_contiguous():
        _absolute_small_kernel[(triton.cdiv(A.numel(), 1024),)](
            A.view(-1), A.view(-1), A.numel(), BLOCK_SIZE=1024
        )
        return A
    absolute_func(A, out0=A)
    return A
