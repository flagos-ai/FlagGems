import logging

import torch

from flag_gems.ops.nonzero_static import _nonzero_static_impl

from ..utils import CORE_NUM
from .cumsum import cumsum

logger = logging.getLogger(__name__)

ASCEND_SINGLE_BLOCK_MAX_NUMEL = 8192
ASCEND_BLOCK_SIZE = 4096
ASCEND_REDUCED_BLOCK_SIZE = 1024
ASCEND_REDUCED_BLOCK_RATIO = 128
ASCEND_SMALL_COUNTS_MAX_BLOCKS = 256
ASCEND_SCAN_GROUP_SIZE = 64
ASCEND_SPARSE_BLOCK_SIZE = 2048
ASCEND_SPARSE_GROUP_RATIO = 1024
ASCEND_SPARSE_SCAN_GROUP_SIZE = 64
ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ = 1
ASCEND_SPARSE_COUNT_GROUP_BLOCKS = 8
ASCEND_SPARSE_MAX_COUNT_GROUPS = 4096
ASCEND_SPARSE_MAX_NUMEL = (
    ASCEND_SPARSE_BLOCK_SIZE
    * ASCEND_SPARSE_COUNT_GROUP_BLOCKS
    * ASCEND_SPARSE_MAX_COUNT_GROUPS
)


def _get_block_size(input, size):
    if input.dim() > 4:
        return ASCEND_REDUCED_BLOCK_SIZE
    if (
        size >= 4096
        and input.numel() >= size * ASCEND_REDUCED_BLOCK_RATIO
        and input.numel() < size * ASCEND_SPARSE_GROUP_RATIO
    ):
        return ASCEND_BLOCK_SIZE
    if size > 0 and input.numel() >= size * ASCEND_REDUCED_BLOCK_RATIO:
        return ASCEND_SPARSE_BLOCK_SIZE
    return ASCEND_BLOCK_SIZE


def _use_sparse_groups(input, size):
    return (
        input.dim() <= 4
        and size > 0
        and input.numel() <= ASCEND_SPARSE_MAX_NUMEL
        and input.numel() >= size * ASCEND_SPARSE_GROUP_RATIO
    )


def _get_sparse_scan_group_size(input):
    return ASCEND_SPARSE_SCAN_GROUP_SIZE


def _use_small_linear_output(input, size):
    return (
        1 < input.dim() <= 4
        and size >= 4096
        and input.numel() >= size * ASCEND_REDUCED_BLOCK_RATIO
        and input.numel() < size * ASCEND_SPARSE_GROUP_RATIO
    )


def nonzero_static(input: torch.Tensor, *, size: int, fill_value: int = -1):
    logger.debug("GEMS_ASCEND NONZERO_STATIC")
    return _nonzero_static_impl(
        input,
        size=size,
        fill_value=fill_value,
        cumsum_fn=cumsum,
        transpose_out=False,
        block_size=_get_block_size(input, size),
        single_block_max_numel=ASCEND_SINGLE_BLOCK_MAX_NUMEL,
        small_counts_max_blocks=ASCEND_SMALL_COUNTS_MAX_BLOCKS,
        max_programs=CORE_NUM,
        scan_group_size=ASCEND_SCAN_GROUP_SIZE,
        sparse_scan_group_size=_get_sparse_scan_group_size(input),
        use_sparse_groups=_use_sparse_groups(input, size),
        sparse_group_select_max_nnz=ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ,
        sparse_count_group_blocks=ASCEND_SPARSE_COUNT_GROUP_BLOCKS,
        small_counts_linear_output=_use_small_linear_output(input, size),
        use_bfloat16_bits=True,
    )


def nonzero_static_out(
    input: torch.Tensor, *, size: int, fill_value: int = -1, out: torch.Tensor
):
    logger.debug("GEMS_ASCEND NONZERO_STATIC_OUT")
    return _nonzero_static_impl(
        input,
        size=size,
        fill_value=fill_value,
        out=out,
        cumsum_fn=cumsum,
        transpose_out=False,
        block_size=_get_block_size(input, size),
        single_block_max_numel=ASCEND_SINGLE_BLOCK_MAX_NUMEL,
        small_counts_max_blocks=ASCEND_SMALL_COUNTS_MAX_BLOCKS,
        max_programs=CORE_NUM,
        scan_group_size=ASCEND_SCAN_GROUP_SIZE,
        sparse_scan_group_size=_get_sparse_scan_group_size(input),
        use_sparse_groups=_use_sparse_groups(input, size),
        sparse_group_select_max_nnz=ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ,
        sparse_count_group_blocks=ASCEND_SPARSE_COUNT_GROUP_BLOCKS,
        small_counts_linear_output=_use_small_linear_output(input, size),
        use_bfloat16_bits=True,
    )
