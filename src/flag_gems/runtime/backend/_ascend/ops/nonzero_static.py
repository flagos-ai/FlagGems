import logging

import torch

from flag_gems.ops.nonzero_static import _nonzero_static_impl

from ..utils import CORE_NUM
from .cumsum import cumsum

logger = logging.getLogger(__name__)

ASCEND_SINGLE_BLOCK_MAX_NUMEL = 8192
ASCEND_BLOCK_SIZE = 4096
ASCEND_SMALL_COUNTS_MAX_BLOCKS = 256


def nonzero_static(input: torch.Tensor, *, size: int, fill_value: int = -1):
    logger.debug("GEMS_ASCEND NONZERO_STATIC")
    return _nonzero_static_impl(
        input,
        size=size,
        fill_value=fill_value,
        cumsum_fn=cumsum,
        transpose_out=False,
        block_size=ASCEND_BLOCK_SIZE,
        single_block_max_numel=ASCEND_SINGLE_BLOCK_MAX_NUMEL,
        small_counts_max_blocks=ASCEND_SMALL_COUNTS_MAX_BLOCKS,
        max_programs=CORE_NUM,
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
        block_size=ASCEND_BLOCK_SIZE,
        single_block_max_numel=ASCEND_SINGLE_BLOCK_MAX_NUMEL,
        small_counts_max_blocks=ASCEND_SMALL_COUNTS_MAX_BLOCKS,
        max_programs=CORE_NUM,
    )
