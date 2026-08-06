import logging

import torch

from flag_gems.ops.binary_cross_entropy_with_logits import (
    binary_cross_entropy_with_logits as _generic_binary_cross_entropy_with_logits,
)

logger = logging.getLogger(__name__)


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_KUNLUNXIN BINARY_CROSS_ENTROPY_WITH_LOGITS")
    if self.dtype != torch.float32 or reduction != 2:
        return _generic_binary_cross_entropy_with_logits(
            self, target, weight, pos_weight, reduction
        )

    loss = _generic_binary_cross_entropy_with_logits(
        self, target, weight, pos_weight, 0
    ).reshape(-1)
    chunk_size = 65536
    full_chunks = loss.numel() // chunk_size
    if full_chunks:
        result = (
            loss[: full_chunks * chunk_size]
            .reshape(full_chunks, chunk_size)
            .sum(dim=1)
            .sum()
        )
    else:
        result = torch.zeros((), dtype=loss.dtype, device=loss.device)
    if full_chunks * chunk_size < loss.numel():
        result += loss[full_chunks * chunk_size :].sum()
    return result
