import logging

import torch
import triton

from flag_gems.ops._nested_view_from_jagged import _nested_view_from_jagged

from .copy import _copy_flat_kernel, _pick_flat_block

logger = logging.getLogger(__name__)


def _nested_view_from_jagged_copy(
    self: torch.Tensor,
    offsets: torch.Tensor,
    dummy: torch.Tensor,
    lengths=None,
    ragged_idx=1,
    min_seqlen=None,
    max_seqlen=None,
):
    logger.debug("GEMS_KUNLUNXIN _NESTED_VIEW_FROM_JAGGED_COPY")
    src = self.contiguous() if not self.is_contiguous() else self
    values_copy = torch.empty_like(src)
    n = src.numel()

    if n > 0:
        item_size = src.element_size()
        if (
            item_size <= 4
            and (n * item_size) % 4 == 0
            and n * item_size >= 2**20
            and src.data_ptr() % 4 == 0
            and values_copy.data_ptr() % 4 == 0
        ):
            src_view = src.reshape(-1).view(torch.int32)
            dst_view = values_copy.reshape(-1).view(torch.int32)
            n32 = src_view.numel()
            block = _pick_flat_block(n32)
            _copy_flat_kernel[(triton.cdiv(n32, block),)](
                src_view,
                dst_view,
                n32,
                BLOCK_SIZE=block,
                NEED_MASK=(n32 % block != 0),
                num_warps=32,
                unroll_num=8,
                buffer_size_limit=1024,
            )
        else:
            block = _pick_flat_block(n)
            _copy_flat_kernel[(triton.cdiv(n, block),)](
                src,
                values_copy,
                n,
                BLOCK_SIZE=block,
                NEED_MASK=(n % block != 0),
                num_warps=32,
                unroll_num=8,
                buffer_size_limit=1024,
            )

    return _nested_view_from_jagged(
        values_copy, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen
    )


__all__ = ["_nested_view_from_jagged_copy"]
