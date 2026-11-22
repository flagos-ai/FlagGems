# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from flag_gems.ops.bmm_w8a8_fp8 import bmm_w8a8_fp8 as block_scaled_bmm

from .mm_w8a8_fp8 import mm_w8a8_fp8_out

logger = logging.getLogger(__name__)


def bmm_w8a8_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    block_size: tuple[int, int, int] = (128, 128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 BMM with channel scales when block_size is (1, 1, K)."""
    logger.debug("GEMS_MTHREADS BMM_W8A8_FP8")
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("FP8 BMM expects three-dimensional operands")
    batch, rows, reduction = A.shape
    if B.shape[:2] != (batch, reduction):
        raise ValueError("FP8 BMM operands must have matching batch and K dimensions")
    if tuple(block_size) != (1, 1, reduction):
        return block_scaled_bmm(A, B, A_scale, B_scale, block_size, out_dtype, out)

    columns = B.shape[2]
    if A_scale.shape != (batch, rows, 1) or B_scale.shape != (batch, 1, columns):
        raise ValueError("Channel scales must have shapes (B, M, 1) and (B, 1, N)")
    if out is None:
        out = torch.empty((batch, rows, columns), dtype=out_dtype, device=A.device)
    elif out.shape != (batch, rows, columns):
        raise ValueError("FP8 BMM output must have shape (B, M, N)")
    # Reuse FP8 accumulation, fused channel scaling and layout handling from MM.
    for index in range(batch):
        mm_w8a8_fp8_out(
            A[index],
            B[index],
            A_scale[index],
            B_scale[index],
            out_dtype=out_dtype,
            out=out[index],
        )
    return out
