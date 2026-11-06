# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable FP32 ordering key shared by the explicit radix TopK implementations."""

import triton
import triton.language as tl


@triton.jit
def topk_stable_fp32_key(x):
    """Return the unchanged monotone bit key, normalizing signed zeros.

    This is a Triton compiler helper, not a host ATen operation. NaN payload
    ordering follows the existing bit-key behavior; callers define filtering.
    """
    bits = x.to(tl.uint32, bitcast=True)
    bits = tl.where(x == 0.0, tl.full(x.shape, 0, tl.uint32), bits)
    mask = tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return bits ^ mask


__all__ = ["topk_stable_fp32_key"]
