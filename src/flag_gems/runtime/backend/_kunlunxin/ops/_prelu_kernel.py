# Kunlunxin (XPU) override of _prelu_kernel.
#
# aten::_prelu_kernel(x, weight) computes the elementwise
#   out = where(x >= 0, x, weight * x)
# with weight broadcastable against x. The test/benchmark path always feeds a
# weight of the same shape/stride as x, which takes the pointwise_dynamic fast
# path (dimension collapse to a flat 1D task space).
#
# Two XPU-specific findings drive this implementation:
#
# 1. The generic flag_gems pointwise_dynamic partitions the flat task space
#    with `num_ctas = min(65536, num_tiles)` and 512-element tiles, i.e. tens
#    of thousands of tiny programs on XPU (launch-bound; measured ~2.7 GB/s,
#    70ms for 16M fp16 elements vs torch 0.06ms). The XPU codegen
#    (pointwise_dynamic in this folder) partitions into a fixed 12-CTA grid
#    with one large vectorized tile per CTA, which runs at block-DMA rates.
#
# 2. `tl.where(x >= 0, x, w * x)` (select with a tensor RHS) is ~4x slower
#    than the same kernel expressed without select: on XPU the vectorized
#    select blocks the memory pipeline (0.278ms vs clamp/relu-class 0.05ms on
#    16M fp16). Clamp/leaky-relu show min/max + mul/add formulations run at
#    full bandwidth, so prelu is rewritten as the algebraically identical,
#    select-free, single-rounding expression
#      p       = min(x, 0)          # x<0 -> x, else exact 0
#      out     = (x - p) + p * w   # x>=0 -> x (exact), x<0 -> w*x (exact)
#    For finite inputs this is bit-identical to the ATen formula; NaN inputs
#    still produce NaN (NaN - min(NaN,0) = NaN). Note the select version also
#    evaluates `w * x` for NaN weight with x>=0, which the select-free form
#    can not reproduce (same limitation class as the min/max-based relu/clamp
#    family on this backend).
import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _prelu_kernel_func(x, weight):
    x_neg = tl.minimum(x, 0.0)
    return (x - x_neg) + x_neg * weight


def _prelu_kernel(A, B):
    logger.debug("GEMS_KUNLUNXIN _PRELU_KERNEL")
    return _prelu_kernel_func(A, B)