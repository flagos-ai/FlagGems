# Copyright 2026, The FlagOS Contributors.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def adaptive_attention_span_forward(x):
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + tl_extra_shim.exp2(-x.to(tl.float32) * log2e))


def adaptive_attention_span(A):
    logger.debug("GEMS ADAPTIVE_ATTENTION_SPAN")
    if runtime.device.vendor_name == "metax" or not A.is_cuda:
        return torch.sigmoid(A)
    return adaptive_attention_span_forward(A)


__all__ = ["adaptive_attention_span"]
