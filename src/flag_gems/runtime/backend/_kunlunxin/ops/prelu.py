# Kunlunxin (XPU) override of prelu.
#
# The generic ops/prelu.py uses a hand-written @triton.jit kernel with a fixed
# BLOCK_SIZE=1024, no unrolling / vectorization, and runtime (non-specialized)
# n_elements/S/C args. On XPU this is catastrophically slow (per-channel path
# 9-126ms vs torch ~0.1ms, speedup 0.001-0.02) because:
#   * the per-channel alpha index `c = (offsets // S) % C` uses div/mod on
#     runtime values, which blocks OffsetAnalysis -> the alpha load becomes a
#     discrete scalar gather;
#   * the tiny 1024-tile with no unroll/vectorization leaves the XPU idle.
#
# PReLU is really a broadcasting pointwise op:
#   out = where(x >= 0, x, weight * x)   with weight broadcast along channel dim.
# Expressing it through the XPU-tuned pointwise_dynamic (which handles
# broadcasting + layout and emits vectorized, unrolled block-DMA kernels) both
# fixes correctness-of-speed and removes the div/mod gather.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

import flag_gems

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Same tuned recipe as clamp_max: pure memory-bound elementwise, so keep
# vectorization OPEN (wide vector DMA, packs fp16/bf16) for best bandwidth.
prelu_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=prelu_config,
)
@triton.jit
def prelu_func(x, w):
    return tl.where(x >= 0, x, w * x)


def prelu(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN PRELU")
    if len(args) >= 2:
        x, weight = args[0], args[1]
    else:
        x = kwargs.get("input", kwargs.get("self"))
        weight = kwargs.get("weight")
    if x is None or weight is None:
        raise ValueError("prelu expects (input, weight) as arguments.")

    if x.device.type != flag_gems.device or weight.device.type != flag_gems.device:
        raise AssertionError(f"Tensors must be {flag_gems.device} tensors.")

    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)

    x = x.contiguous()
    weight = weight.contiguous()

    if x.numel() == 0:
        return torch.empty_like(x)

    ndim = x.dim()
    if weight.numel() == 1:
        # Scalar weight: reshape to a rank-matching 1-element tensor so
        # pointwise_dynamic broadcasts it against every element of x.
        w = weight.reshape([1] * ndim) if ndim > 0 else weight.reshape([])
    else:
        if ndim == 0:
            raise AssertionError("Non-scalar weight provided for a 0-dim input.")
        C = x.shape[0] if ndim == 1 else x.shape[1]
        if weight.numel() != C:
            raise AssertionError(
                f"Weight numel ({weight.numel()}) must equal channel dimension "
                f"size ({C})."
            )
        # Broadcast weight along the channel dim: [C] -> [1, C, 1, ...].
        if ndim == 1:
            w_shape = [C]
        else:
            w_shape = [1, C] + [1] * (ndim - 2)
        w = weight.reshape(w_shape)

    return prelu_func(x, w)
