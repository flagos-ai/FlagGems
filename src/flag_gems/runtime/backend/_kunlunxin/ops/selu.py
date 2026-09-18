import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_ALPHA = tl.constexpr(1.6732632423543772848170429916717)
_SCALE = tl.constexpr(1.0507009873554804934193349852946)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def selu_func(x):
    x_fp32 = x.to(tl.float32)
    return _SCALE * tl.where(x_fp32 > 0, x_fp32, _ALPHA * (tl.exp(x_fp32) - 1.0))


_TIERS = (
    (16384, 2048, 4),
    (262144, 8192, 8),
    (None, 16384, 16),
)


@triton.jit
def selu_flat_kernel(
    A,
    O,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(A + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(A + offsets)

    x_f32 = x.to(tl.float32)
    x_neg = tl.minimum(x_f32, 0.0)
    y = _SCALE * (
        tl.maximum(x_f32, 0.0) + tl.minimum(_ALPHA * (tl.exp(x_neg) - 1.0), 0.0)
    )

    if NEED_MASK:
        tl.store(O + offsets, y.to(x.dtype), mask=mask)
    else:
        tl.store(O + offsets, y.to(x.dtype))


def _pick_tier(numel):
    for hi, block, warps in _TIERS:
        if hi is None or numel <= hi:
            return block, warps
    return 16384, 16


def _use_flat(A):
    return A.is_contiguous()


def _launch_flat(A, out):
    n_elements = A.numel()
    if n_elements == 0:
        return out
    block, warps = _pick_tier(n_elements)
    need_mask = (n_elements % block) != 0
    grid = (triton.cdiv(n_elements, block),)
    selu_flat_kernel[grid](
        A.reshape(-1),
        out,
        n_elements,
        BLOCK_SIZE=block,
        NEED_MASK=need_mask,
        num_warps=warps,
    )
    return out


def selu(A):
    logger.debug("GEMS_KUNLUNXIN SELU")
    if _use_flat(A):
        return _launch_flat(A, torch.empty_like(A))
    return selu_func(A)


def selu_(A):
    logger.debug("GEMS_KUNLUNXIN SELU_")
    if _use_flat(A):
        return _launch_flat(A, A)
    return selu_func(A, out0=A)
