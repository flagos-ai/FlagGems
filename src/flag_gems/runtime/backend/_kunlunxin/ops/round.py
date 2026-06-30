import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.xpu.libdevice import rint as _rint

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


# rint(fp32) implements round-half-to-even, matching torch.round semantics.
# XPU libdevice rint only supports fp32, so always cast to fp32 for computation.
# The scale trick handles non-zero decimals: round(x, d) = rint(x * 10^d) / 10^d.
@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def round_func(x, scale):
    x_fp32 = x.to(tl.float32)
    return _rint(x_fp32 * scale) / scale


def _scale(decimals):
    return 10.0**decimals


def round(input, decimals=0):
    logger.debug("GEMS_KUNLUNXIN ROUND")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input.clone()
    if input.numel() == 0:
        return torch.empty_like(input)
    if not input.is_contiguous():
        input = input.contiguous()
    return round_func(input, _scale(decimals))


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS_KUNLUNXIN ROUND_OUT")
    if out is None:
        return round(input, decimals=decimals)
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        out.copy_(input)
        return out
    if input.numel() == 0:
        return out
    if not input.is_contiguous():
        input = input.contiguous()
    round_func(input, _scale(decimals), out0=out)
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN ROUND_")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input
    if input.numel() == 0:
        return input
    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )
    round_func(input, _scale(decimals), out0=input)
    return input
