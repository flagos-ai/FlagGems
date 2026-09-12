import logging

import torch
import triton  # noqa: F401
import triton.language as tl  # noqa: F401
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def alias_copy_func(x):
    return x


def alias_copy(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ALIAS_COPY")
    if x.numel() == 0:
        return torch.empty_like(x)
    # An alias copy is a plain move, so let tle.gpu do it; the pointwise kernel
    # stays as the fallback for what tle cannot express (e.g. bool).
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tle_copy(x, out):
        return out
    return alias_copy_func(x)


def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ALIAS_COPY_OUT")
    if x.dtype != out.dtype:
        raise RuntimeError("alias_copy_out: dtype of input and output must match.")
    if x.numel() != out.numel():
        raise RuntimeError(
            "alias_copy_out: input and output must have the same number of elements."
        )
    if x.device != out.device:
        raise RuntimeError(
            "alias_copy_out: input and output must be on the same device."
        )
    if out.numel() == 0:
        return out
    if tle_copy(x, out):
        return out
    alias_copy_func(x, out0=out)
    return out
