"""Kunlunxin special_erfinv (aten::special_erfinv) vendor override.

special_erfinv is mathematically identical to torch.erfinv, so this module
delegates to the tuned erfinv override instead of maintaining a separate
polynomial kernel. The standalone poly kernel used here previously lowered
poorly on XPU (~330ms/16.7M vs erfinv's ~0.3-0.5ms); reusing erfinv keeps a
single tuned implementation and stays in-domain accurate.
"""

import logging

import torch

from .erfinv import _launch_erfinv, erfinv, erfinv_

logger = logging.getLogger(__name__)


def special_erfinv(x: torch.Tensor):
    """Special erfinv function (aten::special_erfinv)."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERFINV")
    return erfinv(x)


def special_erfinv_out(x: torch.Tensor, out: torch.Tensor):
    """Special erfinv out function (aten::special_erfinv.out)."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERFINV_OUT")
    if out.shape != x.shape:
        out.resize_(x.shape)
    assert out.dtype == x.dtype, "out tensor must have the same dtype as input"
    x_in = x if x.is_contiguous() else x.contiguous()
    if out.is_contiguous():
        _launch_erfinv(x_in, out)
        return out
    tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
    _launch_erfinv(x_in, tmp)
    out.copy_(tmp)
    return out


def special_erfinv_(x: torch.Tensor):
    """Special erfinv in-place function (aten::special_erfinv_)."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ERFINV_")
    return erfinv_(x)
