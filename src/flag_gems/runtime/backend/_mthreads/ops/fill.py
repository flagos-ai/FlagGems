import logging

import torch

from flag_gems.ops.fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor_func,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


def _value_on_device(value):
    """Whether ``value`` resides on the GPU (musa) device.

    ``Tensor.is_cuda`` is unreliable on the mthreads backend: importing
    ``transformer_engine`` monkeypatches it to ``True`` for *every* tensor
    (including CPU ones), which makes the original ``not value.is_cuda``
    guard in the shared implementation misroute a CPU scalar value into the
    tensor kernel and crash with "Pointer argument ... cannot be accessed
    from Triton (cpu tensor?)". ``is_musa`` is not affected, so use it.
    """
    return value.is_musa


def fill_tensor(input, value):
    if not _value_on_device(value):
        return fill_scalar(input, value.item())
    logger.debug("GEMS MTHREADS FILL (Dynamic)")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS MTHREADS FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not _value_on_device(value):
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    if not _value_on_device(value):
        return fill_scalar_(self, value.item())
    logger.debug("GEMS MTHREADS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self
