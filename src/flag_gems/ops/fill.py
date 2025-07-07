import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


def fill_scalar(input, value):
    logger.debug("GEMS FILL (Dynamic)")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS FILL (Dynamic)")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


def fill_scalar_(self, value):
    logging.debug("GEMS FILL_SCALAR_")
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self
