import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True])
@triton.jit
def where_self(self, condition, other):
    return tl.where(condition, self, other)


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def where_self_scalar(other, condition, self):
    return tl.where(condition, self, other)


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def where_other_scalar(self, condition, other):
    return tl.where(condition, self, other)


def where(condition, self, other):
    logging.debug("GEMS WHERE")
    if isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor):
        return where_self(self, condition, other)
    elif isinstance(other, torch.Tensor):
        return where_self_scalar(other, condition, self)
    elif isinstance(self, torch.Tensor):
        return where_other_scalar(self, condition, other)
