import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_extension import fmod as _fmod

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func(x, y):
    # Convert to float32 for computation to avoid libdevice float16/bfloat16 issues
    dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    result = _fmod(x_fp32, y_fp32)
    return result.to(dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def fmod_func_tensor_scalar(x, y):
    # Convert to float32 for computation to avoid libdevice float16/bfloat16 issues
    dtype = x.dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    result = _fmod(x_fp32, y_fp32)
    return result.to(dtype)


def fmod(A, B):
    logger.debug("GEMS FMOD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return fmod_func(A, B)
    elif isinstance(A, torch.Tensor):
        return fmod_func_tensor_scalar(A, B)
    else:
        # Both scalar - fallback to PyTorch
        return torch.fmod(torch.tensor(A), B)


def fmod_(A, B):
    logger.debug("GEMS FMOD_")
    if isinstance(B, torch.Tensor):
        return fmod_func(A, B, out0=A)
    else:
        return fmod_func_tensor_scalar(A, B, out0=A)
