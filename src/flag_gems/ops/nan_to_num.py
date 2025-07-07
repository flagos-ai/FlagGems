import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_isnan = tl_extra_shim.isnan

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    x_nan = _isnan(x.to(tl.float32))
    x_posinf = x == float("inf")
    x_neginf = x == -float("inf")
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x


# nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS NAN_TO_NUM TENSOR")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return nan_to_num_func(A, nan, posinf, neginf)
