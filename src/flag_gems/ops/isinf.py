import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

try:
    import torch_npu  # noqa: F401
except:  # noqa: E722
    _isinf = tl_extra_shim.isinf

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isinf_func(x):
    return _isinf(x.to(tl.float32))


def isinf(A):
    logger.debug("GEMS ISINF")
    return isinf_func(A)
