import logging

import torch
import triton  # noqa: F401
import triton.language as tl  # noqa: F401
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=False,
    unroll_num=8,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _sym_constrain_range_for_size_copy(x):
    return x


def _extract_dep_token(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _functional_sym_constrain_range_for_size(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN _FUNCTIONAL_SYM_CONSTRAIN_RANGE_FOR_SIZE")
    # The functional variant returns a fresh dep_token carrying the data
    # dependency; the actual size constraint on the symint is a trace-time
    # no-op. We only need to hand back a copy of the dep_token tensor.
    tensor_arg = _extract_dep_token(args, kwargs)
    if tensor_arg is None:
        return args[0] if len(args) > 0 else None
    if tensor_arg.is_contiguous() and tensor_arg.numel() > 0:
        return _sym_constrain_range_for_size_copy(tensor_arg)
    return tensor_arg.clone()
