import logging
from dataclasses import replace

import triton
import triton.language as tl

from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import get_codegen_config

device = device.name
config_ = replace(get_codegen_config(), max_tile_size=4096 * 2)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logging.debug("SOPHGO GEMS EQ")
    return eq_func(A, B)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq_scalar(A, B):
    logging.debug("SOPHGO GEMS EQ SCALAR")
    return eq_func_scalar(A, B)
