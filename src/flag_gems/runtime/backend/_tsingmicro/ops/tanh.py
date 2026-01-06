import logging

import triton
import triton.language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils import tl_extra_shim

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)
try:
    import torch_npu  # noqa: F401
except:  # noqa: E722
    pow = tl_extra_shim.pow
_tanh = tl_extra_shim.tanh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=my_config)
@triton.jit
def tanh_kernel(x):
    return _tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=my_config)
@triton.jit
def tanh_backward_kernel(y, dy):
    y = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y * y)


def tanh(self):
    logger.debug("GEMS TANH FORWARD")
    out = tanh_kernel(self)
    return out


def tanh_backward(grad_output, output):
    logger.debug("GEMS TANH BACKWARD")
    in_grad = tanh_backward_kernel(output, grad_output)
    return in_grad


def tanh_(A):
    logger.debug("GEMS TANH_ FORWARD")
    out = tanh_kernel(A, out0=A)
    return out
