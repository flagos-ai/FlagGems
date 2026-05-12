import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


config_ = CodeGenConfig(
    2048,
    tuple([48, 1, 1]),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS_ASCEND DIAGONAL BACKWARD")
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    copy_func.instantiate(grad_output.ndim)(grad_output, out0=diag)
    return grad_input
