import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


config_ = CodeGenConfig(
    512,
    tuple([48, 1, 1]),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
    config=config_,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag


def polar(abs, angle):
    logger.debug("GEMS_ASCEND POLAR")
    output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)

    polar_kernel(abs, angle, out0=output[..., 0], out1=output[..., 1])

    return torch.view_as_complex(output)
