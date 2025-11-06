import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.tensor_wrapper import StridedBuffer

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


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_ASCEND FLIP")
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        assert not flip_dims_b[
            dim
        ], "dim {} appears multiple times in the list of dims".format(dim)
        flip_dims_b[dim] = True
    n = 0
    offset = 0
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()
    out = torch.empty_like(A)
    # a flipped view of A
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)

    # TODO: flip op can have a custom task simplification method, but we skip it now and just use A's rank.
    overload = copy_func.instantiate(A.ndim)
    overload(flipped_A, out0=out)
    return out
