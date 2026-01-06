import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def copy_func(x):
    return x


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS FLIP")
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
        B = torch.empty_like(A)
        overload = copy_func.instantiate(A.ndim)
        overload(A, out0=B)
        # B = A
        return B
    out = torch.empty_like(A)
    # a flipped view of A
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)

    # TODO: flip op can have a custom task simplification method, but we skip it now and just use A's rank.
    overload = copy_func.instantiate(A.ndim)
    overload(flipped_A, out0=out)
    return out
