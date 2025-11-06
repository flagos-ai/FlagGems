import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


config_ = CodeGenConfig(
    1024,
    (40, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ASCEND CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    out0_strides = out0.stride()
    out0_offsets = list(
        itertools.accumulate(
            [s[dim] * out0_strides[dim] for s in inp_shapes[:-1]], initial=0
        )
    )

    for a, out0_offset in zip(A, out0_offsets):
        in_view = StridedBuffer(a, a.shape, a.stride())
        out_view = StridedBuffer(out0, a.shape, out0.stride(), offset=out0_offset)
        copy_func.instantiate(a.ndim)(in_view, out0=out_view)
    return out0
