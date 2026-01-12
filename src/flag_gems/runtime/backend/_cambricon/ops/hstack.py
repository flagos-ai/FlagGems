import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton

from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def hstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON HSTACK")

    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    if tensors[0].ndim == 0:
        tensors[0] = tensors[0].view(1)
    inp0_shape = tensors[0].shape
    out_shape = list(inp0_shape)
    inp_shapes = [inp0_shape]

    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]

    for ty in dtypes[1:]:
        dtype = torch.promote_types(dtype, ty)

    for i, tensor in enumerate(tensors):
        if tensor.dtype != dtype:
            tensors[i] = tensor.to(dtype)

    if len(inp0_shape) == 1:
        dim = 0
    else:
        dim = 1

    for tensor_num, tensor in enumerate(tensors[1:]):
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.ndim != tensors[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {tensors[0].ndim} and {tensor.ndim}"
            )

        inp_shape = tensor.shape
        inp_shapes.append(inp_shape)

        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. \
                        Expected size {inp0_shape[i]} but got size {inp_shape[i]} \
                        for tensor number {tensor_num + 1} in the list."
                )

    out_shape[dim] = sum(s[dim] for s in inp_shapes)

    out0 = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    out0_strides = out0.stride()
    out0_offsets = list(
        itertools.accumulate(
            [s[dim] * out0_strides[dim] for s in inp_shapes[:-1]], initial=0
        )
    )

    for a, out0_offset in zip(tensors, out0_offsets):
        in_view = StridedBuffer(a, a.shape, a.stride())
        out_view = StridedBuffer(out0, a.shape, out0.stride(), offset=out0_offset)
        copy_func.instantiate(a.ndim)(in_view, out0=out_view)

    return out0
