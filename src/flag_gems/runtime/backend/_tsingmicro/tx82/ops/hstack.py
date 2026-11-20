# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

# f32 can only represent integers up to 2**24 exactly; under precision-mode 0/1
# i32 strides are demoted to f32 arithmetic, so larger address intermediates
# lose precision and elements land at wrong offsets. Force pm=2 for exact
# integer arithmetic (same rationale as flip).
_F32_PRECISION_THRESHOLD = 2**24


def _needs_i64_precision(inputs, out0):
    in_max = 0
    for a in inputs:
        if any(abs(a.stride(i)) > _F32_PRECISION_THRESHOLD for i in range(a.ndim)):
            return True
        in_max = max(
            in_max,
            sum((a.shape[i] - 1) * abs(a.stride(i)) for i in range(a.ndim)),
        )
    # output views are freshly allocated, so the store's max physical offset
    # (logical - offset) * stride is bounded by out0.numel()
    return max(in_max, out0.numel()) > _F32_PRECISION_THRESHOLD


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def hstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> torch.Tensor:
    logger.debug("GEMS_TSINGMICRO HSTACK")

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
    out0_dim_offsets = list(
        itertools.accumulate([s[dim] for s in inp_shapes[:-1]], initial=0)
    )

    # Per-dim offsets cannot be expressed with block pointers (boundary_check
    # clips tiles whose offsets exceed the logical shape), so always use the
    # masked-load kernel variant.
    config = copy.copy(copy_func.config)
    config.prefer_block_pointer = False
    copy_func.config = config

    needs_i64 = _needs_i64_precision(tensors, out0)
    precision_token = set_precision_mode("2") if needs_i64 else None
    try:
        for a, out0_dim_offset in zip(tensors, out0_dim_offsets):
            per_dim_offsets = [0] * len(a.shape)
            # negative: store computes addr = base + (logical - offset) * stride,
            # so a negative offset shifts the write region forward along `dim`
            per_dim_offsets[dim] = -out0_dim_offset
            in_view = StridedBuffer(a, a.shape, a.stride())
            out_view = StridedBuffer(
                out0, a.shape, out0.stride(), offsets=per_dim_offsets
            )
            copy_func.instantiate(a.ndim)(in_view, out0=out_view)
    finally:
        if precision_token is not None:
            restore_precision_mode(precision_token)

    return out0
