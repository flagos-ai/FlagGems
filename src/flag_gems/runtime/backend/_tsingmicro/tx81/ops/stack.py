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


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_TSINGMICRO STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    in0_shape = inp0_shape[:dim] + [1] + inp0_shape[dim:]
    out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
    out0 = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    # each input occupies a single element along `dim`, so input i maps to
    # element offset i (in elements, not bytes)
    out0_offsets = list(itertools.accumulate([1 for _ in inp_shapes[:-1]], initial=0))

    # Per-dim offsets cannot be expressed with block pointers (boundary_check
    # clips tiles whose offsets exceed the logical shape), so always use the
    # masked-load kernel variant.
    config = copy.copy(copy_func.config)
    config.prefer_block_pointer = False
    copy_func.config = config

    needs_i64 = _needs_i64_precision(tensors, out0)
    precision_token = set_precision_mode("2") if needs_i64 else None
    try:
        for a, out0_offset in zip(tensors, out0_offsets):
            a = a.reshape(in0_shape)
            per_dim_offsets = [0] * len(in0_shape)
            # negative: store computes addr = base + (logical - offset) * stride,
            # so a negative offset shifts the write region forward along `dim`
            per_dim_offsets[dim] = -out0_offset
            in_view = StridedBuffer(a, in0_shape, a.stride())
            out_view = StridedBuffer(
                out0, in0_shape, out0.stride(), offsets=per_dim_offsets
            )
            copy_func.instantiate(a.ndim)(in_view, out0=out_view)
    finally:
        if precision_token is not None:
            restore_precision_mode(precision_token)

    return out0
