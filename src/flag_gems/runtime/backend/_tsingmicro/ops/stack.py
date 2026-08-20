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

import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)


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
    out0_strides = out0.stride()
    out0_offsets = list(
        itertools.accumulate([out0_strides[dim] for _ in inp_shapes[:-1]], initial=0)
    )

    for a, out0_offset in zip(tensors, out0_offsets):
        a = a.reshape(in0_shape)
        in_view = StridedBuffer(a, in0_shape, a.stride())
        out_view = StridedBuffer(out0, in0_shape, out0.stride(), offset=out0_offset)
        copy_func.instantiate(a.ndim)(in_view, out0=out_view)

    return out0
