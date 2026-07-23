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
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


config_ = CodeGenConfig(
    1024,
    (16, 1, 1),
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
    logger.debug("GEMS_TSINGMICRO CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # Find max ndim among all tensors
    max_ndim = max(_.ndim for _ in A)

    assert dim >= -max_ndim and dim < max_ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive (relative to max_ndim)
    dim = dim % max_ndim

    # Handle mixed-rank tensors: PyTorch allows cat when a lower-rank
    # tensor is empty (has 0 in some dimension). It implicitly unsqueezes
    # the empty tensor to match max_ndim, placing its 0-size at the dim position.
    if any(_.ndim != max_ndim for _ in A):
        # Find a reference tensor with max_ndim for non-dim sizes
        ref = next(_ for _ in A if _.ndim == max_ndim)
        ref_shape = ref.shape

        for i, a in enumerate(A):
            if a.ndim < max_ndim:
                new_shape = list(ref_shape)
                new_shape[dim] = a.shape[dim % a.ndim]
                A[i] = a.reshape(new_shape)

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
