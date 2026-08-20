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

import logging
import math
import os

import torch
import triton

from flag_gems.ops.repeat_interleave import (
    repeat_interleave_tensor as _repeat_interleave_tensor,
)
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.shape_utils import c_contiguous_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

_base_codegen_config = get_codegen_config()
_copy_codegen_config = CodeGenConfig(
    max_tile_size=_base_codegen_config.max_tile_size,
    max_grid_size=_base_codegen_config.max_grid_size,
    max_num_warps_per_cta=_base_codegen_config.max_num_warps_per_cta,
    prefer_block_pointer=False,
    prefer_1d_tile=True,
)


@pointwise_dynamic(
    num_inputs=1, promotion_methods=[(0, "DEFAULT")], config=_copy_codegen_config
)
@triton.jit
def copy_func(x):
    return x


def _compute_output_numel(inp_shape, repeats, dim):
    out_shape = list(inp_shape)
    if dim is None:
        return math.prod(out_shape) * repeats
    out_shape[dim] *= repeats
    return math.prod(out_shape)


def _repeat_interleave_self_int_impl(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS TSINGMICRO REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    # Keep the Triton path, but use the 1D-tile codegen to avoid block pointer
    # handling on the inserted zero-stride repeat dimension.
    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(
        output, out_view_shape, c_contiguous_stride(out_view_shape)
    )
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    original_precision_priority = os.environ.get("PRECISION_MODE", None)
    out_numel = _compute_output_numel(inp.shape, repeats, dim)
    if out_numel > 2**24:
        os.environ["PRECISION_MODE"] = "1"

    try:
        return _repeat_interleave_self_int_impl(
            inp, repeats, dim=dim, output_size=output_size
        )
    finally:
        if original_precision_priority is not None:
            os.environ["PRECISION_MODE"] = original_precision_priority
        else:
            os.environ.pop("PRECISION_MODE", None)


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS TSINGMICRO REPEAT_INTERLEAVE_SELF_TENSOR")

    if repeats.numel() == 0:
        return inp.clone()

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    if repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim, but got "
            "repeats.size(0) = {} and input.size({}) = {}".format(
                repeats.size(0), dim, inp_shape[dim]
            )
        )

    indices = _repeat_interleave_tensor(repeats, output_size=output_size)
    return torch.index_select(inp, dim, indices)


repeat_interleave_tensor = _repeat_interleave_tensor
