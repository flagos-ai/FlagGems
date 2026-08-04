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

import torch
import triton

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.shape_utils import has_internal_overlapping

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    1536,
    (40, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy(src):
    return src


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logger.debug("GEMS_ASCEND SLICE_SCATTER")
    assert src.device == inp.device, "inp and src reside on different devices."
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim

    start = start or 0
    end = end or inp.size(dim)
    if end < 0:
        end = end % inp.size(dim)

    valid_shape = list(inp.shape)
    valid_shape[dim] = triton.cdiv(end - start, step)
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp):
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    ndim = inp.ndim
    copy(inp, out0=out)

    indices = [slice(None)] * ndim
    indices[dim] = slice(start, end, step)
    out_ = out[indices]
    copy(src, out0=out_)

    return out
