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
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ZERO_")
    if not x.is_contiguous():
        # fill imports sum, which imports zero_; defer this dependency until
        # package initialization has finished.
        # `zeros_kernel` walks flat offsets 0..numel-1 from the view's data
        # pointer and ignores strides, so a non-contiguous view would be zeroed
        # outside its own memory and partly left untouched. Fall back to the
        # scalar fill, which writes through the tensor's strides (the same path
        # ATen's `zero_` uses) and needs no temporary buffer. See Issue #7115.
        # The generic `fill_scalar_` is called directly rather than through
        # `x.fill_(0)`: some vendor overrides of `fill_` still store flat
        # offsets and would reintroduce the same defect on a strided view.
        #
        # Imported here rather than at module scope to keep
        # `zeros -> fill -> sum -> zeros` from closing at import time:
        # `fill` pulls in `sum` for its backward, and `sum.sum_out` calls
        # `zero_`.  At module scope this makes `import flag_gems` fail.
        from flag_gems.ops.fill import fill_scalar_ as _fill_scalar_

        _fill_scalar_(x, 0)
        return x
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](x, N, BLOCK_SIZE=1024)
    return x
