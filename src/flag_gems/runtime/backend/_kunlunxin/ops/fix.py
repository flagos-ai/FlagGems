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

from flag_gems.ops.fix import _copy_kernel, _fix_trunc_kernel

from ..utils.tle_copy import tle_copy
from .copy import copy_ as gems_copy_
from .trunc import trunc as gems_trunc

logger = logging.getLogger("flag_gems.ops.fix")


def _fix_complex(self: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(gems_trunc(torch.view_as_real(self)).contiguous())


def _launch_fix_kernel(x: torch.Tensor, out: torch.Tensor, block_size: int = 1024):
    assert x.is_cuda and out.is_cuda, "Input and output must be on CUDA device"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.device == out.device, "Input and output must be on the same device"
    assert (
        x.is_contiguous() and out.is_contiguous()
    ), "Only contiguous tensors are supported"

    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, block_size),)

    if x.is_floating_point():
        _fix_trunc_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    else:
        _copy_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)


def fix(self: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN FIX")
    if self.is_complex():
        return _fix_complex(self)

    out = torch.empty_like(self)
    _launch_fix_kernel(self, out)
    return out


def fix_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN FIX_OUT")
    if self.is_complex():
        truncated = _fix_complex(self)
        out_real = torch.view_as_real(out)
        src_real = torch.view_as_real(truncated)
        if not tle_copy(src_real, out_real):
            gems_copy_(out_real, src_real)
        return out

    _launch_fix_kernel(self, out)
    return out
