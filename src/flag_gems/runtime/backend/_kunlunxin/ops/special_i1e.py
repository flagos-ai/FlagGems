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

import flag_gems
from flag_gems.ops.special_i1e import _special_i1e_kernel

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024


def _run_special_i1e_kernel(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    x_contiguous = x.contiguous()
    out_contiguous = out.contiguous()
    n_elements = out_contiguous.numel()
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _special_i1e_kernel[grid](
        x_contiguous,
        out_contiguous,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
    )

    if out_contiguous.data_ptr() != out.data_ptr():
        out.copy_(out_contiguous)
    return out


def special_i1e(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1E")
    return _run_special_i1e_kernel(x, torch.empty_like(x))


def special_i1e_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I1E_OUT")
    return _run_special_i1e_kernel(self, out)


__all__ = ["special_i1e", "special_i1e_out"]
