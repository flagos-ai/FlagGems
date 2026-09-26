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
from flag_gems.ops.i0 import i0_kernel

from .copy import copy_ as gems_copy_

logger = logging.getLogger("flag_gems.ops.i0")


def _launch_i0(out: torch.Tensor, x: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Input and output must be {flag_gems.device} tensors")
    assert (
        out.numel() == x.numel()
    ), "Input and output must have the same number of elements"
    assert out.device == x.device, "Input and output must be on the same device"

    x_in = x
    out_in = out

    if not x_in.is_floating_point():
        x_in = x_in.to(torch.get_default_dtype())

    if x_in.dtype != out_in.dtype:
        x_in = x_in.to(out_in.dtype)

    x_contig = x_in.contiguous()
    out_was_noncontig = not out_in.is_contiguous()
    out_contig = out_in.contiguous() if out_was_noncontig else out_in

    n_elements = out_contig.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    i0_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out_was_noncontig:
        gems_copy_(out_in, out_contig)
    return out_in


def i0(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN I0")
    if x.device.type != flag_gems.device:
        raise ValueError(f"i0: input tensor must be on {flag_gems.device} device")
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x, dtype=out_dtype)
    _launch_i0(out, x)
    return out


def i0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN I0_OUT")
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(
            f"i0_out: input and output tensors must be on {flag_gems.device} device"
        )
    if not out.is_floating_point():
        raise TypeError("i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "i0_out: input and output must have the same number of elements"
        )
    _launch_i0(out, x)
    return out
