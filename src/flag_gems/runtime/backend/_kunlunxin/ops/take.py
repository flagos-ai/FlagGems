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

from flag_gems.ops.take import take_kernel
from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy
from .copy import copy_ as gems_copy_

logger = logging.getLogger("flag_gems.ops.take")


def _launch_take(input, index, out_flat):
    input_flat = input.contiguous().view(-1)
    in_numel = input_flat.numel()
    n_index = index.numel()
    if n_index == 0:
        return
    index_flat = index.contiguous().view(-1)
    grid = (triton.cdiv(n_index, 1024),)
    with torch_device_fn.device(input.device):
        take_kernel[grid](
            input_flat, index_flat, out_flat, n_index, in_numel, BLOCK_SIZE=1024
        )


def take(input, index):
    logger.debug("GEMS_KUNLUNXIN TAKE")
    if input.device != index.device:
        raise RuntimeError("input and index must be on the same device")
    out_flat = torch.empty(index.numel(), device=input.device, dtype=input.dtype)
    _launch_take(input, index, out_flat)
    return out_flat.view(index.shape)


def take_out(input, index, *, out):
    logger.debug("GEMS_KUNLUNXIN TAKE_OUT")
    if not (input.device == index.device == out.device):
        raise RuntimeError("input, index and out must be on the same device")
    if out.dtype != input.dtype:
        raise RuntimeError(f"out must have dtype {input.dtype}, but got {out.dtype}")
    if tuple(out.shape) != tuple(index.shape):
        out.resize_(index.shape)
    if out.is_contiguous():
        out_flat = out.view(-1)
        _launch_take(input, index, out_flat)
    else:
        tmp_flat = torch.empty(index.numel(), device=input.device, dtype=input.dtype)
        _launch_take(input, index, tmp_flat)
        src = tmp_flat.view(index.shape)
        if not tle_copy(src, out):
            gems_copy_(out, src)
    return out
