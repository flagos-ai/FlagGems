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

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.utils import CORE_NUM

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

_BLOCK_SIZE = 1024


@triton.jit
def _log_kernel_(x_ptr, n_elements, n_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-stride loop: the launch is capped at the AI vector core count, so one
    # program may have to walk more than one block.
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)
        y_f32 = tl.log(x_f32)
        y = y_f32.to(x.dtype)
        tl.store(x_ptr + offsets, y, mask=mask)


def _launch_log_(x, n_elements):
    # triton-ascend does not queue programs beyond the number of AI vector cores
    # the device has: a larger grid is padded up to the next multiple of
    # CORE_NUM, and every padding CTA reports program_id == 0. Because log_ is
    # in-place, such a repeated program 0 applied log a second time to the first
    # block, leaving log(log(x)) -- NaN for x < 1 -- in elements [0..1023]. That
    # is the 1024 NaNs reported in #6446, which surfaced through
    # Tensor.exponential_() because its torch_npu composite implementation calls
    # uniform_() followed by log_(). Capping the grid at CORE_NUM keeps the
    # launch to a single wave with no repeated CTA; the kernel above covers the
    # remaining blocks with its grid-stride loop.
    n_blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    grid = (min(n_blocks, CORE_NUM),)
    with torch_device_fn.device(x.device):
        _log_kernel_[grid](x, n_elements, n_blocks, BLOCK_SIZE=_BLOCK_SIZE)


def log_(*args, **kwargs):
    logger.debug("GEMS_ASCEND LOG_")
    x = args[0] if len(args) > 0 else kwargs.get("input", None)

    if x is None:
        raise ValueError(
            "log_ expects a tensor as the first argument or keyword 'input'."
        )
    if not isinstance(x, torch.Tensor):
        raise TypeError("log_ expects a torch.Tensor as input.")

    if x.dtype not in _SUPPORTED_DTYPES:
        # The Triton kernel only handles real floating-point dtypes. For anything
        # else (e.g. integer tensors), an in-place log cannot store the float
        # result, so raise instead of silently truncating -- matching torch, which
        # errors on int32.log_() and similar. We raise directly rather than
        # delegating to torch.ops.aten.log_, which would recurse back into this
        # patched op while gems is active.
        raise TypeError(f"log_ does not support dtype {x.dtype}")

    if not x.is_contiguous():
        # Operate on a contiguous copy and copy the result back.
        y = x.contiguous()
        n_elements = y.numel()
        if n_elements == 0:
            return x
        _launch_log_(y, n_elements)
        x.copy_(y)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    _launch_log_(x, n_elements)
    return x
