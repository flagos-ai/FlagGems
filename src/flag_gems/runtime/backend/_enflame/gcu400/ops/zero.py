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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

BLOCK_SIZE = 16384
NUM_WARPS = 1
GRID_SIZE = 24


@libentry()
@triton.jit
def zero_kernel(
    out_ptr,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out_ptr + offsets, 0.0, mask=mask)


def _launch_zero_kernel(tensor: torch.Tensor) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor), "Expected a torch.Tensor"
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    n_elements = tensor.numel()
    if n_elements == 0:
        return tensor
    grid_fn = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), GRID_SIZE),
    )
    with torch_device_fn.device(tensor.device):
        zero_kernel[grid_fn](
            tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
        )
    return tensor


def zero(*args, **kwargs):
    logger.debug("GEMS_ENFLAME ZERO")
    target = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        target = args[0]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        target = kwargs["self"]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        target = kwargs["input"]
    elif "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        target = kwargs["out"]
    else:
        raise ValueError(
            "zero expects a Tensor as the first argument or in kwargs as 'self', 'input', or 'out'"
        )
    return _launch_zero_kernel(target)


def zero_out(*args, **kwargs):
    logger.debug("GEMS_ENFLAME ZERO_OUT")
    out = None
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        out = kwargs["out"]
    elif len(args) >= 1 and isinstance(args[0], torch.Tensor):
        out = args[0]
    else:
        raise ValueError(
            "zero_out expects an output Tensor as the first positional argument or 'out' kwarg"
        )
    return _launch_zero_kernel(out)


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ZERO_")
    return _launch_zero_kernel(x)
