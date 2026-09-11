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

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["steps"])
def linspace_kernel(
    out_ptr,
    start,
    end_val,
    step_size,
    steps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < steps
    out_val = start + step_size * idx
    last_mask = idx == (steps - 1)
    out_val = tl.where(last_mask, end_val, out_val)
    tl.store(out_ptr + idx, out_val, mask=mask)


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME LINSPACE")
    assert steps >= 1, "steps must be >= 1"

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if steps == 1:
        return torch.fill(out, start)

    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()
    step_size = (float(end) - float(start)) / (steps - 1)

    if steps <= 65536:
        BLOCK = 8192
        nw = 4
    else:
        BLOCK = 65536
        nw = 1
    while triton.cdiv(steps, BLOCK) > 65535:
        BLOCK *= 2

    grid = (triton.cdiv(steps, BLOCK),)
    linspace_kernel[grid](
        out,
        start,
        float(end),
        step_size,
        steps,
        BLOCK=BLOCK,
        num_warps=nw,
    )
    return out
