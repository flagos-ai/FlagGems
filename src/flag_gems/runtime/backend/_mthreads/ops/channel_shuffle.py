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

logger = logging.getLogger(__name__)


@triton.jit
def channel_shuffle_kernel(
    in_ptr,
    out_ptr,
    C,
    HW,
    groups,
    c_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: axis0 = N * C (channel blocks), axis1 = HW tiles
    pid = tl.program_id(0)
    hw_pid = tl.program_id(1)

    # Decode (n, c) once per program instead of per element
    n = pid // C
    c = pid % C

    # Channel shuffle: new_c = (c % c_per_group) * groups + c // c_per_group
    group_id = c // c_per_group
    channel_in_group = c - group_id * c_per_group
    new_c = channel_in_group * groups + group_id

    # Source / destination are contiguous HW blocks; only the block order changes
    in_base = (n * C + c) * HW
    out_base = (n * C + new_c) * HW

    offsets = hw_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    x = tl.load(in_ptr + in_base + offsets, mask=mask)
    tl.store(out_ptr + out_base + offsets, x, mask=mask)


def channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS CHANNEL_SHUFFLE")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    # Channel shuffle expects (*, C, H, W) where C is divisible by groups
    if x.ndim < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions (C, H, W), got {x.ndim}"
        )

    C, H, W = x.shape[-3:]
    N = x.numel() // (C * H * W)
    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert C % g == 0, f"C ({C}) must be divisible by groups ({g})"

    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out

    HW = H * W
    BLOCK_SIZE = triton.next_power_of_2(min(HW, 1024))
    grid = (N * C, triton.cdiv(HW, BLOCK_SIZE))
    with torch_device_fn.device(x.device):
        channel_shuffle_kernel[grid](
            x,
            out,
            C,
            HW,
            g,
            C // g,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out
