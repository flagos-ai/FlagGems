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
from flag_gems.utils import libentry

device = device.name
logger = logging.getLogger(__name__)


# Flat 1D configs over the pooled spatial plane. Each program owns one (n, c)
# slice and streams BLOCK pooled elements, scattering them to their original
# location in the output plane. This maps cleanly onto hygon 64-lane
# wavefronts and avoids the 2D tiling / stride-decode overhead of the generic
# kernel: since max_pool2d indices are already flat offsets into the per
# channel out_h * out_w plane and the output is contiguous, the flat index is
# the output offset directly.
def _max_unpool2d_configs():
    configs = []
    for block in [256, 512, 1024, 2048]:
        for num_warps in [4, 8]:
            configs.append(
                triton.Config({"BLOCK": block}, num_warps=num_warps, num_stages=1)
            )
    return configs


@libentry()
@triton.autotune(
    configs=_max_unpool2d_configs(),
    key=["pooled_hw", "out_hw"],
)
@triton.jit
def max_unpool2d_kernel(
    pooled_ptr,
    indices_ptr,
    output_ptr,
    pooled_hw,
    out_hw,
    BLOCK: tl.constexpr,
    USE_INT64_IDX: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < pooled_hw

    if USE_INT64_IDX:
        base_in = pid_nc.to(tl.int64) * pooled_hw + offs
        base_out = pid_nc.to(tl.int64) * out_hw
    else:
        base_in = pid_nc * pooled_hw + offs
        base_out = pid_nc * out_hw

    vals = tl.load(pooled_ptr + base_in, mask=mask, other=0.0)
    idx = tl.load(indices_ptr + base_in, mask=mask, other=0)

    # Bounds check: flat indices decoded from max_pool2d may land outside the
    # output plane when input dims are not evenly divisible by stride
    # (e.g. ceil_mode=False MaxPool on odd-sized dims). In that case skip.
    in_bounds = (idx >= 0) & (idx < out_hw)
    store_mask = mask & in_bounds

    tl.store(output_ptr + base_out + idx, vals, mask=store_mask)


def max_unpool2d(pooled: torch.Tensor, indices: torch.Tensor, output_size: list):
    logger.debug("GEMS_HYGON MAX_UNPOOL2D")

    pooled = pooled.contiguous()
    indices = indices.contiguous()

    n, c, pooled_h, pooled_w = pooled.shape
    out_h, out_w = output_size[0], output_size[1]

    output = torch.zeros((n, c, out_h, out_w), device=pooled.device, dtype=pooled.dtype)

    if output.numel() == 0:
        return output

    pooled_hw = pooled_h * pooled_w
    out_hw = out_h * out_w

    # Guard against int32 overflow of the linear element offset.
    use_int64 = (n * c * max(pooled_hw, out_hw)) >= 2**31

    grid = lambda meta: (
        n * c,
        triton.cdiv(pooled_hw, meta["BLOCK"]),
    )

    with torch_device_fn.device(pooled.device):
        max_unpool2d_kernel[grid](
            pooled,
            indices,
            output,
            pooled_hw,
            out_hw,
            USE_INT64_IDX=use_int64,
        )

    return output
