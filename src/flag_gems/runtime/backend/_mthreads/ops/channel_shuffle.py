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

from flag_gems.ops.channel_shuffle import channel_shuffle as default_channel_shuffle
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def channel_shuffle_kernel(
    x_ptr,
    out_ptr,
    total,
    HW: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Decompose flat output index (contiguous NCHW layout):
    #   offs = n*C*HW + c*HW + sp, where sp in [0, HW)
    c = (offs // HW) % C
    # channel_shuffle: output channel c reads input channel
    #   c_in = (c % G) * (C // G) + (c // G)
    c_in = (c % G) * (C // G) + (c // G)
    read_idx = offs + (c_in - c) * HW
    val = tl.load(x_ptr + read_idx, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


def _use_triton_kernel(x: torch.Tensor, groups) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.ndim != 4 or not x.is_contiguous() or x.numel() == 0:
        return False
    try:
        g = int(groups)
    except Exception:
        return False
    c = x.shape[1]
    if g <= 0 or c % g != 0:
        return False
    return True


def channel_shuffle(x: torch.Tensor, groups: int):
    logger.debug("GEMS_MTHREADS CHANNEL_SHUFFLE")
    if not _use_triton_kernel(x, groups):
        return default_channel_shuffle(x, groups)

    N, C, H, W = x.shape
    HW = H * W
    numel = x.numel()
    G = int(groups)
    BLOCK = min(1024, triton.next_power_of_2(numel))
    grid = (triton.cdiv(numel, BLOCK),)
    with torch_device_fn.device(x.device):
        out = torch.empty_like(x)
        channel_shuffle_kernel[grid](x, out, numel, HW=HW, C=C, G=G, BLOCK=BLOCK)
    return out
