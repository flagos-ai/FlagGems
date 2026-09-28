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


@libentry()
@triton.jit
def _adaptive_pool_windows(X, Y, SPEC: tl.constexpr):
    TOTAL: tl.constexpr = SPEC[0]
    IH: tl.constexpr = SPEC[1]
    IW: tl.constexpr = SPEC[2]
    OH: tl.constexpr = SPEC[3]
    OW: tl.constexpr = SPEC[4]
    KH: tl.constexpr = SPEC[5]
    KW: tl.constexpr = SPEC[6]
    BO: tl.constexpr = SPEC[7]
    BR: tl.constexpr = SPEC[8]
    out = tl.program_id(0) * BO + tl.arange(0, BO)
    nc = out // (OH * OW)
    oh = out // OW % OH
    ow = out % OW
    if IH % OH == 0 and IW % OW == 0:
        hs = oh * KH
        he = hs + KH
        ws = ow * KW
        we = ws + KW
    else:
        hs = oh * IH // OH
        he = ((oh + 1) * IH + OH - 1) // OH
        ws = ow * IW // OW
        we = ((ow + 1) * IW + OW - 1) // OW
    acc = tl.zeros((BO, BR), tl.float32)
    for start in range(tl.cdiv(KH * KW, BR)):
        r = start * BR + tl.arange(0, BR)
        h = hs[:, None] + r[None, :] // KW
        w = ws[:, None] + r[None, :] % KW
        if IH % OH == 0 and IW % OW == 0:
            mask = (out[:, None] < TOTAL) & (r[None, :] < KH * KW)
        else:
            mask = (
                (out[:, None] < TOTAL)
                & (h < he[:, None])
                & (w < we[:, None])
                & (r[None, :] < KH * KW)
            )
        v = tl.load(X + nc[:, None] * IH * IW + h * IW + w, mask, other=0).to(
            tl.float32
        )
        acc += v
    if IH % OH == 0 and IW % OW == 0:
        value = tl.sum(acc, 1) / (KH * KW)
    else:
        value = tl.sum(acc, 1) / ((he - hs) * (we - ws)).to(tl.float32)
    tl.store(Y + out, value, out < TOTAL)


def adaptive_avg_pool2d(input: torch.Tensor, output_size):
    logger.debug("GEMS ADAPTIVE_AVG_POOL2D")
    input = input.contiguous()
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    oh, ow = output_size
    n, c, ih, iw = input.shape
    output = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    if output.numel() == 0 or ih == 0 or iw == 0:
        return output
    kh = triton.cdiv(ih, oh) + (ih % oh != 0)
    kw = triton.cdiv(iw, ow) + (iw % ow != 0)
    br = min(triton.next_power_of_2(kh * kw), 1024)
    bo = min(64, max(1, 4096 // br))
    with torch_device_fn.device(input.device):
        _adaptive_pool_windows[(triton.cdiv(output.numel(), bo),)](
            input,
            output,
            (output.numel(), ih, iw, oh, ow, kh, kw, bo, br),
            num_warps=4,
        )
    return output
