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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _pool_divisible(
    X,
    Y,
    ROWS: tl.constexpr,
    IW: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BROWS: tl.constexpr,
    BKH: tl.constexpr,
    BOW: tl.constexpr,
    CHUNK: tl.constexpr,
):
    for block in range(
        tl.program_id(0) * CHUNK,
        tl.minimum((tl.program_id(0) + 1) * CHUNK, tl.cdiv(ROWS, BROWS)),
    ):
        row = block * BROWS + tl.arange(0, BROWS)
        kh = tl.arange(0, BKH)
        col = tl.arange(0, BOW * KW)
        x = tl.load(
            X + (row[:, None, None] * KH + kh[None, :, None]) * IW + col[None, None, :],
            (row[:, None, None] < ROWS)
            & (kh[None, :, None] < KH)
            & (col[None, None, :] < IW),
            other=0,
        ).to(tl.float32)
        # Reduce height before splitting contiguous columns into pooling windows.
        # Keep the reduction tensor rank at most three on Ascend.
        x = tl.sum(x, 1)
        if KW < 8:
            # Preserve contiguous loads; select window columns without a narrow reduction.
            v = tl.zeros((BROWS, BOW), tl.float32)
            for k in tl.static_range(KW):
                cols = tl.arange(0, BOW) * KW + k
                indices = tl.broadcast_to(cols[None, :], (BROWS, BOW))
                v += tl.gather(x, indices, axis=1)
            v = v / (KH * KW)
        else:
            x = tl.reshape(x, (BROWS, BOW, KW))
            v = tl.sum(x, 2) / (KH * KW)
        ow = tl.arange(0, BOW)
        tl.store(
            Y + row[:, None] * OW + ow[None, :],
            v,
            (row[:, None] < ROWS) & (ow[None, :] < OW),
        )


@libentry()
@triton.jit
def _pool_general(
    X,
    Y,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    TOTAL: tl.constexpr,
    CHUNK: tl.constexpr,
):
    for out in range(
        tl.program_id(0) * CHUNK, tl.minimum((tl.program_id(0) + 1) * CHUNK, TOTAL)
    ):
        nc = out // (OH * OW)
        oh = out // OW - nc * OH
        ow = out - (out // OW) * OW
        hs = oh * IH // OH
        he = ((oh + 1) * IH + OH - 1) // OH
        ws = ow * IW // OW
        we = ((ow + 1) * IW + OW - 1) // OW
        h = tl.arange(0, BH)
        w = tl.arange(0, BW)
        acc = tl.zeros((BH, BW), tl.float32)
        for hi in range(hs, he, BH):
            for wi in range(ws, we, BW):
                hr = hi + h
                wr = wi + w
                x = tl.load(
                    X + nc * IH * IW + hr[:, None] * IW + wr[None, :],
                    (hr[:, None] < he) & (wr[None, :] < we),
                    other=0,
                ).to(tl.float32)
                acc += x
        value = tl.sum(tl.sum(acc, 1), 0) / ((he - hs) * (we - ws))
        tl.store(Y + out, value)


def adaptive_avg_pool2d(input, output_size):
    logger.debug("GEMS_ASCEND ADAPTIVE_AVG_POOL2D")
    input = input.contiguous()
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    oh, ow = output_size
    n, c, ih, iw = input.shape
    output = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    if output.numel() == 0 or ih == 0 or iw == 0:
        return output
    with torch_device_fn.device(input.device):
        kh, kw = ih // oh, iw // ow
        if (
            ih % oh == 0
            and iw % ow == 0
            and kw & (kw - 1) == 0
            and kw > 0
            and kh * iw <= 8192
        ):
            bkh, bow = triton.next_power_of_2(kh), triton.next_power_of_2(ow)
            rows = min(8, max(1, 8192 // (bkh * bow * kw)))
            rows = triton.next_power_of_2(rows)
            blocks = triton.cdiv(n * c * oh, rows)
            grid = min(CORE_NUM, blocks)
            chunk = triton.cdiv(blocks, grid)
            _pool_divisible[(grid,)](
                input,
                output,
                n * c * oh,
                iw,
                ow,
                kh,
                kw,
                rows,
                bkh,
                bow,
                chunk,
            )
        else:
            bh = min(32, triton.next_power_of_2(triton.cdiv(ih, oh) + 1))
            bw = min(128, triton.next_power_of_2(triton.cdiv(iw, ow) + 1))
            grid = min(CORE_NUM, output.numel())
            chunk = triton.cdiv(output.numel(), grid)
            _pool_general[(grid,)](
                input, output, ih, iw, oh, ow, bh, bw, output.numel(), chunk
            )
    return output
