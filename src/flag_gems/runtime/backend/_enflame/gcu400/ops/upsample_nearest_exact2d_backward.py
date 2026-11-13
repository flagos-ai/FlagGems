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

"""GCU400 专属 upsample_nearest_exact2d_backward 实现。

公共实现把 (n, c) 直接映射为 grid.y，N*C 较大时（512 / 555 / 24576）
会超过 GCU400 的 255 硬件上限并抛 OutOfResources。这里把 (nc, 输出 tile)
展平成一维任务队列，用固定 program 数加核内跨步循环，解除 grid 维度限制。
"""

import logging
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)

GCU_NUM_PROGRAMS = 48
BLOCK_SIZE = 1024


@triton.jit(
    do_not_specialize=[
        "OH",
        "OW",
        "IH",
        "IW",
        "reciprocal_scale_h",
        "reciprocal_scale_w",
        "scale_h",
        "scale_w",
        "num_blocks",
        "total_input",
    ]
)
def _upsample_nearest_exact2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    scale_h,
    scale_w,
    num_blocks,
    total_input,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    EXACT_SCALE: tl.constexpr,
    INT_H: tl.constexpr,
    INT_W: tl.constexpr,
):
    pid = tl.program_id(0)

    # 反向等价形式：每个输入位置累加所有映射到它的输出梯度。
    # 用 gather-reduce 取代 atomic_add，既消除原子竞争，也省掉 fp32 中转与末尾 dtype 转换。
    for block_id in tl.range(pid, num_blocks, NUM_PROGRAMS):
        idx = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = idx < total_input
        iw = idx % IW
        ih = (idx // IW) % IH
        nc = idx // (IH * IW)

        # 整数 scale 时窗口起点和大小精确，无需候选外溢与反向映射校验。
        if EXACT_SCALE:
            oh_base = ih * INT_H
            ow_base = iw * INT_W
        else:
            # 非整数 scale 用 floor 起点 + 额外一行/一列候选，靠反向映射判定归属。
            oh_base = (ih.to(tl.float32) * scale_h).to(tl.int32)
            ow_base = (iw.to(tl.float32) * scale_w).to(tl.int32)
        accumulator = tl.zeros([BLOCK], dtype=tl.float32)

        for dh in tl.static_range(0, MAX_H):
            for dw in tl.static_range(0, MAX_W):
                oh = oh_base + dh
                ow = ow_base + dw
                if EXACT_SCALE:
                    valid = mask
                else:
                    mapped_ih = (oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32)
                    mapped_iw = (ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32)
                    valid = (
                        mask
                        & (oh < OH)
                        & (ow < OW)
                        & (mapped_ih == ih)
                        & (mapped_iw == iw)
                    )
                offset_o = (nc * OH + oh) * OW + ow
                value = tl.load(grad_output_ptr + offset_o, mask=valid, other=0.0).to(
                    tl.float32
                )
                accumulator += value

        tl.store(
            grad_input_ptr + idx,
            accumulator.to(grad_input_ptr.dtype.element_ty),
            mask=mask,
        )


def _upsample_nearest_exact2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME_GCU400 UPSAMPLE_NEAREST_EXACT2D_BACKWARD")

    assert grad_output.device.type == device.name
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    OH, OW = output_size
    N, C, IH, IW = input_size

    assert grad_output.shape == (N, C, OH, OW), (
        f"grad_output shape {grad_output.shape} does not match "
        f"expected shape (N={N}, C={C}, OH={OH}, OW={OW})"
    )

    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW

    # gather-reduce 不需要原子累加，输出直接按目标 dtype 分配，省掉 fp32 中转与末尾转换。
    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )

    scale_h = 1.0 / reciprocal_scale_h
    scale_w = 1.0 / reciprocal_scale_w
    # 整数倍上采样可精确切出 scale_h x scale_w 的输出窗口，避免多余候选 load。
    exact_scale = (
        abs(scale_h - round(scale_h)) < 1e-6 and abs(scale_w - round(scale_w)) < 1e-6
    )
    if exact_scale:
        max_h = int(round(scale_h))
        max_w = int(round(scale_w))
    else:
        max_h = int(math.ceil(scale_h)) + 1
        max_w = int(math.ceil(scale_w)) + 1
    total_input = N * C * IH * IW
    num_blocks = triton.cdiv(total_input, BLOCK_SIZE)
    num_programs = min(GCU_NUM_PROGRAMS, num_blocks)

    with torch_device_fn.device(grad_output.device):
        _upsample_nearest_exact2d_backward_kernel[(num_programs,)](
            grad_output,
            grad_input,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            scale_h,
            scale_w,
            num_blocks,
            total_input,
            NUM_PROGRAMS=num_programs,
            BLOCK=BLOCK_SIZE,
            MAX_H=max_h,
            MAX_W=max_w,
            EXACT_SCALE=exact_scale,
            INT_H=max_h if exact_scale else 1,
            INT_W=max_w if exact_scale else 1,
            num_warps=4,
        )

    return grad_input
