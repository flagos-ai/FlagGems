# Copyright 2026- Xcoresigma Technology Co., Ltd
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
from typing import Any, Optional

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.math as math

logger = logging.getLogger(__name__)

_CACHED_CORE_NUM = None


def _get_core_num():
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        try:
            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            _CACHED_CORE_NUM = cores_dict["vector_core_num"] or 24
        except (AttributeError, KeyError, TypeError):
            _CACHED_CORE_NUM = 24
    return _CACHED_CORE_NUM


@triton.jit
def swiglu_kernel(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    input_stride_m,
    output_stride_m,
    BLOCK_SIZE_M: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_SIZE_M
    for tile_m_idx in range(0, BLOCK_SIZE_M, TILE_SIZE_M):
        m_idx = m_start + tile_m_idx
        if m_idx < M:
            for tile_h_idx in range(0, H, TILE_SIZE_H):
                offs_m = m_idx + tl.arange(0, TILE_SIZE_M)
                offs_h = tile_h_idx + tl.arange(0, TILE_SIZE_H)
                mask_m = offs_m < M
                mask_h = offs_h < H
                mask = mask_m[:, None] & mask_h[None, :]
                input_offset = offs_m[:, None] * input_stride_m + offs_h[None, :]
                x_a = tl.load(input_a_ptr + input_offset, mask=mask, other=0.0)
                x_b = tl.load(input_b_ptr + input_offset, mask=mask, other=0.0)
                x_a_f = x_a.to(tl.float32)
                sig = 1.0 / (1.0 + math.exp(-x_a_f))
                t = (x_a_f * sig).to(x_a.dtype)
                out = t * x_b

                output_offset = offs_m[:, None] * output_stride_m + offs_h[None, :]
                if (m_idx + TILE_SIZE_M <= M) and (tile_h_idx + TILE_SIZE_H <= H):
                    # dsa.copy writes the whole tile without masking, so it is
                    # only safe when the tile lies fully inside the output.
                    out_buf = tle.dsa.to_buffer(out, space=tle.dsa.ascend.UB)
                    with tle.dsa.hint(inter_no_alias=True):
                        tle.dsa.copy(
                            out_buf,
                            output_ptr + output_offset,
                            [TILE_SIZE_M, TILE_SIZE_H],
                        )
                else:
                    tl.store(output_ptr + output_offset, out, mask=mask)


def swiglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS_ASCEND SWIGLU")
    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of must be even number, got {input_tensor.shape[-1]}"
        )

    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)

    input_a, input_b = torch.split(input_2d, H, dim=1)
    output_2d = torch.empty(M, H, device=input_a.device, dtype=input_a.dtype)

    num_cores = _get_core_num()

    TILE_SIZE_M = min(triton.next_power_of_2(M), 32)
    TILE_SIZE_H = min(triton.next_power_of_2(H), 256)
    if M * H < 256 * 64:
        num_cores = 1
    # Round BLOCK_SIZE_M up to a multiple of TILE_SIZE_M so that the per-core
    # row ranges never overlap; tiles past M are masked inside the kernel.
    num_tiles_m = triton.cdiv(M, TILE_SIZE_M)
    num_cores = min(num_cores, num_tiles_m)
    BLOCK_SIZE_M = triton.cdiv(num_tiles_m, num_cores) * TILE_SIZE_M
    swiglu_kernel[(num_cores,)](
        input_a,
        input_b,
        output_2d,
        M,
        H,
        input_a.stride(0),
        output_2d.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        TILE_SIZE_M=TILE_SIZE_M,
        TILE_SIZE_H=TILE_SIZE_H,
        multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    return output_2d.view(*shape[:-1], H)
