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

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.math as math

logger = logging.getLogger(__name__)


@triton.jit
def swiglu_kernel(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    input_stride_m,
    output_stride_m,
    beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_SIZE_M
    for tile_m_idx in range(0, BLOCK_SIZE_M, TILE_SIZE_M):
        m_idx = m_start + tile_m_idx
        for tile_h_idx in range(0, H, TILE_SIZE_H):
            offs_m = m_idx + tl.arange(0, TILE_SIZE_M)
            offs_h = tile_h_idx + tl.arange(0, TILE_SIZE_H)
            mask_m = offs_m < M
            mask_h = offs_h < H
            mask = mask_m[:, None] & mask_h[None, :]
            input_offset = offs_m[:, None] * input_stride_m + offs_h[None, :]
            x_a = tl.load(input_a_ptr + input_offset, mask=mask, other=0.0)
            x_b = tl.load(input_b_ptr + input_offset, mask=mask, other=0.0)

            tmp = x_a * x_b
            sig = 1.0 / (1.0 + math.exp(-1 * x_a * beta))
            out = tmp * sig.to(tl.float16)

            output_offset = offs_m[:, None] * output_stride_m + offs_h[None, :]
            out_buf = tle.dsa.to_buffer(out, space=tle.dsa.ascend.UB)
            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(
                    out_buf, output_ptr + output_offset, [TILE_SIZE_M, TILE_SIZE_H]
                )


def swiglu(input_tensor: torch.Tensor, scalarValue: float) -> torch.Tensor:
    logger.debug("GEMS SWIGLU")
    assert (
        input_tensor.shape[-1] % 2 == 0
    ), "The last dimension of the input tensor must be even."

    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    if len(input_tensor.shape) > 2:
        input_2d = input_tensor.contiguous().view(-1, 2 * H)
    else:
        input_2d = input_tensor.contiguous()

    input_a, input_b = torch.split(input_2d, H, dim=1)
    output_2d = torch.empty(M, H, device=input_a.device, dtype=input_a.dtype)

    def get_core_num():
        try:
            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            return cores_dict["vector_core_num"]
        except (AttributeError, KeyError, TypeError):
            return None

    num_cores = 24 if get_core_num() is None else get_core_num()

    TILE_SIZE_M = min(M, 32)
    TILE_SIZE_H = min(H, 256)
    if M * H < 256 * 64:
        num_cores = 1
    BLOCK_SIZE_M = int((M + num_cores - 1) // num_cores)
    swiglu_kernel[(num_cores,)](
        input_a,
        input_b,
        output_2d,
        M,
        H,
        input_a.stride(0),
        output_2d.stride(0),
        beta=scalarValue,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        TILE_SIZE_M=TILE_SIZE_M,
        TILE_SIZE_H=TILE_SIZE_H,
        multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    return output_2d
