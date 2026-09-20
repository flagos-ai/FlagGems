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
def _swiglu_process_tile(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    m_idx,
    tile_h_idx,
    M: tl.constexpr,
    H: tl.constexpr,
    input_stride_m,
    output_stride_m,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
):
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
    if x_a.dtype == tl.bfloat16:
        # bf16 vector multiply is emulated on the vector core; staying in fp32
        # and rounding once is cheaper than rounding to bf16 in between.
        out = (x_a_f * sig * x_b.to(tl.float32)).to(x_a.dtype)
    else:
        t = (x_a_f * sig).to(x_a.dtype)
        out = t * x_b

    output_offset = offs_m[:, None] * output_stride_m + offs_h[None, :]
    # dsa.copy writes the whole tile without masking, so it is only
    # safe when the tile lies fully inside the output.
    if (
        TILE_SIZE_M * TILE_SIZE_H > 1
        and m_idx + TILE_SIZE_M <= M
        and tile_h_idx + TILE_SIZE_H <= H
    ):
        out_buf = tle.dsa.to_buffer(out, space=tle.dsa.ascend.UB)
        tle.dsa.copy(
            out_buf,
            output_ptr + output_offset,
            [TILE_SIZE_M, TILE_SIZE_H],
        )
    else:
        tl.store(output_ptr + output_offset, out, mask=mask)


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
    BLOCK_SIZE_H: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    SPLIT_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    m_start = pid_m * BLOCK_SIZE_M
    for tile_m_idx in range(0, BLOCK_SIZE_M, TILE_SIZE_M):
        m_idx = m_start + tile_m_idx
        if m_idx < M:
            # Keep the compile-time loop bounds when H is not split across
            # cores: runtime bounds block the loop pipeline optimization and
            # cost ~25% on large shapes.
            if SPLIT_H:
                h_end = tl.minimum((pid_h + 1) * BLOCK_SIZE_H, H)
                for tile_h_idx in range(pid_h * BLOCK_SIZE_H, h_end, TILE_SIZE_H):
                    _swiglu_process_tile(
                        input_a_ptr,
                        input_b_ptr,
                        output_ptr,
                        m_idx,
                        tile_h_idx,
                        M,
                        H,
                        input_stride_m,
                        output_stride_m,
                        TILE_SIZE_M,
                        TILE_SIZE_H,
                    )
            else:
                for tile_h_idx in range(0, H, TILE_SIZE_H):
                    _swiglu_process_tile(
                        input_a_ptr,
                        input_b_ptr,
                        output_ptr,
                        m_idx,
                        tile_h_idx,
                        M,
                        H,
                        input_stride_m,
                        output_stride_m,
                        TILE_SIZE_M,
                        TILE_SIZE_H,
                    )


def swiglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS_ASCEND SWIGLU")
    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of must be even number, got {input_tensor.shape[-1]}"
        )

    shape = input_tensor.shape
    H_out = shape[-1] // 2
    M = input_tensor.numel() // (2 * H_out)
    input_2d = input_tensor.contiguous().view(M, 2 * H_out)
    H = H_out

    num_cores = _get_core_num()

    # When M alone cannot feed every core (e.g. a 1D tensor has M == 1), the
    # kernel would have to split H across cores, whose runtime loop bounds
    # disable the loop pipeline. Re-tile (M, 2H) into (M*k, 2H/k) instead so
    # the M dimension alone saturates the cores; both views are free and the
    # output shape is restored at the end.
    if M * H >= 256 * 64 and M < num_cores:
        h_sub = min(H, 1024)
        while H % h_sub != 0:
            h_sub //= 2
        if h_sub < H:
            k = H // h_sub
            input_2d = input_2d.view(M * k, 2 * h_sub)
            M, H = M * k, h_sub

    input_a, input_b = torch.split(input_2d, H, dim=1)
    output_2d = torch.empty(M, H, device=input_a.device, dtype=input_a.dtype)

    # Tile shape follows the row width: long rows want longer contiguous
    # column segments (fewer row jumps per tile), short rows want more rows
    # per tile so each DMA moves more bytes. Keep the tile at ~8K elements
    # (larger tiles exceed UB and fail to compile).
    if H >= 8192:
        TILE_SIZE_M = min(triton.next_power_of_2(M), 16)
        TILE_SIZE_H = min(triton.next_power_of_2(H), 512)
    else:
        TILE_SIZE_H = min(triton.next_power_of_2(H), 256)
        tile_m_cap = min(max(16, 8192 // TILE_SIZE_H), 128)
        # Bigger tiles must leave enough tiles per core: with too few, the
        # tail imbalance (some cores get 5 tiles, others 4) outweighs the
        # larger DMA transfers.
        if triton.cdiv(M, tile_m_cap) < num_cores * 8:
            tile_m_cap = 32
        TILE_SIZE_M = min(triton.next_power_of_2(M), tile_m_cap)
    if M * H < 256 * 64:
        num_cores = 1
    # Split the cores over M first; when M alone still cannot fill them,
    # spread the leftover cores over H so that every core gets a contiguous
    # (BLOCK_SIZE_M, BLOCK_SIZE_H) output slice.
    num_tiles_m = triton.cdiv(M, TILE_SIZE_M)
    num_tiles_h = triton.cdiv(H, TILE_SIZE_H)
    num_cores = min(num_cores, num_tiles_m * num_tiles_h)
    cores_m = min(num_cores, num_tiles_m)
    cores_h = num_cores // cores_m
    # Round the block sizes up to multiples of the tile sizes so that the
    # per-core ranges never overlap; tiles past M/H are masked in the kernel.
    BLOCK_SIZE_M = triton.cdiv(num_tiles_m, cores_m) * TILE_SIZE_M
    BLOCK_SIZE_H = triton.cdiv(num_tiles_h, cores_h) * TILE_SIZE_H
    swiglu_kernel[(cores_m, cores_h)](
        input_a,
        input_b,
        output_2d,
        M,
        H,
        input_a.stride(0),
        output_2d.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        TILE_SIZE_M=TILE_SIZE_M,
        TILE_SIZE_H=TILE_SIZE_H,
        SPLIT_H=cores_h > 1,
        multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    return output_2d.view(*shape[:-1], H_out)
