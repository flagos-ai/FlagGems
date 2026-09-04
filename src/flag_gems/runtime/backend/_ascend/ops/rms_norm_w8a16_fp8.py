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

"""Ascend W8A16 RMSNorm.

Activation is 16-bit (FP16/BF16). Weight is grouped INT8 plus per-group scale
because Ascend UB / compiler cannot load FP8. Layout matches the NVIDIA
FP8-W8A16 RMSNorm path: group_size=128.

Dispatch:
- Power-of-two N <= 4096: 1D kernel with BLOCK_M rows so INT8 weight +
  scale are loaded once per program and reused. Mid-size M uses
  BLOCK_M=8; small and very large M use BLOCK_M=2 (higher occupancy).
- Power-of-two N <= 8192: 1D row kernel. Full-row BLOCK_M overflows
  Ascend UB by a few hundred bytes. Scale is loaded uniquely and
  broadcast via reshape (no ``cols // group_size`` gather, no ``(G, 128)``
  FP32 2D tile — that layout has a 512B row stride and conflicts on UB).
- Otherwise: tiled 1D kernel (GROUPS_PER_TILE=64) to stay in UB
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Unique scale load + broadcast. Gathering ``scale[cols // GROUP_SIZE]``
    # is scalar-bound on Ascend (~7x slower than the grouped kernel).
    w_scale = tl.load(w_scale_ptr + tl.arange(0, NUM_GROUPS)).to(tl.float32)
    y = tl.reshape(
        tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
        * rrms
        * tl.reshape(w, (NUM_GROUPS, GROUP_SIZE))
        * w_scale[:, None],
        (BLOCK_SIZE,),
    )
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_blockm_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask_n = cols < N
    # Weight is reused across BLOCK_M rows; reload-per-row is extra HBM
    # and extra scalar prologue when M is large.
    w = tl.load(w_ptr + cols, mask=mask_n, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + tl.arange(0, NUM_GROUPS)).to(tl.float32)
    w2 = tl.reshape(w, (NUM_GROUPS, GROUP_SIZE))
    for i in range(0, BLOCK_M):
        row = pid * BLOCK_M + i
        row_mask = row < M
        x = tl.load(in_ptr + row * N + cols, mask=mask_n & row_mask, other=0.0).to(
            tl.float32
        )
        rrms = 1 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        y = tl.reshape(
            tl.reshape(x, (NUM_GROUPS, GROUP_SIZE)) * rrms * w2 * w_scale[:, None],
            (BLOCK_SIZE,),
        )
        tl.store(out_ptr + row * N + cols, y, mask=mask_n & row_mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_grouped_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = ext.program_id(0)
    # Contiguous 1D load. A 2D ``(NUM_GROUPS, 128)`` FP32 tile has a 512B
    # row stride (16 UB banks * 32B) and bank-conflicts on Ascend.
    BLOCK: tl.constexpr = NUM_GROUPS * GROUP_SIZE
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + tl.arange(0, NUM_GROUPS)).to(tl.float32)
    y = tl.reshape(
        tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
        * rrms
        * tl.reshape(w, (NUM_GROUPS, GROUP_SIZE))
        * w_scale[:, None],
        (BLOCK,),
    )
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_loop_kernel_fixed(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_grouped_tiled_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
):
    pid = ext.program_id(0)
    TILE_N: tl.constexpr = GROUPS_PER_TILE * GROUP_SIZE
    num_groups = N // GROUP_SIZE

    acc = 0.0
    for g0 in range(0, num_groups, GROUPS_PER_TILE):
        start_n = g0 * GROUP_SIZE
        cols = start_n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x)
    rrms = 1 / tl.sqrt(acc / N + eps)

    for g0 in range(0, num_groups, GROUPS_PER_TILE):
        start_n = g0 * GROUP_SIZE
        cols = start_n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        groups = g0 + tl.arange(0, GROUPS_PER_TILE)
        gmask = groups < num_groups
        w_scale = tl.load(w_scale_ptr + groups, mask=gmask, other=0.0).to(tl.float32)
        y = tl.reshape(
            tl.reshape(x, (GROUPS_PER_TILE, GROUP_SIZE))
            * rrms
            * tl.reshape(w, (GROUPS_PER_TILE, GROUP_SIZE))
            * w_scale[:, None],
            (TILE_N,),
        )
        tl.store(out_ptr + pid * N + cols, y, mask=mask)


def rms_norm_w8a16_fp8(
    x, normalized_shape, weight_fp8, weight_scale, eps=1e-5, group_size=128
):
    logger.debug("GEMS_ASCEND RMS_NORM W8A16 FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    if N % group_size != 0:
        raise ValueError(
            f"normalized_shape product {N} must be divisible by group_size={group_size}"
        )
    if weight_fp8.dtype != torch.int8:
        raise TypeError(
            f"Ascend W8A16 RMSNorm expects INT8 weight, got {weight_fp8.dtype}"
        )
    if weight_scale.numel() != N // group_size:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} != {N // group_size} groups"
        )
    x = x.contiguous()
    weight_q = weight_fp8.contiguous()
    weight_scale = weight_scale.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    num_groups = N // group_size
    with torch_device_fn.device(x.device):
        # Power-of-two N <= 8192: contiguous 1D load + unique scale broadcast.
        # Do not gather on ``cols // group_size`` (~97% scalar) and do not use
        # a ``(G, 128)`` FP32 2D tile (512B row stride, UB bank conflict).
        if N <= 4096 and N == triton.next_power_of_2(N):
            # BLOCK_M=8 is best for a few hundred rows; BLOCK_M=2 wins on
            # tiny M (occupancy) and large M (more programs, same reuse).
            if 256 <= M < 1024:
                block_m, num_warps = 8, 2
            else:
                block_m = 2
                num_warps = 2 if 16 <= M < 256 else 4
            grid = triton.cdiv(M, block_m)
            rms_norm_fp8_w8a16_blockm_kernel[grid,](
                y,
                x,
                weight_q,
                weight_scale,
                M,
                N,
                eps,
                N,
                group_size,
                num_groups,
                block_m,
                num_warps=num_warps,
            )
        elif N <= 8192 and N == triton.next_power_of_2(N):
            rms_norm_fp8_w8a16_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                N,
                group_size,
                num_groups,
                num_warps=4,
            )
        else:
            # 128*128 grouped tile overflows Ascend UB; 64*128 1D tiles are safe.
            rms_norm_fp8_w8a16_grouped_tiled_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                GROUPS_PER_TILE=64,
                num_warps=4,
            )
    return y
