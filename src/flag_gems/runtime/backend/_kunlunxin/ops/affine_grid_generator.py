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

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def affine_grid_generator_kernel(
    output_ptr,
    theta_ptr,
    N,
    D,
    H,
    W,
    OUTPUT_STRIDE0,
    OUTPUT_STRIDE1,
    OUTPUT_STRIDE2,
    OUTPUT_STRIDE3,
    OUTPUT_STRIDE4,
    THETA_STRIDE0,
    THETA_STRIDE1,
    THETA_STRIDE2,
    ALIGN_CORNERS: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    coordinate_count: tl.constexpr = 3 if IS_3D else 2
    num_tasks = N * D * H * W * coordinate_count
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tasks

    coordinate = offsets % coordinate_count
    spatial_index = offsets // coordinate_count
    w = spatial_index % W
    spatial_index = spatial_index // W
    h = spatial_index % H
    spatial_index = spatial_index // H
    d = spatial_index % D
    n = spatial_index // D

    w_float = w.to(tl.float32)
    h_float = h.to(tl.float32)
    d_float = d.to(tl.float32)
    w_size = tl.full((), W, tl.float32)
    h_size = tl.full((), H, tl.float32)
    d_size = tl.full((), D, tl.float32)

    if ALIGN_CORNERS:
        # PyTorch defines the unit-sized dimension coordinate as zero.
        x = tl.where(W > 1, 2.0 * w_float / (w_size - 1.0) - 1.0, 0.0)
        y = tl.where(H > 1, 2.0 * h_float / (h_size - 1.0) - 1.0, 0.0)
        z = tl.where(D > 1, 2.0 * d_float / (d_size - 1.0) - 1.0, 0.0)
    else:
        x = (2.0 * w_float + 1.0) / w_size - 1.0
        y = (2.0 * h_float + 1.0) / h_size - 1.0
        z = (2.0 * d_float + 1.0) / d_size - 1.0

    theta_base = n * THETA_STRIDE0
    theta_00 = tl.load(theta_ptr + theta_base).to(tl.float32)
    theta_01 = tl.load(theta_ptr + theta_base + THETA_STRIDE2).to(tl.float32)
    theta_02 = tl.load(theta_ptr + theta_base + 2 * THETA_STRIDE2).to(tl.float32)
    theta_10 = tl.load(theta_ptr + theta_base + THETA_STRIDE1).to(tl.float32)
    theta_11 = tl.load(
        theta_ptr + theta_base + THETA_STRIDE1 + THETA_STRIDE2
    ).to(tl.float32)
    theta_12 = tl.load(
        theta_ptr + theta_base + THETA_STRIDE1 + 2 * THETA_STRIDE2
    ).to(tl.float32)

    if IS_3D:
        theta_03 = tl.load(
            theta_ptr + theta_base + 3 * THETA_STRIDE2
        ).to(tl.float32)
        theta_13 = tl.load(
            theta_ptr + theta_base + THETA_STRIDE1 + 3 * THETA_STRIDE2
        ).to(tl.float32)
        theta_20 = tl.load(
            theta_ptr + theta_base + 2 * THETA_STRIDE1
        ).to(tl.float32)
        theta_21 = tl.load(
            theta_ptr + theta_base + 2 * THETA_STRIDE1 + THETA_STRIDE2
        ).to(tl.float32)
        theta_22 = tl.load(
            theta_ptr + theta_base + 2 * THETA_STRIDE1 + 2 * THETA_STRIDE2
        ).to(tl.float32)
        theta_23 = tl.load(
            theta_ptr + theta_base + 2 * THETA_STRIDE1 + 3 * THETA_STRIDE2
        ).to(tl.float32)
        result = tl.where(
            coordinate == 0,
            theta_00 * x + theta_01 * y + theta_02 * z + theta_03,
            tl.where(
                coordinate == 1,
                theta_10 * x + theta_11 * y + theta_12 * z + theta_13,
                theta_20 * x + theta_21 * y + theta_22 * z + theta_23,
            ),
        )
        output_offset = (
            n * OUTPUT_STRIDE0
            + d * OUTPUT_STRIDE1
            + h * OUTPUT_STRIDE2
            + w * OUTPUT_STRIDE3
            + coordinate * OUTPUT_STRIDE4
        )
    else:
        result = tl.where(
            coordinate == 0,
            theta_00 * x + theta_01 * y + theta_02,
            theta_10 * x + theta_11 * y + theta_12,
        )
        output_offset = (
            n * OUTPUT_STRIDE0
            + h * OUTPUT_STRIDE1
            + w * OUTPUT_STRIDE2
            + coordinate * OUTPUT_STRIDE3
        )

    tl.store(output_ptr + output_offset, result, mask=mask)


def affine_grid_generator(
    theta: torch.Tensor, size: torch.Size, align_corners: bool
) -> torch.Tensor:
    logger.debug("GEMS AFFINE_GRID_GENERATOR")

    if len(size) == 4:
        n, _, h, w = size
        if theta.shape != (n, 2, 3):
            raise RuntimeError(
                f"Expected theta to have shape ({n}, 2, 3), got {tuple(theta.shape)}"
            )
        output = torch.empty((n, h, w, 2), dtype=theta.dtype, device=theta.device)
        d = 1
        is_3d = False
    elif len(size) == 5:
        n, _, d, h, w = size
        if theta.shape != (n, 3, 4):
            raise RuntimeError(
                f"Expected theta to have shape ({n}, 3, 4), got {tuple(theta.shape)}"
            )
        output = torch.empty((n, d, h, w, 3), dtype=theta.dtype, device=theta.device)
        is_3d = True
    else:
        raise RuntimeError(
            f"Expected size to have 4 or 5 dimensions, got {len(size)} dimensions"
        )

    block_size = 256
    coordinate_count = 3 if is_3d else 2
    grid = (triton.cdiv(n * d * h * w * coordinate_count, block_size),)
    affine_grid_generator_kernel[grid](
        output,
        theta,
        n,
        d,
        h,
        w,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4) if is_3d else 1,
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        ALIGN_CORNERS=align_corners,
        IS_3D=is_3d,
        BLOCK_SIZE=block_size,
    )
    return output
