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

logger = logging.getLogger(__name__)

MAXBIT = 30


@triton.jit
def sobol_scramble_kernel(
    sobolstate_ptr,
    ltm_ptr,
    dimension,
    MAXBIT: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 一个 CTA 处理 BLOCK_D 个维度，减少大 dimension 下的 CTA 调度开销
    dim_offsets = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < dimension
    rows = tl.arange(0, BLOCK)
    bit_mask = rows < MAXBIT
    tile_mask = dim_mask[:, None] & bit_mask[None, :]

    state = tl.load(
        sobolstate_ptr + dim_offsets[:, None] * MAXBIT + rows[None, :],
        mask=tile_mask,
        other=0,
    )

    for j in tl.static_range(0, MAXBIT):
        ltm_col_offsets = (
            dim_offsets[:, None] * MAXBIT * MAXBIT + rows[None, :] * MAXBIT + j
        )
        load_mask = tile_mask
        ltm_col = tl.load(ltm_ptr + ltm_col_offsets, mask=load_mask, other=0)

        k_mask = rows[None, :] < j
        prod = ltm_col & state
        prod = tl.where(k_mask, prod, 0)
        parity = tl.sum(prod.to(tl.int32), axis=1) % 2
        state = tl.where(rows[None, :] == j, state ^ parity[:, None], state)

    tl.store(
        sobolstate_ptr + dim_offsets[:, None] * MAXBIT + rows[None, :],
        state,
        mask=tile_mask,
    )


def _sobol_engine_scramble_(sobolstate, ltm, dimension):
    """对 Sobol state 做 Owen 置乱（GF(2) 下三角变换，原地修改）。"""
    logger.debug("GEMS _SOBOL_ENGINE_SCRAMBLE_")

    assert (
        sobolstate.dtype == torch.long
    ), f"sobolstate must be int64, got {sobolstate.dtype}"
    assert ltm.dtype == torch.long, f"ltm must be int64, got {ltm.dtype}"
    assert sobolstate.shape == (
        dimension,
        MAXBIT,
    ), f"sobolstate shape must be ({dimension}, {MAXBIT}), got {sobolstate.shape}"
    assert ltm.shape == (
        dimension,
        MAXBIT,
        MAXBIT,
    ), f"ltm shape must be ({dimension}, {MAXBIT}, {MAXBIT}), got {ltm.shape}"

    grid = (triton.cdiv(dimension, 8),)
    sobol_scramble_kernel[grid](
        sobolstate,
        ltm,
        dimension,
        MAXBIT=MAXBIT,
        BLOCK=32,
        BLOCK_D=8,
        num_warps=4,
    )
    return sobolstate
