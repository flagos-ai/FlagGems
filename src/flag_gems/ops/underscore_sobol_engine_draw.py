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


@triton.jit
def sobol_engine_draw_kernel(
    result_ptr,
    quasi_out_ptr,
    quasi_ptr,
    sobolstate_ptr,
    n,
    dimension,
    num_generated,
    recipd: tl.constexpr,
    MAXBIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Generate n Sobol sequence points - blocked version for larger workloads.

    This kernel is still sequential in the sample dimension because each sample
    depends on the quasi state from the previous sample. We block over dimensions.
    """
    pid_dim = tl.program_id(0)

    # Load dimension range for this block
    dim_offs = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offs < dimension

    # Load initial quasi state for these dimensions
    quasi_vals = tl.load(quasi_ptr + dim_offs, mask=dim_mask, other=0).to(tl.int64)

    # Generate all n samples sequentially
    for i in range(n):
        # Find rightmost zero bit position of (num_generated + i)
        val = num_generated + i
        bit_pos = 0
        found = 0
        temp = val
        for bit_idx in range(MAXBIT):
            is_one = temp & 1
            bit_pos = tl.where((found == 0) & (is_one == 1), bit_idx + 1, bit_pos)
            found = tl.where((found == 0) & (is_one == 0), 1, found)
            temp = temp >> 1

        # XOR quasi with sobolstate[:, bit_pos] for this dimension block
        direction_vals = tl.load(
            sobolstate_ptr + dim_offs * MAXBIT + bit_pos, mask=dim_mask, other=0
        ).to(tl.int64)
        quasi_vals = quasi_vals ^ direction_vals

        # Store result as float in [0, 1)
        result_vals = quasi_vals.to(tl.float32) * recipd
        result_idx = i * dimension + dim_offs
        tl.store(result_ptr + result_idx, result_vals, mask=dim_mask)

    # Store final quasi state
    tl.store(quasi_out_ptr + dim_offs, quasi_vals, mask=dim_mask)


@triton.jit
def sobol_draw_kernel_simple(
    result_ptr,
    quasi_out_ptr,
    quasi_ptr,
    sobolstate_ptr,
    n: tl.constexpr,
    dimension: tl.constexpr,
    num_generated,
    recipd: tl.constexpr,
    MAXBIT: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    """
    Sequential Sobol generation kernel.
    Each sample depends on the previous quasi state, so we process sequentially.
    Only use this for small n where sequential processing is acceptable.
    """
    # Load initial quasi state (shared across all samples)
    # DIM_BLOCK is next_power_of_2(dimension)
    dim_range = tl.arange(0, DIM_BLOCK)
    dim_mask = dim_range < dimension
    quasi_vals = tl.load(quasi_ptr + dim_range, mask=dim_mask, other=0).to(tl.int64)

    # Generate each sample sequentially
    for i in range(n):
        # Find rightmost zero bit of (num_generated + i)
        val = num_generated + i
        bit_pos = 0
        found = 0
        temp = val
        for bit_idx in range(MAXBIT):
            is_one = temp & 1
            bit_pos = tl.where((found == 0) & (is_one == 1), bit_idx + 1, bit_pos)
            found = tl.where((found == 0) & (is_one == 0), 1, found)
            temp = temp >> 1

        # XOR quasi with direction vectors at position bit_pos
        direction_vals = tl.load(
            sobolstate_ptr + dim_range * MAXBIT + bit_pos, mask=dim_mask, other=0
        ).to(tl.int64)
        quasi_vals = quasi_vals ^ direction_vals

        # Convert to float and store result
        result_vals = quasi_vals.to(tl.float32) * recipd
        result_offs = i * dimension + dim_range
        tl.store(result_ptr + result_offs, result_vals, mask=dim_mask)

    # Store final quasi state
    tl.store(quasi_out_ptr + dim_range, quasi_vals, mask=dim_mask)


def underscore_sobol_engine_draw(
    quasi, n, sobolstate, dimension, num_generated, *, dtype=torch.float32
):
    """
    Generate n quasi-random Sobol sequence points.

    Args:
        quasi: Current state vector [dimension], dtype=int64
        n: Number of points to generate
        sobolstate: Direction vectors [dimension, MAXBIT], dtype=int64
        dimension: Dimensionality of the sequence
        num_generated: Number of points generated so far
        dtype: Output dtype (float32 or float64)

    Returns:
        result: Generated points [n, dimension], dtype=dtype
        quasi_out: Updated state vector [dimension], dtype=int64
    """
    logger.debug("GEMS UNDERSCORE_SOBOL_ENGINE_DRAW")

    MAXBIT = 30
    recipd = 1.0 / (2**MAXBIT)

    # Allocate output tensors
    result = torch.empty((n, dimension), dtype=dtype, device=quasi.device)
    quasi_out = torch.empty_like(quasi)

    if dimension <= 32 and n <= 10000:
        # Use simple sequential kernel for small cases
        # Only need 1 program since it processes all samples sequentially
        DIM_BLOCK = triton.next_power_of_2(dimension)
        grid = (1,)
        sobol_draw_kernel_simple[grid](
            result,
            quasi_out,
            quasi,
            sobolstate,
            n,
            dimension,
            num_generated,
            recipd,
            MAXBIT,
            DIM_BLOCK,
        )
    else:
        # Use blocked kernel for larger cases
        # Still sequential in sample dimension, but parallel over dimensions
        BLOCK_DIM = min(32, triton.next_power_of_2(dimension))
        grid = (triton.cdiv(dimension, BLOCK_DIM),)

        sobol_engine_draw_kernel[grid](
            result,
            quasi_out,
            quasi,
            sobolstate,
            n,
            dimension,
            num_generated,
            recipd,
            MAXBIT,
            0,  # BLOCK_N unused now
            BLOCK_DIM,
        )

    return result, quasi_out
