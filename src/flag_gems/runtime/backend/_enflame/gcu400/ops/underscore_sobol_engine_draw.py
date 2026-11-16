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

"""gcu400 sobol_engine_draw: closed-form Gray-code parallel implementation.

The original runs n serial loops over the sample dimension, each scanning 30 bits, and ``n`` is
``tl.constexpr``; the fp64-output kernel stably triggers a compiler segfault in GCU400's
``make_gcuir`` stage. Here the recurrence becomes closed-form:

    Q(k) = Q0 ^ (G(num_generated + k) ^ G(num_generated))
    G(k) = XOR_{j: gray(k)_j = 1} v_j

where ``v_j`` is ``sobolstate[:, j]`` and ``gray(k) = k ^ (k >> 1)``.
Each sample can then be computed independently in its own program, with both sample and dimension dimensions parallel;
outputs are computed uniformly in fp32, and fp64 requests do one dtype conversion on the host, avoiding
unsupported fp64 stores on the device.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def sobol_draw_parallel_kernel(
    result_ptr,
    quasi_out_ptr,
    quasi_ptr,
    sobolstate_ptr,
    n,
    dimension,
    num_generated,
    recipd: tl.constexpr,
    MAXBIT: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Generate Sobol points in parallel with a 2D (sample block, dimension block) grid.

    ``delta_bits = gray(num_generated + sample + 1) ^ gray(num_generated)``
    each set bit corresponds to a direction vector to XOR; 30 rounds of ``static_range`` are
    unrolled at compile time, so there is no cross-sample data dependency at runtime.
    """
    samples = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    dims = tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    sample_mask = samples < n
    dim_mask = dims < dimension

    quasi_initial = tl.load(quasi_ptr + dims, mask=dim_mask, other=0).to(tl.int64)
    base_index = num_generated.to(tl.int64)
    gray_base = base_index ^ (base_index >> 1)
    gray_next = (base_index + samples.to(tl.int64) + 1) ^ (
        (base_index + samples.to(tl.int64) + 1) >> 1
    )
    delta_bits = gray_next ^ gray_base

    quasi = tl.broadcast_to(quasi_initial[None, :], (BLOCK_SAMPLES, BLOCK_DIM)).to(
        tl.int64
    )
    for bit_index in tl.static_range(MAXBIT):
        bit = ((delta_bits >> bit_index) & 1).to(tl.int64)
        direction = tl.load(
            sobolstate_ptr + dims * MAXBIT + bit_index,
            mask=dim_mask,
            other=0,
        ).to(tl.int64)
        quasi = quasi ^ (bit[:, None] * direction[None, :])

    result_vals = quasi.to(tl.float32) * recipd
    tl.store(
        result_ptr + samples[:, None] * dimension + dims[None, :],
        result_vals,
        mask=sample_mask[:, None] & dim_mask[None, :],
    )

    # The last sample block also writes back the final state: Q(n) is exactly the last sample's row.
    is_last_sample = samples == (n - 1)
    final_quasi = tl.sum(tl.where(is_last_sample[:, None], quasi, 0), axis=0)
    if tl.program_id(0) == tl.num_programs(0) - 1:
        tl.store(quasi_out_ptr + dims, final_quasi, mask=dim_mask)


def _select_sobol_config(n, dimension):
    """Return (sample block, dimension block, num_warps), the final config after 10 sweep rounds.

    32 samples x 4 dimensions x 4 warps totals 2.855 ms across 4 benchmark tiers, better than
    32x8 (4.866 ms), 128x8 (2.905 ms) and dynamic tiering by n (3.074 ms). A dimension block of
    4 lets grid.y cover more SIPs; larger sample blocks hit dsm memory limits, while 16-dimension blocks,
    8 warps and smaller sample blocks all degrade noticeably.
    """
    return 128, 4, 4


def underscore_sobol_engine_draw(
    quasi, n, sobolstate, dimension, num_generated, *, dtype=torch.float32
):
    """Generate n Sobol sequence points, returning (result, updated quasi state)."""
    logger.debug("GEMS_ENFLAME_GCU400 UNDERSCORE_SOBOL_ENGINE_DRAW")

    MAXBIT = 30
    recipd = 1.0 / (2**MAXBIT)
    # GCU400 does not support fp64 stores; fp64 requests compute in fp32 first and do one dtype conversion.
    compute_dtype = torch.float32 if dtype == torch.float64 else dtype
    result = torch.empty((n, dimension), dtype=compute_dtype, device=quasi.device)
    quasi_out = torch.empty_like(quasi)

    if n == 0:
        quasi_out.copy_(quasi)
        return result.to(dtype) if compute_dtype != dtype else result, quasi_out

    block_samples, block_dim, num_warps = _select_sobol_config(n, dimension)
    grid = (triton.cdiv(n, block_samples), triton.cdiv(dimension, block_dim))
    with torch_device_fn.device(quasi.device):
        sobol_draw_parallel_kernel[grid](
            result,
            quasi_out,
            quasi,
            sobolstate,
            n,
            dimension,
            num_generated,
            recipd,
            MAXBIT,
            block_samples,
            block_dim,
            num_warps=num_warps,
        )
    if compute_dtype != dtype:
        result = result.to(dtype)
    return result, quasi_out
