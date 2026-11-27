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
def sobol_draw_kernel(
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
):
    pid_samples = tl.program_id(0)
    dim_index = tl.program_id(1)
    n_progs = tl.num_programs(0)
    total_chunks = (n + BLOCK_SAMPLES - 1) // BLOCK_SAMPLES

    quasi0 = tl.load(quasi_ptr + dim_index).to(tl.int32)
    base_index = num_generated.to(tl.int64)
    gray_base = base_index ^ (base_index >> 1)
    max_index = base_index + n

    v00 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 0).to(tl.int32)
    v01 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 1).to(tl.int32)
    v02 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 2).to(tl.int32)
    v03 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 3).to(tl.int32)
    v04 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 4).to(tl.int32)
    v05 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 5).to(tl.int32)
    v06 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 6).to(tl.int32)
    v07 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 7).to(tl.int32)

    v08 = 0
    v09 = 0
    v10 = 0
    v11 = 0
    v12 = 0
    v13 = 0
    v14 = 0
    v15 = 0
    if max_index >= 256:
        v08 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 8).to(tl.int32)
        v09 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 9).to(tl.int32)
        v10 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 10).to(tl.int32)
        v11 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 11).to(tl.int32)
        v12 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 12).to(tl.int32)
        v13 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 13).to(tl.int32)
        v14 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 14).to(tl.int32)
        v15 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 15).to(tl.int32)

    v16 = 0
    v17 = 0
    v18 = 0
    v19 = 0
    if max_index >= 65536:
        v16 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 16).to(tl.int32)
        v17 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 17).to(tl.int32)
        v18 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 18).to(tl.int32)
        v19 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 19).to(tl.int32)

    v20 = 0
    v21 = 0
    v22 = 0
    v23 = 0
    if max_index >= 1048576:
        v20 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 20).to(tl.int32)
        v21 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 21).to(tl.int32)
        v22 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 22).to(tl.int32)
        v23 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 23).to(tl.int32)

    v24 = 0
    v25 = 0
    v26 = 0
    v27 = 0
    v28 = 0
    v29 = 0
    if max_index >= 16777216:
        v24 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 24).to(tl.int32)
        v25 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 25).to(tl.int32)
        v26 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 26).to(tl.int32)
        v27 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 27).to(tl.int32)
        v28 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 28).to(tl.int32)
        v29 = tl.load(sobolstate_ptr + dim_index * MAXBIT + 29).to(tl.int32)

    # Scalar closed-form final state: quasi_out = Q0 ^ D(gray(base+n) ^ gray(base))
    final_index = base_index + n
    gray_final = final_index ^ (final_index >> 1)
    delta_final = (gray_final ^ gray_base).to(tl.int32)
    qf = quasi0
    qf = qf ^ (v00 & (-((delta_final >> 0) & 1)))
    qf = qf ^ (v01 & (-((delta_final >> 1) & 1)))
    qf = qf ^ (v02 & (-((delta_final >> 2) & 1)))
    qf = qf ^ (v03 & (-((delta_final >> 3) & 1)))
    qf = qf ^ (v04 & (-((delta_final >> 4) & 1)))
    qf = qf ^ (v05 & (-((delta_final >> 5) & 1)))
    qf = qf ^ (v06 & (-((delta_final >> 6) & 1)))
    qf = qf ^ (v07 & (-((delta_final >> 7) & 1)))
    qf = qf ^ (v08 & (-((delta_final >> 8) & 1)))
    qf = qf ^ (v09 & (-((delta_final >> 9) & 1)))
    qf = qf ^ (v10 & (-((delta_final >> 10) & 1)))
    qf = qf ^ (v11 & (-((delta_final >> 11) & 1)))
    qf = qf ^ (v12 & (-((delta_final >> 12) & 1)))
    qf = qf ^ (v13 & (-((delta_final >> 13) & 1)))
    qf = qf ^ (v14 & (-((delta_final >> 14) & 1)))
    qf = qf ^ (v15 & (-((delta_final >> 15) & 1)))
    qf = qf ^ (v16 & (-((delta_final >> 16) & 1)))
    qf = qf ^ (v17 & (-((delta_final >> 17) & 1)))
    qf = qf ^ (v18 & (-((delta_final >> 18) & 1)))
    qf = qf ^ (v19 & (-((delta_final >> 19) & 1)))
    qf = qf ^ (v20 & (-((delta_final >> 20) & 1)))
    qf = qf ^ (v21 & (-((delta_final >> 21) & 1)))
    qf = qf ^ (v22 & (-((delta_final >> 22) & 1)))
    qf = qf ^ (v23 & (-((delta_final >> 23) & 1)))
    qf = qf ^ (v24 & (-((delta_final >> 24) & 1)))
    qf = qf ^ (v25 & (-((delta_final >> 25) & 1)))
    qf = qf ^ (v26 & (-((delta_final >> 26) & 1)))
    qf = qf ^ (v27 & (-((delta_final >> 27) & 1)))
    qf = qf ^ (v28 & (-((delta_final >> 28) & 1)))
    qf = qf ^ (v29 & (-((delta_final >> 29) & 1)))

    # grid-stride sample-block loop: direction vectors stay in registers and are reused across blocks
    for chunk in tl.range(pid_samples, total_chunks, n_progs):
        samples = chunk * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        sample_mask = samples < n
        next_index = base_index + samples.to(tl.int64) + 1
        gray_next = next_index ^ (next_index >> 1)
        delta_bits = (gray_next ^ gray_base).to(tl.int32)

        quasi = tl.full((BLOCK_SAMPLES,), 0, tl.int32) + quasi0
        quasi = quasi ^ (v00 & (-((delta_bits >> 0) & 1)))
        quasi = quasi ^ (v01 & (-((delta_bits >> 1) & 1)))
        quasi = quasi ^ (v02 & (-((delta_bits >> 2) & 1)))
        quasi = quasi ^ (v03 & (-((delta_bits >> 3) & 1)))
        quasi = quasi ^ (v04 & (-((delta_bits >> 4) & 1)))
        quasi = quasi ^ (v05 & (-((delta_bits >> 5) & 1)))
        quasi = quasi ^ (v06 & (-((delta_bits >> 6) & 1)))
        quasi = quasi ^ (v07 & (-((delta_bits >> 7) & 1)))
        if max_index >= 256:
            quasi = quasi ^ (v08 & (-((delta_bits >> 8) & 1)))
            quasi = quasi ^ (v09 & (-((delta_bits >> 9) & 1)))
            quasi = quasi ^ (v10 & (-((delta_bits >> 10) & 1)))
            quasi = quasi ^ (v11 & (-((delta_bits >> 11) & 1)))
            quasi = quasi ^ (v12 & (-((delta_bits >> 12) & 1)))
            quasi = quasi ^ (v13 & (-((delta_bits >> 13) & 1)))
            quasi = quasi ^ (v14 & (-((delta_bits >> 14) & 1)))
            quasi = quasi ^ (v15 & (-((delta_bits >> 15) & 1)))
        if max_index >= 65536:
            quasi = quasi ^ (v16 & (-((delta_bits >> 16) & 1)))
            quasi = quasi ^ (v17 & (-((delta_bits >> 17) & 1)))
            quasi = quasi ^ (v18 & (-((delta_bits >> 18) & 1)))
            quasi = quasi ^ (v19 & (-((delta_bits >> 19) & 1)))
        if max_index >= 1048576:
            quasi = quasi ^ (v20 & (-((delta_bits >> 20) & 1)))
            quasi = quasi ^ (v21 & (-((delta_bits >> 21) & 1)))
            quasi = quasi ^ (v22 & (-((delta_bits >> 22) & 1)))
            quasi = quasi ^ (v23 & (-((delta_bits >> 23) & 1)))
        if max_index >= 16777216:
            quasi = quasi ^ (v24 & (-((delta_bits >> 24) & 1)))
            quasi = quasi ^ (v25 & (-((delta_bits >> 25) & 1)))
            quasi = quasi ^ (v26 & (-((delta_bits >> 26) & 1)))
            quasi = quasi ^ (v27 & (-((delta_bits >> 27) & 1)))
            quasi = quasi ^ (v28 & (-((delta_bits >> 28) & 1)))
            quasi = quasi ^ (v29 & (-((delta_bits >> 29) & 1)))

        result_vals = quasi.to(tl.float32) * recipd
        tl.store(
            result_ptr + samples * dimension + dim_index,
            result_vals,
            mask=sample_mask,
        )

    # Every program computes the same qf; only the last program writes the final state
    if pid_samples == n_progs - 1:
        tl.store(quasi_out_ptr + dim_index, qf.to(tl.int64))


def _select_sobol_config(n, dimension):
    """Returns (BLOCK_SAMPLES, num_warps), finalized by the 2026-11-17 GCU400 sweep.

    Small n uses 512 sample blocks with one warp (launch/event floor dominates); the middle
    range uses 2048 x 2 warps so each program gets one block at the 10K tier; large n falls
    back to 1024 x 2 warps, paying the load chain per program rather than per block. 4096 x 2
    warps exceeds the register budget (measured 20.79/23.44/24.17/37.80us regressions).
    """
    if n <= 2048:
        return 512, 1
    if n <= 32768:
        return 2048, 2
    return 1024, 2


def underscore_sobol_engine_draw(
    quasi, n, sobolstate, dimension, num_generated, *, dtype=torch.float32
):
    """Generates n Sobol sequence points, returning (result, updated quasi state)."""
    logger.debug("GEMS_ENFLAME UNDERSCORE_SOBOL_ENGINE_DRAW")

    MAXBIT = 30
    recipd = 1.0 / (2**MAXBIT)
    # GCU400 does not support fp64 stores; fp64 requests are computed in fp32 first and then undergo one dtype
    # conversion.
    compute_dtype = torch.float32 if dtype == torch.float64 else dtype
    result = torch.empty((n, dimension), dtype=compute_dtype, device=quasi.device)
    quasi_out = torch.empty_like(quasi)

    if n == 0:
        quasi_out.copy_(quasi)
        return result.to(dtype) if compute_dtype != dtype else result, quasi_out

    block_samples, num_warps = _select_sobol_config(n, dimension)
    total_chunks = triton.cdiv(n, block_samples)
    # Cap the number of programs per dimension at ceil(48/dimension): the total number of programs stays near ~48 SIPs,
    # the direction-vector load chain is paid only once per program, and the remaining sample blocks are consumed by the
    # grid-stride loop.
    programs_per_dim = max(1, -(-48 // dimension))
    grid = (min(total_chunks, programs_per_dim), dimension)
    with torch_device_fn.device(quasi.device):
        sobol_draw_kernel[grid](
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
            num_warps=num_warps,
        )
    if compute_dtype != dtype:
        result = result.to(dtype)
    return result, quasi_out
