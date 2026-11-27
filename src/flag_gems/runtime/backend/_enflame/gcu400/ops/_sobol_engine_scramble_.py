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

"""gcu400 _sobol_engine_scramble_: aten-faithful GF(2) bit transform via int8 MMA.

aten semantics (aten/src/ATen/native/SobolEngineOps.cpp): every state slot is
scrambled INDEPENDENTLY - the transform never mixes slots. For dimension d and
slot j (bits b of v = sobolstate[d][j], output bits q):

    v_new[j] = sum_q 2^q * (sum_b ltm'[29-q][29-b] * bit_b(v[j]) mod 2)
    ltm' = ltm with the diagonal FORCED to 1 (fill_(1), overwrite)
    bits >= 30 of the input are ignored and the output is always < 2^30 (assign).

The previous gcu400/generic kernels instead XOR slots into each other via
parity(sum_{k<j} ltm[k][j] * state[k]); that only coincides with aten on the
degenerate test subspace (0/1 states + tril ltm) where both sides are the
identity. This rewrite matches aten bit-exactly on arbitrary states (0/1 and
full 30-bit values, tril and non-tril ltm, 43-case GCU matrix + 300-case CPU
mirror against torch._sobol_engine_scramble_).

Performance (GCU400): the old kernel ran 30 serial rounds per program, each
with a strided (BLOCK_D, 32) ltm load + where + tl.sum reduction + where -
roughly 240 serial reduction barriers per program (measured ~0.9-1.1 us per
small tl.sum on this part), which dominated the 2.97 ms @ dim=5000 baseline.
This kernel has exactly ONE tl.dot + ONE tl.sum per dimension:

  * the GF(2) contraction rides the int8 MMA path: Sel[j, b] = bit_b(v[j])
    times At[b, q] = ltm'[29-q][29-b] (loaded through an int32 low-word view),
    accumulated in int32, and aten's ``t1 % 2`` becomes ``R & 1`` - exact,
    no carries (products are 0/1, sums <= 30).
  * the parity bits are repacked with a single last-axis tl.sum
    (summands hit disjoint bits, so the int32 sum is carry-free).
  * grid = min(dimension, 48) programs with a grid-stride loop over dimensions:
    48 concurrent programs is the measured GCU400 sweet spot (empty-kernel
    dispatch costs ~0.05 us/program, so per-dimension grids pay a hard
    ~255 us floor at dim=5000).
  * ltm is read through int32 low words (entries are 0/1); the DRAM lines are
    the same either way, so int64/int32/int8 loads measure identically - the
    low-word view just halves the lane traffic.

Coalescing / occupancy pass (this revision):
  * the ltm' tile is now loaded as ``At2[q, b]`` so that the *source-bit* axis
    (stride 8 B in the int64 ltm) is the fast axis of the tile. The previous
    ``At[b, q]`` layout walked 240 B (one full 30-int64 row) per lane, i.e. one
    cache line per lane; the row-major layout makes the 32 lanes of a row land
    in ~4 consecutive lines.
  * the transpose is folded into the operand build: ``Sel^T[b, j]`` is built
    directly from ``v`` with a swapped broadcast, so no ``tl.trans`` is needed.
  * ``num_warps=1``: the tiles are 32x32 int8, so one warp already issues the
    whole MMA; extra warps only pay barrier/DSM overhead (4 warps 0.757 ms ->
    1 warp 0.391 ms at dim=5000).

Benchmark (do_bench median, dim = 100/500/1000/5000, num_warps=1):
53/165/307/1459 us -> 23/54/91/391 us, arithmetic mean 3.1x on top of the
previous rewrite (0.39 ms at dim=5000, i.e. ~3.7x over the MMA rewrite alone).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAXBIT = 30

# Measured GCU400 concurrency sweet spot for this kernel (see module docstring).
_SCRAMBLE_PROGRAMS = 48


@triton.jit
def sobol_scramble_kernel(
    sobolstate_ptr,
    ltm_ptr,
    dimension,
    MAXBIT: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    jl = tl.arange(0, 32)  # slot lanes (30 valid)
    bl = tl.arange(0, 32)  # source-bit lanes (30 valid)
    ql = tl.arange(0, 32)  # output-bit lanes (30 valid)
    bm = bl < MAXBIT
    jm = jl < MAXBIT
    qm = ql < MAXBIT
    # ltm entries are {0, 1} int64; read the low 32-bit words (little-endian).
    ltm32 = ltm_ptr.to(tl.pointer_type(tl.int32))
    # Repack weights: acc[j] = sum_q (R & 1) << q. Lanes >= 30 read 0 from the
    # masked At tile, so their never-shifted weight receives no set bit.
    qsh = tl.where(qm, ql, 0)
    qw = 1 << qsh

    for dd in range(pid, dimension, NPROG):
        # At2[q, b] = ltm'[dd][29-q][29-b]; diagonal (b == q) forced to 1.
        # Axis order matters: b has stride 8 B in the int64 ltm, so keeping it
        # as the fast axis turns the old "one cache line per lane" pattern into
        # consecutive lines per row.
        at2 = tl.load(
            ltm32
            + 2
            * (
                dd * (MAXBIT * MAXBIT)
                + (29 - ql)[:, None] * MAXBIT
                + (29 - bl)[None, :]
            ),
            mask=qm[:, None] & bm[None, :],
            other=0,
        )
        at2 = tl.where(ql[:, None] == bl[None, :], 1, at2).to(tl.int8)
        v64 = tl.load(sobolstate_ptr + dd * MAXBIT + jl, mask=jm, other=0)
        v = (v64 & 0x3FFFFFFF).to(tl.int32)  # aten reads only bits 0..29
        # Sel^T[b, j] = bit_b(v[j]) built directly with a swapped broadcast, so
        # no tl.trans is needed. The MMA accumulates sum_b At2 * Sel^T in int32
        # and aten's `t1 % 2` is exactly `R & 1` (products are 0/1, no carries).
        sel_t = ((v[None, :] >> bl[:, None]) & 1).to(tl.int8)
        r_t = tl.dot(at2, sel_t, out_dtype=tl.int32)
        # Disjoint output bits -> the int32 sum over the q axis is carry-free.
        acc = tl.sum((r_t & 1) * qw[:, None], axis=0)
        tl.store(sobolstate_ptr + dd * MAXBIT + jl, acc.to(tl.int64), mask=jm)


def _sobol_engine_scramble_(sobolstate, ltm, dimension):
    """Owen scrambling of the Sobol state (per-slot GF(2) bit transform, in place)."""
    logger.debug("GEMS_ENFLAME _SOBOL_ENGINE_SCRAMBLE_")

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

    grid = (min(dimension, _SCRAMBLE_PROGRAMS),)
    sobol_scramble_kernel[grid](
        sobolstate,
        ltm,
        dimension,
        MAXBIT=MAXBIT,
        NPROG=_SCRAMBLE_PROGRAMS,
        num_warps=1,
    )
    return sobolstate
