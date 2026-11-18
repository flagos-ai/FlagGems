# Copyright 2026, The FlagOS Contributors.
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

"""Kunlunxin (XPU) override of ``_sobol_engine_ff_``.

The generic kernel (``flag_gems.ops._sobol_engine_ff_``) uses one program per
dimension with fully scalar int64 loads / XOR / stores.  On this TritonXPU
stack scalar integer ``^`` (and other scalar integer arithmetic) silently
drops the result on random lanes — an ``identity``-copy kernel and a
``store``-constant kernel are bit-exact, while ``v ^ w`` returns the
pre-write value on a random subset of the programs (confirmed with isolated
probes; the same garbage/silent-drop shows for ``-``, ``*`` and ``<<``).
The ``underscore_sobol_engine_draw`` kernels, which process a whole
DIM_BLOCK-wide vector with ``mask=.../other=0`` loads, are correct on XPU,
so this override uses the same vectorized-over-dimensions layout: one
program, DIM_BLOCK lanes, sequential over the ``n`` fast-forward steps.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _sobol_engine_ff_kernel(
    quasi_ptr,
    sobolstate_ptr,
    dimension: tl.constexpr,
    n: tl.constexpr,
    num_generated,
    MAXBIT: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    """
    Sobol engine fast-forward kernel using Gray code bit manipulation.

    The algorithm XORs direction numbers corresponding to the rightmost set
    bit of each step in the range ``(num_generated, num_generated + n]``.

    All dimensions are processed simultaneously as a DIM_BLOCK-wide vector
    (the same layout as the ``sobol_draw_kernel_simple``/``sobol_engine_draw_kernel``
    kernels that are known to be correct on TritonXPU).  The per-step XOR is
    associative, so the range collapses to the parity of how often each
    direction is applied, evaluated with a MAXBIT-step static scan rather than
    a length-``n`` serial loop.
    """
    dims = tl.arange(0, DIM_BLOCK)
    dim_mask = dims < dimension

    # Load initial quasi values for this dimension block
    quasi_vals = tl.load(quasi_ptr + dims, mask=dim_mask, other=0).to(tl.int64)

    # Fast-forward by n steps in O(MAXBIT) per dimension.
    #
    # The recurrence applies direction v[j] once for every step whose
    # rightmost set bit is j (i.e. ctz(step) == j); applying the same v[j] an
    # even number of times cancels under XOR.  So the whole range can be
    # reduced to the parity of how many of the n steps fall in each class j.
    #
    # For e = num_generated + n, the number of integers i in (s, s+n] with
    # ctz(i) == j is  (e>>j - e>>(j+1)) - (s>>j - s>>(j+1)),  whose parity is
    # all that matters.  This replaces a length-n serial loop with a branchless
    # closed form: no loop over n, only a 30-step static scan.  (Verified
    # element-for-element against the original trailing-zero loop on CPU for
    # 3000+ (num_generated, n) combinations, including the benchmark shapes.)
    end = num_generated + n
    for j in tl.static_range(MAXBIT):
        count = ((end >> j) - (end >> (j + 1))) - (
            (num_generated >> j) - (num_generated >> (j + 1))
        )
        odd = (count & 1) == 1
        # Load the direction number unconditionally (masked only by the
        # dimension bound) and select with ``tl.where``.  A load whose mask
        # folds in the runtime scalar ``odd`` (``mask=dim_mask & odd``) is
        # silently dropped by this TritonXPU build -- every lane returns the
        # fill value, so the step never applies its XOR.  The unconditional
        # load + where is bit-exact (verified on device against the CPU
        # reference) and keeps the O(MAXBIT) closed form.
        direction_vals = tl.load(
            sobolstate_ptr + dims * MAXBIT + j,
            mask=dim_mask,
            other=0,
        ).to(tl.int64)
        quasi_vals = quasi_vals ^ tl.where(
            odd, direction_vals, tl.zeros_like(direction_vals)
        )

    # Store result
    tl.store(quasi_ptr + dims, quasi_vals, mask=dim_mask)


def _sobol_engine_ff_(quasi, n, sobolstate, dimension, num_generated):
    """
    Fast-forward Sobol quasi-random number generator state.

    Args:
        quasi: int64 tensor of shape (dimension,) - state to update in-place
        n: int - number of steps to advance
        sobolstate: int64 tensor of shape (dimension, MAXBIT) - direction numbers
        dimension: int - dimensionality of Sobol sequence
        num_generated: int - current position in sequence

    Returns:
        quasi: the updated state tensor (modified in-place)
    """
    logger.debug("GEMS_KUNLUNXIN _SOBOL_ENGINE_FF_")

    # Sobol direction matrices use MAXBIT=30
    MAXBIT = 30
    DIM_BLOCK = triton.next_power_of_2(dimension)

    # One program processes all dimensions as a vector.
    grid = (1,)

    _sobol_engine_ff_kernel[grid](
        quasi,
        sobolstate,
        dimension,
        n,
        num_generated,
        MAXBIT,
        DIM_BLOCK,
    )

    return quasi
