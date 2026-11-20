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
program, DIM_BLOCK lanes.

Performance note (closed-form fast-forward): the naive loop executes ``n``
masked gather loads (the S4 scan bottleneck, 99.94% of time in this kernel),
with ``n`` up to 4096.  XOR is commutative, so the net XOR equals, for each
bit ``b`` in [0, MAXBIT), flipping ``dir[., b]`` iff the parity of the number
of indices ``x`` in ``(num_generated, num_generated+n]`` with trailing-zero
count ``v2(x) == b`` is odd.  With ``N = num_generated`` that parity count is

    count_b = ((N+n)>>b) - (N>>b) - ((N+n)>>(b+1)) + (N>>(b+1))

which is computed in closed form (O(MAXBIT) instead of O(n)); the identity
was verified against the naive loop on 20000 random (N, n, direcs) tuples
(see ``harness/perf_ir/verify_closed_form.py``, CLOSED_FORM_EQUIVALENT=YES).

Round-2 micro-optimization: when ``dimension`` is a power of two
(``dimension == DIM_BLOCK``, i.e. the whole benchmark matrix), the
always-true ``dim_mask`` is dropped from the final store, which then goes
through the unmasked path.  The masked-memory path costs about 2x on this
backend even for a constant-true mask; measured on P800 the dtype-balanced
Gems speedup rises by ~2.5% (dim=256 cases ~5.6%).

Masked-load caveat (CI compiler, flagtree 0.7.1a1+xpu3.6): a ``tl.load``
whose mask is *not* a compile-time constant silently drops the predicate on
this stack -- the load returns the unmasked data for every lane.  With the
runtime parity in the mask this XOR-ed every direction number in (the
closed-form result degenerated to the XOR of all MAXBIT directions).  The
direction-number loads are therefore unconditional: the indices are clamped
into bounds and the parity is applied arithmetically (0 or all-ones).
Only compile-time masks (``dim_mask``, from the ``dimension`` constexpr) are
used as load/store predicates.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ``num_generated`` must stay a runtime scalar: Triton specializes an integer
# argument equal to 1 into a compile-time Python int, and ``num_generated`` is
# 1-based once the engine has produced its first point -- the in-kernel
# ``num_generated.to(tl.int64)`` then fails with "'int' object has no
# attribute 'to'".  The test matrix has no num_generated == 1 case, so this is
# only caught by direct calls.
@triton.jit(do_not_specialize=["num_generated"])
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
    Sobol engine fast-forward kernel using the closed-form fast-forward
    identity (O(MAXBIT), see module docstring).

    The naive Gray-code loop XORs ``dir[v2(N+i)]`` for i=1..n; by
    commutativity the result is obtained by XOR-ing ``dir[., b]`` for each
    bit ``b`` whose count parity in ``(N, N+n]`` is odd.  All dimensions are
    processed simultaneously as a DIM_BLOCK-wide vector (the same layout as
    the ``sobol_draw_kernel_simple``/``sobol_engine_draw_kernel`` kernels
    that are known to be correct on TritonXPU).
    """
    dims = tl.arange(0, DIM_BLOCK)
    dim_mask = dims < dimension

    # Load initial quasi values for this dimension block
    quasi_vals = tl.load(quasi_ptr + dims, mask=dim_mask, other=0).to(tl.int64)

    # Fast-forward by n steps in closed form.  For each bit b in [0, MAXBIT),
    # count the indices x in (N, N+n] whose trailing-zero count equals b:
    #   count_b = floor((N+n)/2^b) - floor(N/2^b)
    #           - floor((N+n)/2^{b+1}) + floor(N/2^{b+1})
    # XOR-in dir[., b] when the count is odd.
    N = num_generated.to(tl.int64)
    end = N + n

    if dimension == DIM_BLOCK:
        load_dims = dims
    else:
        # Tail lanes must not read past sobolstate: the direction loads below
        # are unconditional (masked loads with a runtime mask are broken on
        # this backend, see the module docstring), so clamp their addresses.
        load_dims = tl.minimum(dims, dimension - 1)

    for b in tl.static_range(MAXBIT):
        cnt = (end >> b) - (N >> b) - (end >> (b + 1)) + (N >> (b + 1))
        # Apply the odd-parity scalar arithmetically (all-ones to keep the
        # direction, 0 to drop it) instead of through a load predicate: a
        # ``tl.load`` mask that is not compile-time constant is ignored by
        # the CI compiler (see the module docstring).
        keep = 0 - ((cnt & 1) != 0).to(tl.int64)
        direction_vals = tl.load(sobolstate_ptr + load_dims * MAXBIT + b).to(tl.int64)
        # XOR with quasi value
        quasi_vals = quasi_vals ^ (direction_vals & keep)

    # Store result.  For a full power-of-2 block every lane is in bounds, so
    # the masked-memory path (about 2x cost on this backend) can be skipped.
    if dimension == DIM_BLOCK:
        tl.store(quasi_ptr + dims, quasi_vals)
    else:
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
