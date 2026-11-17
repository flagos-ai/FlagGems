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
"""Kunlunxin XPU3 override for ``aten::_sobol_engine_ff_``.

The generic implementation advances the state with a runtime loop
``for i in range(1, n + 1)`` whose body scans the trailing-zero bit of
``num_generated + i``.  On this backend that dynamic loop mis-lowers for
``n == 1`` (single-iteration loop): the 14 ``n == 1`` cases of
``tests/test_sobol_engine_ff_.py`` return wrong 64-bit integers while every
``n > 1`` case passes.

This override removes the runtime loop entirely.  Fast-forwarding by ``n``
steps XORs ``direction[ctz(s)]`` over ``s in (num_generated, num_generated+n]``;
by linearity the result is ``XOR_j direction[j]`` over every bit position ``j``
whose count of steps with lowest set bit ``j`` in that half-open interval is
odd.  The per-bit count is a closed form over ``MAXBIT`` (30) positions, so the
kernel is a fully static unroll of pure integer arithmetic on one 1-D tile and
never relies on the mis-lowered dynamic loop.

The generic source is left untouched; ``flag_gems.ops._sobol_engine_ff_`` (the
callable the test imports directly) is rebound to this override at import.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAXBIT = 30


@triton.jit
def _sobol_engine_ff_kernel(
    quasi_ptr,
    sobolstate_ptr,
    dimension,
    n,
    num_generated,
    quasi_stride,
    ss_stride0,
    ss_stride1,
    MAXBIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane_ok = offs < dimension

    # Load current quasi value.  ``row`` is clamped so every lane reads a valid
    # direction row even when the block overruns ``dimension``.
    quasi_val = tl.load(quasi_ptr + offs * quasi_stride, mask=lane_ok, other=0)
    row = tl.minimum(offs, dimension - 1)

    one64 = tl.full([1], 1, tl.int64)
    begin = num_generated * one64
    end = begin + n * one64

    for j in tl.static_range(MAXBIT):
        # c(m) = number of integers in [1, m] whose lowest set bit is j, i.e.
        # integers congruent to 2**j modulo 2**(j+1).
        cnt_end = tl.where(
            end > (1 << j) - 1,
            ((tl.maximum(end, 1 << j) - (1 << j)) >> (j + 1)) + 1,
            0,
        )
        cnt_begin = tl.where(
            begin > (1 << j) - 1,
            ((tl.maximum(begin, 1 << j) - (1 << j)) >> (j + 1)) + 1,
            0,
        )
        selected = (cnt_end - cnt_begin) & 1
        direction = tl.load(sobolstate_ptr + row * ss_stride0 + j * ss_stride1)
        quasi_val = quasi_val ^ tl.where(lane_ok & (selected != 0), direction, 0)

    tl.store(quasi_ptr + offs * quasi_stride, quasi_val, mask=lane_ok)


def _sobol_engine_ff_(quasi, n, sobolstate, dimension, num_generated):
    """Fast-forward the Sobol quasi-random generator state in place."""
    logger.debug("GEMS _SOBOL_ENGINE_FF_ (kunlunxin)")

    if quasi.numel() == 0 or dimension <= 0:
        return quasi

    maxbit = sobolstate.shape[1] if sobolstate.dim() > 1 else MAXBIT
    block = 64
    grid = (triton.cdiv(dimension, block),)

    _sobol_engine_ff_kernel[grid](
        quasi,
        sobolstate,
        dimension,
        n,
        num_generated,
        quasi.stride(0),
        sobolstate.stride(0),
        sobolstate.stride(1),
        MAXBIT=maxbit,
        BLOCK=block,
    )
    return quasi


def _patch_generic_namespace():
    """Rebind the generic callable to this backend-local override.

    ``tests/test_sobol_engine_ff_.py`` imports ``flag_gems.ops._sobol_engine_ff_``
    directly (a plain function, no SpecOpRegistrar), so patching the generic
    module and package attributes is what routes the test to this kernel.  The
    generic source stays untouched and other vendors keep the original kernel.
    """
    import sys

    try:
        gen_mod = sys.modules.get("flag_gems.ops._sobol_engine_ff_")
        if gen_mod is not None:
            gen_mod._sobol_engine_ff_ = _sobol_engine_ff_
        gen_pkg = sys.modules.get("flag_gems.ops")
        if gen_pkg is not None:
            gen_pkg._sobol_engine_ff_ = _sobol_engine_ff_
        root = sys.modules.get("flag_gems")
        if root is not None:
            root._sobol_engine_ff_ = _sobol_engine_ff_
    except Exception:  # pragma: no cover - best effort rebinding
        pass


_patch_generic_namespace()


__all__ = ["_sobol_engine_ff_kernel", "_sobol_engine_ff_"]
