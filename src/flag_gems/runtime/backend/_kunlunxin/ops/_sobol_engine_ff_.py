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

import logging
import sys

import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ``sobolstate`` is laid out as (dimension, MAXBIT); aten uses MAXBIT == 30.
# ``1 << j`` below must stay inside int64, hence the hard cap.
_MAX_SUPPORTED_BITS = 62


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

    quasi_val = tl.load(quasi_ptr + offs * quasi_stride, mask=lane_ok, other=0)

    # Strided gathers must stay unmasked on this backend (a masked strided
    # 1D gather can read out of bounds and poison the device context), so the
    # row index is clamped and the out-of-range lanes are discarded in
    # registers via ``tl.where`` below.
    row = tl.minimum(offs, dimension - 1)

    # ``one64`` forces int64 scalar arithmetic even when triton specialises
    # ``n`` / ``num_generated`` to python ints (the ``== 1`` specialisation).
    one64 = tl.full([1], 1, tl.int64)
    begin = num_generated * one64
    end = begin + n * one64

    for j in tl.static_range(MAXBIT):
        # count of k in [0, x) with k % 2**(j+1) == 2**j - 1
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
    logger.debug("GEMS_KUNLUNXIN _SOBOL_ENGINE_FF_")
    logging.getLogger("flag_gems.ops._sobol_engine_ff_").debug("GEMS _SOBOL_ENGINE_FF_")

    dimension = int(dimension)
    n = int(n)
    num_generated = int(num_generated)
    if dimension <= 0 or n <= 0 or quasi.numel() == 0:
        # Advancing by zero draws is the identity, matching aten.
        return quasi

    maxbit = min(sobolstate.shape[-1], _MAX_SUPPORTED_BITS)
    block = min(1024, triton.next_power_of_2(dimension))
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
        maxbit,
        block,
    )
    return quasi


def _patch_generic_namespace():
    for module_name in ("flag_gems.ops", "flag_gems.ops._sobol_engine_ff_"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "_sobol_engine_ff_", _sobol_engine_ff_)


_patch_generic_namespace()
