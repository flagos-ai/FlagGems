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
#
# Kunlunxin (XPU) override of rrelu_.
#
# The generic training kernel groups four element tiles per program behind a
# single Philox call.  On the XPU triton fork that four-store form is compiled
# to a no-op whenever the tail tiles are fully masked off: a program that owns
# a tile whose later groups are entirely out of range writes nothing, so any
# tensor smaller than four tiles (4 * BLOCK elements) comes back unchanged.
# A single masked load/store per program -- the layout the eval kernel already
# uses and that works at every size -- does not have this failure, so the
# training kernel below keeps one element tile per program and draws one Philox
# value per element.  The math is the generic one: ``out = x`` for ``x >= 0``
# and ``x * u`` with ``u ~ U(lower, upper)`` otherwise.
#
# The eval path is the generic one, imported directly.
import logging

import triton
import triton.language as tl

from flag_gems.ops.rrelu_ import _rrelu_eval_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

TRAIN_BLOCK = 1024
TRAIN_NUM_WARPS = 4


@triton.jit(do_not_specialize=["lower", "upper", "philox_seed", "philox_offset"])
def _rrelu_train_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    lower,
    upper,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 += offsets
    zero = c0 * 0
    r0, _r1, _r2, _r3 = tl.philox(philox_seed, c0, c1, zero, zero)

    lower_f = lower.to(tl.float32)
    upper_f = upper.to(tl.float32)
    slope = lower_f + uint_to_uniform_float(r0) * (upper_f - lower_f)

    x_f = x.to(tl.float32)
    y = tl.where(x_f >= 0, x_f, x_f * slope)
    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


def _rrelu_train_(A, lower, upper, generator=None):
    n_elements = A.numel()
    if n_elements == 0:
        return A
    grid = (triton.cdiv(n_elements, TRAIN_BLOCK),)
    # One Philox counter per element here, so the generator advances by the
    # element count rather than by element-count / four.
    increment = n_elements
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(A.device):
        _rrelu_train_kernel[grid](
            A,
            A,
            n_elements,
            float(lower),
            float(upper),
            philox_seed,
            philox_offset,
            BLOCK=TRAIN_BLOCK,
            num_warps=TRAIN_NUM_WARPS,
        )
    return A


def rrelu_(
    A,
    lower=0.125,
    upper=0.33333333333333331,
    training=False,
    generator=None,
):
    logger.debug("GEMS_KUNLUNXIN RRELU_")
    if training:
        return _rrelu_train_(A, lower, upper, generator)
    return _rrelu_eval_(A, lower, upper)
