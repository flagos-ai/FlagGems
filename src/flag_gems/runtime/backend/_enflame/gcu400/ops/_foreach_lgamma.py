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

"""GCU400-specific ``_foreach_lgamma``: per-tensor 1D flat kernel + pure Triton Lanczos.

The lgamma row of the common ``foreach_unary`` executor falls through to the GCU libdevice ``lgamma`` extern.
That routine is large (series + reflection + local memory operations) and under the shared executor's
large-shape launch (16x(1024,1024), >=512 concurrent CTAs) triggers a device-level Sip exception -> KMD abort
(P16, see compiler_problems/P16_flat_reshape_longloop_sip_abort/sources/
flaggems_enflame_foreach_lgamma_extern_concurrency_sip.md); up to 128 CTAs works.
This uses a per-tensor 1D fixed grid with in-kernel striding (<= FIXED_GRID CTAs per launch),
and replaces the body with a reflected Lanczos approximation (only ``tl.log``/``tl.sin``, no extern),
with fp32 computation throughout. Integral / bool / non-contiguous inputs fall back to common ``apply_unary``
(the extern is concurrency-safe for small shapes, with identical semantics and exception behavior).
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Generated per round by the optimization driver script; compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 384
BLOCK_SIZE = 1024
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
REVERSE_LAUNCH_ORDER = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@triton.jit
def _lgamma_pos_lanczos(y):
    """``lgamma(y)`` for ``y >= 0.5``: Lanczos g=7/N=9 (body of the mvlgamma_ fallback).

    Coefficients match ``_fallback_lgamma`` in ``gcu400/ops/mvlgamma_.py``; called only for
    ``y >= 0.5``; ``y < 0.5`` is handled by the outer reflection formula.
    """
    z = y - 1.0
    a = 0.9999999999998099
    a += 676.5203681218851 / (z + 1.0)
    a += -1259.1392167224028 / (z + 2.0)
    a += 771.3234287776531 / (z + 3.0)
    a += -176.6150291621406 / (z + 4.0)
    a += 12.507343278686905 / (z + 5.0)
    a += -0.13857109526572012 / (z + 6.0)
    a += 9.984369578019572e-6 / (z + 7.0)
    a += 1.5056327351493116e-7 / (z + 8.0)
    t = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(a)


@triton.jit
def _lgamma_full(x):
    """Full-domain ``lgamma``: ``x < 0.5`` uses reflection, non-positive integer poles and +/-inf give +inf.

    ``lgamma(x) = log(pi/|sin(pi x)|) - lgamma(1-x)`` (x < 0.5, including negative values).
    Both branches share one Lanczos parameter: ``y = where(x < 0.5, 1-x, x)`` always lands in
    ``>= 0.5`` - the reflection branch needs ``lgamma(1-x)`` and the positive branch needs ``lgamma(x)``,
    so one evaluation serves both branches (avoiding evaluating both sides of where and paying for the
    9 Lanczos divisions twice). The argument of ``sin(pi*x)`` is reduced to ``[-0.5, 0.5]`` before multiplying
    by pi - direct multiplication makes the rounding error grow unboundedly with large ``|x|``. For
    ``|reduced| < 1e-7``, ``sin(z) == z`` to below fp32 precision, so the linear branch is exact (isomorphic
    to the common ``_approx_lgamma_func``). NaN fails all comparisons, takes the positive branch and propagates;
    0 and negative integers are poles (+inf), matching ATen's gammaln semantics.
    """
    small = x < 0.5
    y = tl.where(small, 1.0 - x, x)
    value = _lgamma_pos_lanczos(y)
    reduced = x - tl.floor(x + 0.5)
    z = 3.141592653589793 * reduced
    sin_pi = tl.where(tl.abs(reduced) < 1.0e-7, z, tl.sin(z))
    refl = tl.log(3.141592653589793 / tl.abs(sin_pi)) - value
    res = tl.where(small, refl, value)
    pole = (x <= 0.0) & (x == tl.floor(x))
    return tl.where(pole | (tl.abs(x) == float("inf")), float("inf"), res)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_lgamma_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """1D fixed grid with striding for elementwise lgamma (pure Triton body, no extern)."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = _lgamma_full(value)
        tl.store(out_ptr + offs, result, mask=mask)


def _can_use_flat_kernel(self):
    """Accept only non-empty lists of contiguous float tensors."""
    if len(self) == 0:
        return False
    try:
        for tensor in self:
            if tensor.dtype not in _FLOAT_DTYPES or tensor.dtype.is_complex:
                return False
            if not tensor.is_contiguous():
                return False
    except AttributeError:
        # Non-Tensor elements: return False so the caller falls back to the common apply_unary for matching exceptions.
        return False
    return True


def _run_flat_kernel(self, inplace):
    if FALLBACK_TO_PUBLIC or not _can_use_flat_kernel(self):
        return apply_unary("lgamma", self, inplace)
    tensors = list(self)
    if not tensors:
        return [] if not inplace else None

    # Reversing only changes launch order; out-of-place outputs must keep the same order as the original list.
    launch_indices = list(range(len(tensors)))
    if REVERSE_LAUNCH_ORDER:
        launch_indices.reverse()
    outputs = [None] * len(tensors) if not inplace else None
    device_context = (
        torch_device_fn.device(tensors[0].device)
        if USE_DEVICE_CONTEXT
        else nullcontext()
    )
    with device_context:
        for index in launch_indices:
            tensor = tensors[index]
            num_elems = tensor.numel()
            out_tensor = tensor if inplace else torch.empty_like(tensor)
            # Empty tensors have no elements, so skip the kernel launch; but the empty result must still be kept
            # in the output list to stay one-to-one with the input (foreach returns as many tensors as it receives).
            if num_elems == 0:
                if not inplace:
                    outputs[index] = out_tensor
                continue
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_lgamma_flat_kernel[grid](
                tensor,
                out_tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
            if not inplace:
                outputs[index] = out_tensor
    return None if inplace else outputs


def _foreach_lgamma(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA")
    return _run_flat_kernel(self, inplace=False)


def _foreach_lgamma_(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA_")
    return _run_flat_kernel(self, inplace=True)
