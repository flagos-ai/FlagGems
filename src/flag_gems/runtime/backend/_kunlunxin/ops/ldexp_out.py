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

"""``aten::ldexp.out`` for Kunlunxin/XPU3.

Same root cause as the sibling overrides (``_kunlunxin/ops/ldexp.py`` and
``ldexp_.py``): the generic ``flag_gems.ops.ldexp`` routes every non-complex
call through ``tl_extra_shim.ldexp`` (libdevice ``ldexpf``), which has **no
implementation** on this FlagTree/XPU3 build, so the kernel fails to link
(``ld.lld: error: undefined symbol: Unsupported``).  Its float path also uses
``tl.exp2``, which is numerically wrong here.

This module keeps the generic ``.out`` host contract and writes the **caller's
``out`` buffer directly** instead of the generic ``compute-temp -> resize_ ->
copy_`` sequence:

* integral exponent, contiguous ``out`` whose layout matches the result and
  whose dtype equals the promoted result dtype -> ``_ldexp_flat_kernel`` is
  launched with ``o_ptr == out`` (same math / flags as the out-of-place path:
  two normal-range ``2**e`` factors, FTZ denormal rescale, zero-sign fix).
* everything else (broadcast / non-contiguous ``out`` / differing output dtype
  / float exponent / complex) -> the proven compute kernels from
  ``_kunlunxin/ops/ldexp.py`` produce a result, then the generic
  ``_copy_result`` writes it into ``out`` (real XPU ``copy_``; complex outputs
  are split into real/imag parts exactly like the generic implementation).

Host semantics mirror ``flag_gems.ops.ldexp._ldexp_impl``: ``_validate_out``
(device / dtype-castability / partial-overlap rejection, exact alias allowed),
``_result_device`` CPU-scalar materialisation, ``out.resize_`` only when the
shape differs, and ``result is out`` identity is preserved on every path.
"""

import importlib
import logging

import torch

from flag_gems.runtime import torch_device_fn

from .ldexp import (
    _FLAT_OPTS,
    _PLAN_CACHE,
    _can_use_flat,
    _complex_ldexp,
    _ldexp_flat_kernel,
    _ldexp_float_func,
    _ldexp_int_func,
    _pick_geometry,
    _run_flat,
)

logger = logging.getLogger(__name__)

_generic = importlib.import_module("flag_gems.ops.ldexp")

_FLAT_OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _run_flat_out(self, other, out):
    """Launch the flat kernel writing into the caller's ``out`` buffer."""
    n = self.numel()
    exact = self.dtype == torch.float32
    wide = other.dtype == torch.int64
    key = (
        torch_device_fn.current_device(),
        self.dtype,
        other.dtype,
        n,
        exact,
        wide,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        if len(_PLAN_CACHE) >= 256:
            _PLAN_CACHE.clear()
        plan = _pick_geometry(n)
        _PLAN_CACHE[key] = plan
    blk, grid, masked = plan
    # ``o_ptr`` is the caller's buffer; each lane loads before it stores, so an
    # exact ``out is self`` alias is safe.  ``out`` is contiguous and its numel
    # equals ``n`` (checked by the caller), so a flat index is valid.
    _ldexp_flat_kernel[(grid,)](
        self,
        other,
        out,
        n,
        BLOCK=blk,
        MASKED=masked,
        EXACT=exact,
        WIDE_EXP=wide,
        **_FLAT_OPTS,
    )


def _resize_out(out, shape):
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)


def ldexp_out(self, other, *, out):
    logger.debug("GEMS_KUNLUNXIN LDEXP_OUT")
    result_dtype = _generic._result_dtype(self, other)
    _generic._validate_out(self, other, out, result_dtype)
    device = _generic._result_device(self, other)
    # CPU zero-dimensional operands are host scalars; materialise them on the
    # device (allocation + fill only).
    if self.device != device:
        self = torch.tensor(self.item(), dtype=self.dtype, device=device)
    if other.device != device:
        other = torch.tensor(other.item(), dtype=other.dtype, device=device)

    if self.is_complex() or other.is_complex():
        result = _complex_ldexp(self, other, result_dtype)
        _resize_out(out, result.shape)
        return _generic._copy_result(out, result)

    if self.numel() == 0:
        _resize_out(out, torch.broadcast_shapes(self.shape, other.shape))
        return out

    integral_exponent = (
        self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    )

    # Direct-write fast path: same shape class as the benchmark matrices, where
    # ``out`` is contiguous and already has the result dtype/shape.  Writes the
    # caller's buffer in place -- no temporary allocation, no copy kernel.
    if (
        integral_exponent
        and out.is_contiguous()
        and out.dtype == result_dtype
        and out.dtype in _FLAT_OUT_DTYPES
        and out.shape == self.shape
        and _can_use_flat(self, other)
    ):
        _run_flat_out(self, other, out)
        return out

    # General path: compute the result with the proven kernels, then write it
    # into ``out`` (covers broadcast, non-contiguous out, dtype casting, float
    # exponent, and complex outputs).
    if integral_exponent:
        if _can_use_flat(self, other):
            result = _run_flat(self, other, result_dtype)
        else:
            result = _ldexp_int_func(self, other)
    else:
        result = _ldexp_float_func(self, other)
    _resize_out(out, result.shape)
    return _generic._copy_result(out, result)
