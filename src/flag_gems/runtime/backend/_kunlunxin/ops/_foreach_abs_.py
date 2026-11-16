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

"""XPU override for ``aten::_foreach_abs_`` (in-place absolute value over Tensor[]).

Two fixes over the generic executor:

1. **Unmasked fast path.**  The shared ``foreach_unary`` executor masks every
   load/store with a runtime ``idx < n_elements`` guard, which blocks the
   block-DMA lowering even when the mask is always-true (KERNEL_OPT_EXPERIENCE.md
   #36).  A homogeneous float list is processed by a single unmasked 2D-grid
   kernel in one launch (see ``_foreach_direct``).

2. **Direct pointer arguments.**  The first unmasked overrides packed the tensor
   pointers into a device-side metadata buffer and reloaded them per program;
   for a memory-bound op like ``abs`` that runtime indirection halves the
   achievable bandwidth.  Here the pointers are passed as scalar kernel
   arguments instead (``_foreach_direct._select_ptr``), so the backend keeps
   block-DMA on ``tl.load``/``tl.store`` and ``abs`` runs at full bandwidth.

``abs`` preserves the input dtype for real inputs, so the kernel computes in the
native dtype with no float promotion.  Input and output coincide for the
in-place schema, so only one set of pointers is needed.

Everything else -- complex (rejected by the in-place schema anyway), mixed
dtypes/shapes, gappy/overlapping views, non-divisible sizes, integral inputs,
lists longer than ``MAX_DIRECT`` -- falls back to ``foreach_unary`` with
``inplace=True``.
"""

import logging

import triton
import triton.language as tl

from flag_gems.ops._foreach_complex_math import _hypot_fn
from flag_gems.utils.foreach import check_tensor_list, foreach_unary, real_dtype_of

from ._foreach_direct import _launch_fast, _select_ptr

logger = logging.getLogger(__name__)


@triton.jit
def _abs_fast(x):
    return tl.abs(x)


@triton.jit
def _abs_complex_func(re, im):
    # Modulus of a complex value; the executor passes real/imag separately.
    return _hypot_fn(re, im)


@triton.jit
def _foreach_abs_unmasked_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over the tensor list (in-place).

    The tensor pointers are passed directly as scalar arguments and selected by
    ``tl.program_id(0)``; the grid is ``(NT, n_elements // BLOCK)`` with no mask
    anywhere.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    ptr = _select_ptr(
        t, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15
    ).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + idx)
    tl.store(ptr + idx, tl.abs(x))


def _foreach_abs_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ABS_")
    tensors = check_tensor_list(self)
    if _launch_fast(tensors, _foreach_abs_unmasked_kernel):
        # In-place schemas return ``()``; returning the list would make the
        # dispatcher reject the kernel.
        return None
    foreach_unary(
        tensors,
        _abs_fast,
        complex_fn=_abs_complex_func,
        out_dtype_fn=real_dtype_of,
        inplace=True,
        allowed_dtypes=None,
    )
    return None
