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

"""XPU override for ``aten::_foreach_round_`` (in-place round over Tensor[]).

Three fixes over the generic executor:

1. **Unmasked fast path.**  The shared ``foreach_unary`` executor's masked
   load/store blocks the block-DMA lowering (KERNEL_OPT_EXPERIENCE.md #36), so a
   homogeneous float list is processed by a single unmasked 2D-grid kernel.

2. **Direct pointer arguments.**  Passing the tensor pointers as scalar kernel
   arguments keeps block-DMA on the load/store (see ``_foreach_direct``).

3. **Extern-free round.**  ``round`` (round half to even) promotes to fp32 and
   uses the magic-number construction in ``_round_half_to_even`` instead of a
   libdevice extern.  The generic executor's ``round_half_to_even_impl`` needs
   two ``tl.floor`` calls, and ``tl_extra_shim.rint`` is a soft extern; both are
   slow, and on flagtree the ``rint`` extern cannot be legalized inside the
   unmasked store (``triton_xpu.unpack reached lowering without a bufPtr``).
   The add/sub construction uses only scalar arithmetic and is exact over the
   whole finite range.

Everything else -- mixed dtypes/shapes, gappy/overlapping views, non-divisible
sizes, integral inputs (which ``round`` leaves unchanged), and the rejected bool
dtype -- falls back to ``foreach_unary`` with ``inplace=True``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import check_tensor_list, foreach_unary, same_dtype

from ._foreach_direct import _launch_fast, _select_ptr

logger = logging.getLogger(__name__)

# round accepts every float and integral dtype but rejects bool and complex
# (matching the generic ``UNARY_OPS["round"].allowed``).
_NO_BOOL_NO_COMPLEX = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    + (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
)


@triton.jit
def _round_half_to_even(x):
    """Round an fp32 value to nearest, ties to even, without an extern call.

    ``libdevice.rint`` is a soft extern on this backend, and on flagtree its
    result cannot be legalized inside the unmasked store (``triton_xpu.unpack
    reached lowering without a bufPtr``).  The magic-number construction uses
    only scalar arithmetic and is exact for every finite fp32 value:

    * ``(x + 1.5*2**23) - 1.5*2**23`` is RNE for ``|x| <= 2**22``: the sum
      lands in [2**23, 2**24) where fp32 has ulp 1, so the add rounds to an
      integer with ties-to-even and the subtraction is exact.
    * ``(x + 2**22) - 2**22`` does the same for ``2**22 < |x| < 2**23``,
      where a value is a multiple of 0.5.
    * ``|x| >= 2**23`` is already integral in fp32 and is returned unchanged.

    The arithmetic runs on ``|x|`` and the sign is restored afterwards: adding
    a positive constant to a large negative value cancels exactly and would
    skip the rounding step.
    """
    ax = tl.abs(x)
    small = (ax + 12582912.0) - 12582912.0  # 1.5 * 2**23
    middle = (ax + 4194304.0) - 4194304.0  # 2**22
    r = tl.where(
        ax <= 4194304.0,
        small,
        tl.where(ax < 8388608.0, middle, ax),
    )
    return tl.where(x < 0.0, -r, r)


@triton.jit
def _round_fast(x):
    # Integral inputs are already rounded; only floats need the fp32 path.
    # The int guard avoids an fp32 round-trip that would corrupt large int64
    # values.
    if x.dtype.is_floating():
        return _round_half_to_even(x.to(tl.float32)).to(x.dtype)
    else:
        return x


@triton.jit
def _foreach_round_unmasked_kernel(
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
    """One unmasked launch over the tensor list (in-place)."""
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    ptr = _select_ptr(
        t, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15
    ).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + idx).to(tl.float32)
    tl.store(ptr + idx, _round_half_to_even(x).to(DT))


def _foreach_round_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ROUND_")
    tensors = check_tensor_list(self)
    if _launch_fast(tensors, _foreach_round_unmasked_kernel):
        # In-place schemas return ``()``; returning the list would make the
        # dispatcher reject the kernel.
        return None
    foreach_unary(
        tensors,
        _round_fast,
        complex_fn=None,
        out_dtype_fn=same_dtype,
        inplace=True,
        allowed_dtypes=_NO_BOOL_NO_COMPLEX,
    )
    return None
