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

"""XPU override for ``aten::_foreach_sign_`` (in-place sign over Tensor[]).

Three fixes over the generic executor:

1. **Unmasked fast path.**  The shared ``foreach_unary`` executor's masked
   load/store blocks the block-DMA lowering (KERNEL_OPT_EXPERIENCE.md #36), so a
   homogeneous float list is processed by a single unmasked 2D-grid kernel.

2. **Direct pointer arguments.**  Passing the tensor pointers as scalar kernel
   arguments (rather than through a device-side metadata buffer) keeps block-DMA
   on the load/store, which is what lets a memory-bound op run at full bandwidth
   (see ``_foreach_direct``).

3. **Select-based math.**  ``sign`` maps ``x > 0`` to ``1`` and ``x < 0`` to
   ``-1`` (NaN stays ``0`` since neither comparison holds).  The generic
   ``(x > 0).to(DT) - (x < 0).to(DT)`` form casts an int1 predicate to fp16/bf16
   directly, which lowers to a pathologically slow sequence on this backend
   (~20x); computing in fp32 with ``tl.where`` over the two predicates and
   casting the fp32 result back is dramatically faster.

Everything else -- mixed dtypes/shapes, gappy/overlapping views, non-divisible
sizes, integral/bool inputs, and the rejected complex dtype -- falls back to
``foreach_unary`` with ``inplace=True``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import check_tensor_list, foreach_unary, same_dtype

from ._foreach_direct import _launch_fast, _select_ptr

logger = logging.getLogger(__name__)

# sign accepts every float and integral dtype plus bool, but rejects complex
# (matching the generic ``UNARY_OPS["sign"].allowed``).
_NO_COMPLEX = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    + (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    + (torch.bool,)
)


@triton.jit
def _sign_fast(x):
    # Mirrors ops/sign.py (and the generic ``_sign_func``): NaN yields 0 because
    # neither comparison holds, and int1 (bool) passes through unchanged.
    if x.dtype == tl.int1:
        return x
    else:
        return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@triton.jit
def _foreach_sign_unmasked_kernel(
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
    y = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
    tl.store(ptr + idx, y.to(DT))


def _foreach_sign_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_SIGN_")
    tensors = check_tensor_list(self)
    if _launch_fast(tensors, _foreach_sign_unmasked_kernel):
        # In-place schemas return ``()``; returning the list would make the
        # dispatcher reject the kernel.
        return None
    foreach_unary(
        tensors,
        _sign_fast,
        complex_fn=None,
        out_dtype_fn=same_dtype,
        inplace=True,
        allowed_dtypes=_NO_COMPLEX,
    )
    return None
