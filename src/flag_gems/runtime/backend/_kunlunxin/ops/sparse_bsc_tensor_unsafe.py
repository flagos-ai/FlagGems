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

"""Kunlunxin (TritonXPU) specialization of ``aten::_sparse_bsc_tensor_unsafe``.

Why this file exists
--------------------
The generic wrapper converts the payload through ``_cast_kernel``, which is

    x = tl.load(src + offs, mask=mask, other=0.0)
    tl.store(dst + offs, x.to(...), mask=mask)

Masked loads *and* masked stores are not honoured on TritonXPU.  For a payload
smaller than one tile (e.g. the 4/16-element BSC test payloads) the inactive
lanes therefore write past ``dst`` — an out-of-bounds store (red-line hazard) —
and any requested dtype conversion goes through that path.

This backend version replaces the cast with a padded-destination, clamp-index
kernel (no masks, no ``other=``) and then delegates the (pure host-side)
container assembly to the generic wrapper *with the payload already at the
requested dtype*, so the generic cast is skipped and only the assembly runs.

float64 payloads
----------------
XPU3 is ``support_fp64=False``: every device factory (``empty``/``zeros``/
``tensor``/``empty_strided``) silently coerces ``float64`` to ``float32``.  A
float64 dtype *view* over float32 storage is preserved, however, and the Triton
compiler does lower ``fp32 -> fp64`` conversion and fp64 stores.  The cast path
therefore allocates ``2 * n`` float32 elements, views them as float64, and casts
into that view; ``fp32 -> fp64`` is exact so the result is bit-identical to
``values.to(torch.float64)`` (verified in ``/tmp/cast64.py``: 1000 random fp32
values, ``max abs diff = 0.0``, bit patterns equal).  This is a widening bit
copy, not fp64 arithmetic.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.sparse_bsc_tensor_unsafe import (
    sparse_bsc_tensor_unsafe as _generic_op,
)
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

_BLOCK = 1024


@libentry()
@triton.jit
def _cast_kernel(
    src_ptr: tl.tensor,
    dst_ptr: tl.tensor,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """Padded, unmasked elementwise cast into ``dst`` (tail writes to padding)."""
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < n
    safe = tl.where(live, offsets, 0)
    x = tl.load(src_ptr + safe)
    tl.store(dst_ptr + offsets, x.to(dst_ptr.dtype.element_ty))


def _cast_values(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast ``values`` to ``dtype`` with a padded Triton kernel."""
    n = values.numel()
    pad = triton.cdiv(n, _BLOCK) * _BLOCK
    if dtype is torch.float64:
        # `torch.empty(dtype=float64)` is coerced to float32 on XPU3; a dtype
        # view over 2*n float32 slots keeps float64 and is exactly n doubles.
        storage = torch.empty(2 * pad, dtype=torch.float32, device=values.device)
        dst = storage.view(torch.float64)
    else:
        dst = torch.empty(pad, dtype=dtype, device=values.device)
    if n:
        _cast_kernel[(pad // _BLOCK, 1, 1)](
            values, dst, n, BLOCK_SIZE=_BLOCK, num_warps=4
        )
    return dst[:n].view(values.shape)


def sparse_bsc_tensor_unsafe(
    ccol_indices,
    row_indices,
    values,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_KUNLUNXIN SPARSE_BSC_TENSOR_UNSAFE")
    if dtype is not None and values.dtype != dtype:
        values = _cast_values(values, dtype)
    # `values` already matches `dtype`, so the generic wrapper's own cast (the
    # masked one) is skipped; only its host-side assembly runs.
    return _generic_op(
        ccol_indices,
        row_indices,
        values,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
