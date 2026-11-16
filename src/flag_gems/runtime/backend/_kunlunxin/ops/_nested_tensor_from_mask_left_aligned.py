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

"""Kunlunxin (XPU) implementation of ``_nested_tensor_from_mask_left_aligned``.

The generic kernel is unsafe on XPU3 and must not be re-run at benchmark sizes
(it deterministically raises ``KL_XID_KERNEL_EXCEPTION``, status 719).  It
relies on three primitives that are unreliable here:

* masked load with ``other=0`` — a masked in-bounds lane may read garbage (#51);
* 2-D axis-1 ``tl.sum`` / ``tl.min`` — silently wrong (#12/#18);
* a custom ``tl.reduce(combine_fn=and)`` — ``(1,1)`` all-True still yields False.

This override is a pure per-row 1-D formulation with none of those:

* one program per mask row, tiles over the columns, **unmasked** in-bounds
  loads (address ``clamp``ed, never ``other``), tail lanes zeroed with
  ``tl.where`` on the *value*;
* no sum at all.  Left-alignment is decided by two extrema:
  ``last_true = max{col : mask[col]}`` and
  ``first_false = min{col : !mask[col]}``; the row is aligned iff
  ``last_true < first_false`` (all-False: -1 < 0; all-True: L-1 < L;
  a False→True gap makes ``last_true >= first_false``).  This is correct by
  definition and needs no counting;
* a second single-program kernel AND-reduces the per-row verdicts with
  ``tl.min`` over 1-D tiles of 1024 lanes (>= 64 lanes per #51's safe range),
  using ``clamp``ed indices so a ``torch.empty`` buffer needs no identity fill.

``last_true``/``first_false`` are folded with scalar ``tl.maximum`` /
``tl.minimum`` and 1-D ``tl.max``/``tl.min``; no 2-D reduction and no custom
combine function is used anywhere.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _left_aligned_row_kernel(
    mask_ptr,
    mid_ptr,
    L,
    row_stride,
    col_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * row_stride

    last_true = tl.full((), -1, tl.int32)
    first_false = tl.full((), L, tl.int32)

    for off in range(0, L, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        valid = cols < L
        # In-bounds address for every lane; the out-of-range lanes are then
        # neutralised on the value (never via a masked-load `other`).
        addr = base + tl.minimum(cols, L - 1).to(tl.int64) * col_stride
        v = tl.where(valid, tl.load(mask_ptr + addr).to(tl.int32), 0)

        last_true = tl.maximum(
            last_true, tl.max(tl.where(v == 1, cols, -1), axis=0)
        )
        first_false = tl.minimum(
            first_false, tl.min(tl.where(v == 0, cols, L), axis=0)
        )

    tl.store(mid_ptr + row, tl.where(last_true < first_false, 1, 0))


@triton.jit
def _left_aligned_all_kernel(mid_ptr, out_ptr, rows, BLOCK: tl.constexpr):
    acc = tl.full((), 1, tl.int32)
    for off in range(0, rows, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        # Duplicating the last row keeps the load in bounds (address clamp) and
        # cannot change a minimum.
        idx_safe = tl.minimum(idx, rows - 1)
        acc = tl.minimum(acc, tl.min(tl.load(mid_ptr + idx_safe), axis=0))
    tl.store(out_ptr, acc)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _nested_tensor_from_mask_left_aligned(t, mask):
    logger.debug("GEMS_KUNLUNXIN _NESTED_TENSOR_FROM_MASK_LEFT_ALIGNED")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"Expected mask to be of ScalarType Bool, but got {mask.dtype} instead."
        )
    if mask.dim() != 2:
        raise RuntimeError("Padding mask should be 2D")
    if t.dim() != 3:
        raise RuntimeError("Input should be a 3D tensor, N * L * D")

    N, L = t.size(0), t.size(1)
    NN, LL = mask.size(0), mask.size(1)
    if N != NN or L != LL:
        raise RuntimeError("Mask size should match input size")

    if N == 0 or L == 0:
        return True

    # Strided access removes the need for a host-side `.contiguous()` copy.
    row_stride, col_stride = mask.stride()

    BLOCK = min(1024, max(64, _next_pow2(int(L))))
    BLOCK_ALL = 1024

    mid = torch.empty((N,), dtype=torch.int32, device=mask.device)
    out = torch.empty((), dtype=torch.int32, device=mask.device)

    with torch_device_fn.device(mask.device):
        _left_aligned_row_kernel[(N,)](
            mask, mid, L, row_stride, col_stride, BLOCK=BLOCK
        )
        _left_aligned_all_kernel[(1,)](
            mid, out, N, BLOCK=BLOCK_ALL
        )

    return bool(out.item())
