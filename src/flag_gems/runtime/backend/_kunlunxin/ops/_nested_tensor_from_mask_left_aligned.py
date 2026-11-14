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
#
# Kunlunxin override of flag_gems._nested_tensor_from_mask_left_aligned.
#
# Why the generic (public) implementation fails on XPU
# ----------------------------------------------------
# The generic kernel processes a whole (BLOCK_M=32) x (BLOCK_N=128) 2D tile in
# one shot:
#
#     rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
#     ptrs = mask + rows[:, None] * L + cols[None, :]
#     vals = tl.load(ptrs, mask=(rows[:, None] < N) & (cols[None, :] < L), other=0)
#     ... tl.sum(vals, axis=1) / tl.min(candidate, axis=1) ...
#
# On the Kunlunxin (P800/XPU3) Triton backend this 2D masked block (a) exceeds
# the uni_sram budget (OutOfResources: uni_sram, reproduced as soon as two
# extra 32-lane result stores are added) and (b) returns garbage for rows whose
# whole tile row is masked (rows >= N): tl.min/tl.sum over a fully-masked
# axis-1 row yield uninitialized values instead of the `other` fill, so the
# final `tl.where(rows < N, ..., 1)` cannot repair them and `tl.reduce(..., and)`
# is polluted -> the op is wrong even for N=1 (returns False for a left-aligned
# [True] row). A variant that wraps the same 1-D blocks in a per-row `if r < N`
# (scf.if) additionally fails to legalize `tt.reduce`
# ("failed to legalize operation 'tt.reduce' that was explicitly marked
# illegal" in ConvertTritonXPUToLLVM), so the row loop must be absent.
#
# Fix: one program per row (grid (N,)), a purely 1-D BLOCK_N-lane block, and
# single aligned load per tile. Per row, count True entries and the index of
# the first False entry; the row is aligned iff count == first_false (L when
# the row is all-True). Rows are AND-reduced with an int8 min in a second
# stage; masked lanes load `other=1` (the identity for min) so no per-row
# out-of-range repair is needed. All blocks are >= 64 lanes (XPU's minimum
# reliable width). The N==0 / L==0 shortcuts return True on the host.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _nested_tensor_from_mask_left_aligned_rows_kernel(
    mask,
    mid,
    L: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row left-alignment check; one program per row (grid (N,)).

    Stores an int8 flag (1 = aligned) into ``mid[n]``. The reduce lives at the
    kernel top level, not inside any scf.if/scf.for carrying the 1-D block.
    """
    n = ext.program_id(0)
    num_true = tl.zeros((BLOCK_N,), dtype=tl.int32)
    first_false = tl.full((BLOCK_N,), L, dtype=tl.int32)
    for off in range(0, L, BLOCK_N):
        idx = off + tl.arange(0, BLOCK_N)
        m = idx < L
        vals = tl.load(mask + n * L + idx, mask=m, other=0).to(tl.int32)
        num_true += vals
        first_false = tl.minimum(first_false, tl.where(vals == 0, idx, L))
    count = tl.sum(num_true, axis=0)
    ff = tl.min(first_false, axis=0)
    tl.store(mid + n, (count == ff).to(tl.int8))


@libentry()
@triton.jit
def _nested_tensor_from_mask_left_aligned_all_kernel(
    mid,
    out,
    MID_SIZE,
    BLOCK: tl.constexpr,
):
    """AND-reduce the per-row flags (int8 min) into a single output bool."""
    idx = tl.arange(0, BLOCK)
    m = idx < MID_SIZE
    vals = tl.load(mid + idx, mask=m, other=1).to(tl.int32)
    tl.store(out, tl.min(vals, axis=0) == 1)


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

    mask = mask.contiguous()
    BLOCK_N = 128
    # Pad to >= 64 lanes (XPU's minimum reliable block width); masked lanes
    # load `other=1`, the identity for the min-reduce.
    block_mid = max(64, triton.next_power_of_2(N))

    mid = torch.empty((N,), dtype=torch.int8, device=mask.device)
    out = torch.empty([], dtype=torch.bool, device=mask.device)

    with torch_device_fn.device(mask.device):
        _nested_tensor_from_mask_left_aligned_rows_kernel[(N,)](
            mask, mid, L, BLOCK_N=BLOCK_N
        )
        _nested_tensor_from_mask_left_aligned_all_kernel[(1, 1)](
            mid, out, N, BLOCK=block_mid
        )

    return bool(out.item())


__all__ = ["_nested_tensor_from_mask_left_aligned"]