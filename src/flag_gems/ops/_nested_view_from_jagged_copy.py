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

import torch
import triton
import triton.language as tl

from flag_gems.ops._nested_view_from_jagged import _nested_view_from_jagged

logger = logging.getLogger(__name__)


_BLOCK_MAX = 65536
_MAX_GRID = 65535


@triton.jit
def _copy_flat_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    """One BLOCK-wide tile per program: dst[i] = src[i]."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs))
    else:
        mask = offs < n
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Grid-stride streaming copy, only used when cdiv(n, BLOCK) > 65535."""
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    n_full = (n // BLOCK) * BLOCK
    for base in range(pid * BLOCK, n_full, step):
        offs = base + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs), cache_modifier=".cs")
    if pid == 0:
        if n_full < n:
            tail = n_full + tl.arange(0, BLOCK)
            mask = tail < n
            tl.store(dst_ptr + tail, tl.load(src_ptr + tail, mask=mask), mask=mask)


@triton.jit
def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _copy_values(values):
    """Allocate the owned destination and fill it with a Triton device copy."""
    n = values.numel()
    out = torch.empty_like(values)
    if n <= _BLOCK_MAX:
        # One program, no dispatch tail: cheapest possible launch.
        block = _next_pow2(n)
        grid = 1
        exact = block == n
    else:
        block = _BLOCK_MAX
        grid = (n + block - 1) // block
        exact = (n % block) == 0
    if grid <= _MAX_GRID:
        _copy_flat_kernel[(grid,)](
            values, out, n, BLOCK=block, EXACT=exact, num_warps=1, num_stages=1
        )
    else:
        _copy_kernel[(1024,)](values, out, n, BLOCK=block, num_warps=1, num_stages=4)
    return out


def _nested_view_from_jagged_copy(
    self: torch.Tensor,
    offsets: torch.Tensor,
    dummy: torch.Tensor,
    lengths=None,
    ragged_idx=1,
    min_seqlen=None,
    max_seqlen=None,
):
    """Constructs a jagged-layout nested tensor from a flat buffer `self`,
    splitting the packed jagged dimension into batch elements according to
    `offsets` / `lengths`. The `_copy` variant owns a fresh copy of the buffer,
    which is produced here by a Triton kernel before building the nested view.

    The copy is one flat BLOCK-wide tile per program, with the tile sized on the
    host from the element count: a launch costs a fixed term plus a per-program
    term on this backend, so a tensor that fits in one wide tile is copied by a
    single program. A mask-free store is used when the tile divides the count
    exactly. The payload bits are copied unchanged -- no cast, no arithmetic.
    """
    logger.debug("GEMS _NESTED_VIEW_FROM_JAGGED_COPY")

    src = self.contiguous() if not self.is_contiguous() else self
    values_copy = _copy_values(src) if src.numel() else torch.empty_like(src)

    return _nested_view_from_jagged(
        values_copy,
        offsets,
        dummy,
        lengths,
        ragged_idx,
        min_seqlen,
        max_seqlen,
    )
