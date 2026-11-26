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

"""XPU override for ``aten::_foreach_cos_`` (in-place cos over a TensorList).

A homogeneous list of at most sixteen contiguous, disjoint tensors is processed
by one fused launch: its signature carries sixteen direct data pointers plus a
constexpr ``K`` (the number of live pointers), and the grid's second dimension
selects the member.  Sixteen small tensors therefore cost one launch instead of
sixteen, which is the shape foreach benchmarks exercise.

Everything else keeps the shared ``_foreach_unary`` executor: promoting
integral inputs (which must be refused, like ATen), complex and float64 inputs,
non-contiguous views, lists longer than sixteen, internally overlapping views
such as ``expand``, and list members that share a storage (whose in-place
ordering is observable and must stay per-element).  That keeps the master
in-place contract -- refusals, view write-through, launch-count bound -- exactly
as the executor defines it.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.foreach import _STATS, check_tensor_list, has_internal_overlap
from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# The fused path handles the measured float dtypes only.  Integral and bool
# inputs must raise ATen's promotion error, complex keeps the complex kernel in
# the shared executor, and float64 has no device support on this backend.
_FAST_DTYPES = frozenset({torch.float16, torch.float32, torch.bfloat16})

# Members per fused launch; longer lists fall back so the executor's
# launch-count bound stays untouched.
_FUSED_MAX = 16


@triton.jit
def _cos_block(p, offs, n, NOMASK: tl.constexpr):
    """cos of one tensor's tile; no predication at all when the tile is full."""
    if NOMASK:
        x = tl.load(p + offs)
        tl.store(p + offs, tl.cos(x.to(tl.float32)).to(x.dtype))
    else:
        mask = offs < n
        x = tl.load(p + offs, mask=mask)
        tl.store(p + offs, tl.cos(x.to(tl.float32)).to(x.dtype), mask=mask)


@triton.jit
def _cos_fused(
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
    n,
    K: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = tl.multiple_of(pid * BLOCK, BLOCK)
    offs = tl.max_contiguous(base + tl.arange(0, BLOCK), BLOCK)
    if K > 0:
        _cos_block(p0, offs, n, NOMASK)
    if K > 1:
        _cos_block(p1, offs, n, NOMASK)
    if K > 2:
        _cos_block(p2, offs, n, NOMASK)
    if K > 3:
        _cos_block(p3, offs, n, NOMASK)
    if K > 4:
        _cos_block(p4, offs, n, NOMASK)
    if K > 5:
        _cos_block(p5, offs, n, NOMASK)
    if K > 6:
        _cos_block(p6, offs, n, NOMASK)
    if K > 7:
        _cos_block(p7, offs, n, NOMASK)
    if K > 8:
        _cos_block(p8, offs, n, NOMASK)
    if K > 9:
        _cos_block(p9, offs, n, NOMASK)
    if K > 10:
        _cos_block(p10, offs, n, NOMASK)
    if K > 11:
        _cos_block(p11, offs, n, NOMASK)
    if K > 12:
        _cos_block(p12, offs, n, NOMASK)
    if K > 13:
        _cos_block(p13, offs, n, NOMASK)
    if K > 14:
        _cos_block(p14, offs, n, NOMASK)
    if K > 15:
        _cos_block(p15, offs, n, NOMASK)


@triton.jit
def _cos_fused_split(
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
    n,
    K: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Same work as ``_cos_fused`` but one program per (tensor, tile) pair.

    Grid is (tiles, K), so each program touches a single buffer.  That raises
    the number of concurrent blocks available to hide transcendental latency.
    """
    pid = tl.program_id(0)
    t = tl.program_id(1)
    base = tl.multiple_of(pid * BLOCK, BLOCK)
    offs = tl.max_contiguous(base + tl.arange(0, BLOCK), BLOCK)
    if t == 0:
        _cos_block(p0, offs, n, NOMASK)
    elif t == 1:
        _cos_block(p1, offs, n, NOMASK)
    elif t == 2:
        _cos_block(p2, offs, n, NOMASK)
    elif t == 3:
        _cos_block(p3, offs, n, NOMASK)
    elif t == 4:
        _cos_block(p4, offs, n, NOMASK)
    elif t == 5:
        _cos_block(p5, offs, n, NOMASK)
    elif t == 6:
        _cos_block(p6, offs, n, NOMASK)
    elif t == 7:
        _cos_block(p7, offs, n, NOMASK)
    elif t == 8:
        _cos_block(p8, offs, n, NOMASK)
    elif t == 9:
        _cos_block(p9, offs, n, NOMASK)
    elif t == 10:
        _cos_block(p10, offs, n, NOMASK)
    elif t == 11:
        _cos_block(p11, offs, n, NOMASK)
    elif t == 12:
        _cos_block(p12, offs, n, NOMASK)
    elif t == 13:
        _cos_block(p13, offs, n, NOMASK)
    elif t == 14:
        _cos_block(p14, offs, n, NOMASK)
    elif t == 15:
        _cos_block(p15, offs, n, NOMASK)


def _geom(n):
    """Pinned block geometry."""
    if n >= 4096:
        return 8192, 8
    if n >= 1024:
        return 1024, 4
    return 256, 2


_SPLIT_MAX_TILES = 64


def _launch_fused(chunk):
    k = len(chunk)
    args = list(chunk)
    if k < _FUSED_MAX:
        args.extend([chunk[0]] * (_FUSED_MAX - k))
    n = chunk[0].numel()
    block, warps = _geom(n)
    wide = chunk[0].element_size() >= 4
    if n >= 262144 and not wide:
        # 16-bit tiles fit the on-chip tile budget at double BLOCK (32-bit does
        # not), which halves the CTA count of the split grid.
        block = block * 2
    elif n >= 65536 and chunk[0].dtype == torch.bfloat16:
        # bfloat16 prefers the fatter tile even at the smallest class.
        block = block * 2
        tiles = triton.cdiv(n, block)
        nomask = (n % block) == 0
        _cos_fused_split[(tiles, k)](*args, n, k, nomask, block, num_warps=warps)
        return
    tiles = triton.cdiv(n, block)
    nomask = (n % block) == 0
    if tiles <= _SPLIT_MAX_TILES:
        _cos_fused_split[(tiles, k)](*args, n, k, nomask, block, num_warps=warps)
    else:
        _cos_fused[(tiles,)](*args, n, k, nomask, block, num_warps=warps)


def _is_fusable(tensors):
    """Whether one fused launch reproduces the per-tensor in-place semantics.

    Requires a non-empty list of at most ``_FUSED_MAX`` contiguous float
    members with one dtype, one element count and one device, no internally
    overlapping view, and pairwise disjoint storages.  Anything else keeps the
    shared executor, which owns those contracts.
    """
    if not tensors or len(tensors) > _FUSED_MAX:
        return False
    first = tensors[0]
    dtype = first.dtype
    n = first.numel()
    device = first.device
    if dtype not in _FAST_DTYPES or n == 0 or not first.is_contiguous():
        return False
    storages = set()
    for t in tensors:
        if t.dtype != dtype or t.numel() != n or t.device != device:
            return False
        if not t.is_contiguous() or has_internal_overlap(t):
            return False
        storage = t.untyped_storage().data_ptr()
        if storage in storages:
            # Two members share a storage: in-place ordering is observable.
            return False
        storages.add(storage)
    return True


def _foreach_cos_(self):
    """In-place ``aten::_foreach_cos_`` over a TensorList."""
    logger.debug("GEMS _FOREACH_COS_")
    tensors = check_tensor_list(self)
    if _is_fusable(tensors):
        _STATS.reset()
        _launch_fused(tensors)
        _STATS.executor_launches += 1
        _STATS.groups += 1
        return None
    return apply_unary("cos", tensors, inplace=True)
