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

"""XPU override for ``aten::_foreach_exp_`` (in-place exp over a TensorList).

The fused path is the generated candidate's kernel: a homogeneous list of at
most sixteen contiguous float tensors of one dtype and one element count is
processed by a single launch whose signature carries sixteen direct data
pointers; the 1-D program id selects the member through ``i = pid // BPT`` and
a branch chain.  16-bit dtypes additionally have a paired variant that drives
two tensors per program.

Everything else keeps the shared ``_foreach_unary`` executor, which owns the
in-place contract: integral/bool promotion refusals, complex and float64,
non-contiguous views (write-through with their own strides), ``expand``-style
internal overlap, and list members that share a storage.  That also avoids the
candidate's per-tensor fallback kernels, which are slower than the grouped
executor on this backend.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.foreach import check_tensor_list, has_internal_overlap
from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# Only the measured float dtypes use the fused kernel.  Integral and bool
# inputs must raise ATen's promotion error, complex keeps the shared executor,
# and float64 has no device support on this backend.
_FAST_DTYPES = frozenset({torch.float16, torch.float32, torch.bfloat16})

_FUSED_MAX = 16
_BLOCK_WIDE = 8192
_BLOCK_NARROW = 4096
_SMALL_TOTAL = 1 << 22
_WARPS = 8


@triton.jit
def _exp_body(p, offs):
    x = tl.load(p + offs).to(tl.float32)
    tl.store(p + offs, tl.exp(x).to(p.dtype.element_ty))


@triton.jit
def _exp_fused(
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
    BPT,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // BPT
    offs = (pid - i * BPT) * BLOCK + tl.arange(0, BLOCK)
    if i == 0:
        _exp_body(p0, offs)
    elif i == 1:
        _exp_body(p1, offs)
    elif i == 2:
        _exp_body(p2, offs)
    elif i == 3:
        _exp_body(p3, offs)
    elif i == 4:
        _exp_body(p4, offs)
    elif i == 5:
        _exp_body(p5, offs)
    elif i == 6:
        _exp_body(p6, offs)
    elif i == 7:
        _exp_body(p7, offs)
    elif i == 8:
        _exp_body(p8, offs)
    elif i == 9:
        _exp_body(p9, offs)
    elif i == 10:
        _exp_body(p10, offs)
    elif i == 11:
        _exp_body(p11, offs)
    elif i == 12:
        _exp_body(p12, offs)
    elif i == 13:
        _exp_body(p13, offs)
    elif i == 14:
        _exp_body(p14, offs)
    else:
        _exp_body(p15, offs)


@triton.jit
def _exp_fused_pair(
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
    BPT,
    ODD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // BPT
    offs = (pid - i * BPT) * BLOCK + tl.arange(0, BLOCK)
    if i == 0:
        _exp_body(p0, offs)
        _exp_body(p1, offs)
    elif i == 1:
        _exp_body(p2, offs)
        _exp_body(p3, offs)
    elif i == 2:
        _exp_body(p4, offs)
        _exp_body(p5, offs)
    elif i == 3:
        _exp_body(p6, offs)
        _exp_body(p7, offs)
    elif i == 4:
        _exp_body(p8, offs)
        _exp_body(p9, offs)
    elif i == 5:
        _exp_body(p10, offs)
        _exp_body(p11, offs)
    elif i == 6:
        _exp_body(p12, offs)
        _exp_body(p13, offs)
    else:
        _exp_body(p14, offs)
        if not ODD:
            _exp_body(p15, offs)


def _geometry(tensors):
    """Block width and whether the paired kernel is used, as measured."""
    n = tensors[0].numel()
    narrow = tensors[0].dtype.itemsize < 4
    block = _BLOCK_WIDE
    paired = False
    if narrow:
        if n * len(tensors) <= _SMALL_TOTAL:
            block = _BLOCK_NARROW
        else:
            paired = True
    return block, paired


def _is_fusable(tensors):
    """Whether one fused launch reproduces the per-tensor in-place semantics.

    Requires a non-empty list of at most ``_FUSED_MAX`` contiguous float
    members with one dtype, one element count and one device, a tile-aligned
    element count, no internally overlapping view, and pairwise disjoint
    storages.  Anything else keeps the shared executor, which owns those
    contracts.
    """
    if not tensors or len(tensors) > _FUSED_MAX:
        return False
    first = tensors[0]
    dtype = first.dtype
    n = first.numel()
    device = first.device
    if dtype not in _FAST_DTYPES or n == 0:
        return False
    block, _ = _geometry(tensors)
    if n % block:
        return False
    storages = set()
    for t in tensors:
        if t.dtype is not dtype or t.numel() != n or t.device != device:
            return False
        if not t.is_contiguous() or has_internal_overlap(t):
            return False
        storage = t.untyped_storage().data_ptr()
        if storage in storages:
            # Two members share a storage: in-place ordering is observable.
            return False
        storages.add(storage)
    return True


def _launch_fused(tensors):
    n = len(tensors)
    last = tensors[-1]
    args = (
        tensors[0],
        tensors[1] if n > 1 else last,
        tensors[2] if n > 2 else last,
        tensors[3] if n > 3 else last,
        tensors[4] if n > 4 else last,
        tensors[5] if n > 5 else last,
        tensors[6] if n > 6 else last,
        tensors[7] if n > 7 else last,
        tensors[8] if n > 8 else last,
        tensors[9] if n > 9 else last,
        tensors[10] if n > 10 else last,
        tensors[11] if n > 11 else last,
        tensors[12] if n > 12 else last,
        tensors[13] if n > 13 else last,
        tensors[14] if n > 14 else last,
        last,
    )
    block, paired = _geometry(tensors)
    bpt = tensors[0].numel() // block
    if paired:
        _exp_fused_pair[((bpt * ((n + 1) >> 1)),)](
            *args, bpt, ODD=(n & 1) == 1, BLOCK=block, num_warps=_WARPS
        )
    else:
        _exp_fused[(bpt * n,)](*args, bpt, BLOCK=block, num_warps=_WARPS)


def _foreach_exp_(self):
    """In-place ``aten::_foreach_exp_`` over a TensorList."""
    logger.debug("GEMS _FOREACH_EXP_")
    tensors = check_tensor_list(self)
    if _is_fusable(tensors):
        _launch_fused(tensors)
        return None
    return apply_unary("exp", tensors, inplace=True)
