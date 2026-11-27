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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .neg import neg

logger = logging.getLogger(__name__)

NUM_SIPS = 24
PAIR_BLOCK = 8192
PAIR_NUM_WARPS = 8


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def resolve_neg_pair_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    HIGH_HALF: tl.constexpr,
    FULL_PAIRS: tl.constexpr,
):
    # Materialize a flat stride-2 neg view in a single pass: element i lives at
    # x_ptr + 2*i (floats). Pairs of consecutive floats are loaded as one int64
    # vector (contiguous traffic), the raw element is extracted from the
    # HIGH_HALF or low 32 bits, negated, and stored contiguously.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    if FULL_PAIRS:
        n_pairs = N_total
    else:
        n_pairs = N_total - 1
    num_blocks = (n_pairs + BLOCK - 1) // BLOCK
    x64_ptr = x_ptr.to(tl.pointer_type(tl.int64))
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < n_pairs
        v = tl.load(x64_ptr + off, mask=mask)
        if HIGH_HALF:
            bits = (v >> 32).to(tl.int32)
        else:
            bits = v.to(tl.int32)
        raw = bits.to(tl.float32, bitcast=True)
        tl.store(out_ptr + off, -raw, mask=mask)
    if (not FULL_PAIRS) and pid == 0:
        if N_total > 0:
            last = tl.load(x_ptr + 2 * (N_total - 1))
            tl.store(out_ptr + (N_total - 1), -last)


def _flat_elem_stride(A):
    # Return the constant element stride s if flat logical index i maps to
    # storage offset off + s * i (ignoring size-1 dims), else None.
    shape = A.shape
    if A.ndim == 0:
        return 1
    strides = A.stride()
    s = None
    acc = 1
    for d in range(A.ndim - 1, -1, -1):
        if shape[d] == 1:
            continue
        if s is None:
            s = strides[d]
        if strides[d] != s * acc:
            return None
        acc *= shape[d]
    return s if s is not None else 1


def _resolve_neg_pair(A):
    N = A.numel()
    out = torch.empty_like(A, memory_format=torch.contiguous_format)
    if N == 0:
        return out
    off = A.storage_offset()
    total_elems = A.untyped_storage().nbytes() // A.element_size()
    if off & 1:
        # Window [off-1, off-1+2N) is always in-bounds and the raw element
        # lands in the high half of each int64 pair.
        src = torch.as_strided(A, (2 * N,), (1,), off - 1)
        high_half = True
        full_pairs = True
    else:
        # Pairs start exactly at A; the last pair's high half may exceed the
        # storage by one element when the storage is tight (tail scalar path).
        src = A
        high_half = False
        full_pairs = off + 2 * N <= total_elems
    n_pairs = N if full_pairs else N - 1
    grid = max(1, min(triton.cdiv(n_pairs, PAIR_BLOCK), NUM_SIPS * 2))
    with torch_device_fn.device(A.device):
        resolve_neg_pair_kernel[(grid,)](
            src,
            out,
            N,
            BLOCK=PAIR_BLOCK,
            HIGH_HALF=high_half,
            FULL_PAIRS=full_pairs,
            num_warps=PAIR_NUM_WARPS,
        )
    return out


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_ENFLAME RESOLVE_NEG")
    if not A.is_neg():
        return A
    N = A.numel()
    if (
        0 < N <= 2**30
        and A.element_size() == 4
        and _flat_elem_stride(A) == 2
        and A.untyped_storage().data_ptr() % 8 == 0
    ):
        return _resolve_neg_pair(A)
    return neg(A)
