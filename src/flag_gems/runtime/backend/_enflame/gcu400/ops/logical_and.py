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
import os

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# torch_gcu process flag that switches int64 storage to real 8-byte lanes.
_INT64_ENV = "TORCH_GCU_ENABLE_INT64_AND_UINT64"


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_and_func(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


def _fake_i64(t):
    # Without TORCH_GCU_ENABLE_INT64_AND_UINT64 torch_gcu stores int64
    # tensors as a compact N x int32 word payload while the metadata still
    # reports 8 bytes/element (P17 narrow-storage family, same as the
    # logical_not / bitwise_left_shift precedents). An i64-typed Triton
    # pointer then interleaves two words per lane and overruns the dense
    # payload on both loads and stores, so such operands must ride the
    # physical int32 lane instead. With the env flag set the storage is
    # real 8-byte int64 and the plain pointwise path below is correct.
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.int64
        and t.device.type in ("cuda", "gcu")
        and not os.environ.get(_INT64_ENV, "")
    )


def _i32_words(t):
    # Zero-copy same-shape int32 alias over the physical word slots; the
    # value-level view (native ATen, D2H) sees element i as the i-th word,
    # which is exactly what this alias hands to the kernel. Only safe for
    # offset-0 tensors (a nonzero storage_offset lands at byte 2k instead
    # of 4k); anything else falls back to the layout-aware ATen converting
    # copy, which reads the same word values.
    if t.storage_offset() == 0 and t.is_contiguous():
        alias = torch.empty(0, dtype=torch.int32, device=t.device)
        alias.set_(t.untyped_storage(), 0, t.shape, t.stride())
        return alias
    return t.to(torch.int32)


def logical_and(A, B):
    # With the global 64-bit switch always on, the int64 narrow-storage compatibility branch has no trigger condition
    # and converges to one kernel call.
    logger.debug("GEMS_ENFLAME LOGICAL_AND")
    return logical_and_func(A, B)


@triton.jit(do_not_specialize=["N"])
def _logical_and_inplace_kernel(A, B, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK)
    for tile_index in tl.range(pid, (N + BLOCK - 1) // BLOCK, num_programs):
        offset = tile_index * BLOCK + element_offsets
        mask = offset < N
        lhs = tl.load(A + offset, mask=mask, other=0)
        rhs = tl.load(B + offset, mask=mask, other=0)
        result = tl.where((lhs != 0) & (rhs != 0), 1, 0)
        tl.store(A + offset, result, mask=mask)


def logical_and_(A, B):
    from flag_gems.runtime import torch_device_fn

    logger.debug("GEMS_ENFLAME LOGICAL_AND_")
    if A.numel() == 0:
        return A
    # Broadcast to A's shape and expand into a contiguous tensor so the kernel can use 1D flat offsets.
    expanded_other = B.expand_as(A)
    if not expanded_other.is_contiguous():
        expanded_other = expanded_other.contiguous()
    target = A if A.is_contiguous() else A.contiguous()
    num_elements = target.numel()
    block_size = 8192
    grid_size = min(48, triton.cdiv(num_elements, block_size))
    with torch_device_fn.device(target.device):
        _logical_and_inplace_kernel[(grid_size,)](
            target,
            expanded_other,
            num_elements,
            BLOCK=block_size,
            num_warps=4,
        )
    if target is not A:
        A.copy_(target)
    return A
