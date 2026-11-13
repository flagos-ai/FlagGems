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
    logger.debug("GEMS_ENFLAME LOGICAL_AND")
    if _fake_i64(A) or _fake_i64(B):
        A = _i32_words(A) if _fake_i64(A) else A
        B = _i32_words(B) if _fake_i64(B) else B
    return logical_and_func(A, B)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit()
def logical_and_func_(x, y):
    return tl.where((x != 0) & (y != 0), 1, 0)


def logical_and_(A, B):
    logger.debug("GEMS_ENFLAME LOGICAL_AND_")
    if _fake_i64(A) or _fake_i64(B):
        B = _i32_words(B) if _fake_i64(B) else B
        if _fake_i64(A):
            if A.storage_offset() == 0 and A.is_contiguous():
                # int32 alias over A's own storage: the 0/1 result words
                # land exactly where the value-level view reads element i.
                A32 = _i32_words(A)
                logical_and_func(A32, B, out0=A32)
                return A
            # Strided/offset fake-int64 destination: stage through a fresh
            # bool buffer and let the layout-aware ATen copy write back.
            A.copy_(logical_and_func(_i32_words(A), B))
            return A
        # Non-int64 destination with a fake-int64 operand: the store lane
        # follows the destination dtype, so writing directly is safe.
        logical_and_func(A, B, out0=A)
        return A
    logical_and_func_(A, B, out0=A)
    return A
