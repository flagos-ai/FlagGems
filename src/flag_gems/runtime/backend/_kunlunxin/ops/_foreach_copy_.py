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

from flag_gems.ops._foreach_binary import BINARY_OPS, _copy_fn
from flag_gems.utils.foreach import (
    _check_same_length,
    check_tensor_list,
    foreach_binary_list,
    tl_dtype,
)

from .copy import _copy_flat_kernel, _pick_flat_block

logger = logging.getLogger(__name__)

_META_BLOCK = 8192
_META_NUM_WARPS = 8
_META_BYTES_LIMIT = (
    4 * 1024 * 1024
)  # total source bytes: below this, one meta launch beats 16 flat launches


@triton.jit
def _foreach_copy_meta_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    src_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    dst_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    n = tl.load(meta_ptr + 2 * NT + t)
    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(src_ptr + idx, mask=mask)
    tl.store(dst_ptr + idx, x, mask=mask)


def _is_fast_path(self, src):
    for dst, s in zip(self, src):
        if dst.dtype != s.dtype:
            return False
        if dst.device != s.device:
            return False
        if dst.shape != s.shape:
            return False
        if not dst.is_contiguous() or not s.is_contiguous():
            return False
    return True


def _meta_launch(self, src):
    nt = len(self)
    device = self[0].device
    numels = [t.numel() for t in self]
    max_numel = max(numels)
    meta = torch.tensor(
        [t.data_ptr() for t in src] + [t.data_ptr() for t in self] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, triton.cdiv(max_numel, _META_BLOCK))
    _foreach_copy_meta_kernel[grid](
        meta, nt, tl_dtype(self[0].dtype), _META_BLOCK, num_warps=_META_NUM_WARPS
    )


def _flat_copy(dst, src):
    n = src.numel()
    item_size = src.element_size()
    if (
        item_size <= 4
        and (n * item_size) % 4 == 0
        and n * item_size >= 2**20
        and src.data_ptr() % 4 == 0
        and dst.data_ptr() % 4 == 0
    ):
        src_view = src.reshape(-1).view(torch.int32)
        dst_view = dst.reshape(-1).view(torch.int32)
        n32 = src_view.numel()
        block = _pick_flat_block(n32)
        _copy_flat_kernel[(triton.cdiv(n32, block),)](
            src_view,
            dst_view,
            n32,
            BLOCK_SIZE=block,
            NEED_MASK=(n32 % block != 0),
            num_warps=32,
            unroll_num=8,
            buffer_size_limit=1024,
        )
    else:
        block = _pick_flat_block(n)
        _copy_flat_kernel[(triton.cdiv(n, block),)](
            src,
            dst,
            n,
            BLOCK_SIZE=block,
            NEED_MASK=(n % block != 0),
            num_warps=32,
            unroll_num=8,
            buffer_size_limit=1024,
        )


def _foreach_copy_(self, src, non_blocking=False):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_COPY_")
    self = check_tensor_list(self)
    src = check_tensor_list(src)
    _check_same_length(self, src, "tensor list")
    if _is_fast_path(self, src):
        total_bytes = sum(t.numel() * t.element_size() for t in src)
        if total_bytes <= _META_BYTES_LIMIT:
            _meta_launch(self, src)
        else:
            for dst, s in zip(self, src):
                _flat_copy(dst, s)
        return None
    foreach_binary_list(
        self,
        src,
        _copy_fn,
        inplace=True,
        allowed_dtypes=BINARY_OPS["copy"].allowed,
        copy_semantics=True,
    )
    return None
