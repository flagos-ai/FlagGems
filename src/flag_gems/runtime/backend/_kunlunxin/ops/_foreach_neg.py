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

from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import check_tensor_list, foreach_unary, tl_dtype

from .neg import neg_, neg_func

logger = logging.getLogger(__name__)

# Fast-path split: contiguous uniform floating lists are either run through a
# single meta-pointer launch (small/medium) or per-tensor pointwise_dynamic
# (large).  The meta path trades vectorization for a single launch; the
# pointwise path trades launches for direct-pointer (>1.5TB/s) vectorization.
_META_BLOCK = 8192
_META_NUM_WARPS = 8
_POINTWISE_MIN_BYTES = 64 * 1024 * 1024  # total input bytes


@triton.jit
def _foreach_neg_meta_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    n = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(in_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, -x, mask=mask)


def _is_fast_path(tensors):
    if not tensors:
        return False
    first = tensors[0]
    if not first.dtype.is_floating_point:
        return False
    for t in tensors:
        if not t.dtype.is_floating_point:
            return False
        if t.dtype != first.dtype or t.device != first.device:
            return False
        if not t.is_contiguous():
            return False
    return True


def _meta_launch(tensors, inplace):
    nt = len(tensors)
    device = tensors[0].device
    if inplace:
        dsts = tensors
    else:
        dsts = [torch.empty_like(t) for t in tensors]
    numels = [t.numel() for t in tensors]
    max_numel = max(numels)
    if max_numel == 0:
        return None if inplace else dsts

    meta = torch.tensor(
        [t.data_ptr() for t in tensors] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, triton.cdiv(max_numel, _META_BLOCK))
    _foreach_neg_meta_kernel[grid](
        meta, nt, tl_dtype(tensors[0].dtype), _META_BLOCK, num_warps=_META_NUM_WARPS
    )
    return None if inplace else dsts


def _apply(tensors, inplace):
    tensors = check_tensor_list(tensors)
    if _is_fast_path(tensors):
        total_bytes = sum(t.numel() * t.element_size() for t in tensors)
        if total_bytes >= _POINTWISE_MIN_BYTES:
            if inplace:
                for t in tensors:
                    neg_(t)
                return None
            dsts = [torch.empty_like(t) for t in tensors]
            for t, d in zip(tensors, dsts):
                neg_func(t, out0=d)
            return dsts
        return _meta_launch(tensors, inplace)

    op = UNARY_OPS["neg"]
    if inplace:
        foreach_unary(
            tensors,
            op.fn,
            complex_fn=op.complex_fn,
            out_dtype_fn=op.out_dtype_fn,
            inplace=True,
            allowed_dtypes=op.allowed,
        )
        return None
    return foreach_unary(
        tensors,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )


def _foreach_neg(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_NEG")
    return _apply(self, inplace=False)


def _foreach_neg_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_NEG_")
    return _apply(self, inplace=True)
