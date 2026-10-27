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

"""Kunlunxin (XPU) override for ``aten::_foreach_log1p``.

``tl.log`` maps to a hardware SFU op, so the bottleneck is the launch layer, not
the math. log1p is bandwidth-bound. Dispatch:
  * large tensors (>= 1M): one direct-argument launch each (pointers as kernel
    args enable wide block-DMA), wide tile + buffer_size_limit.
  * small tensors: one fused metadata launch per (device, dtype) group to avoid
    the per-tensor launch floor.
  * complex / integer: delegate to the generic executor.

Float launches use isCloseVectorization=True: the open-vectorized log can
miscompile bf16 (elements off by +ln2) and is also slower here.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.utils.foreach import (
    check_tensor_list,
    has_internal_overlap,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

_LARGE = 1 << 20
_LARGE_BLOCK = 65536
_LARGE_WARPS = 32
_BSL = 8192
_UNROLL = 16
_SMALL_MAX_BLOCK = 16384


@triton.jit
def _log1p_direct_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + idx)
    out = tl.log(1.0 + x.to(tl.float32)).to(y_ptr.dtype.element_ty)
    if NEED_MASK:
        tl.store(y_ptr + idx, out, mask=mask)
    else:
        tl.store(y_ptr + idx, out)


@triton.jit
def _log1p_fused_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # meta (int64): [0,NT) in ptrs | [NT,2NT) out ptrs | [2NT,3NT) numels
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)
    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0.0)
    else:
        x = tl.load(in_ptr + idx)
    out = tl.log(1.0 + x.to(tl.float32)).to(DT)
    if NEED_MASK:
        tl.store(out_ptr + idx, out, mask=mask)
    else:
        tl.store(out_ptr + idx, out)


def _vec_kwargs(dtype):
    # Close vectorization for every float dtype (matches single-tensor
    # log1p.py): fixes bf16 +ln2 miscompile and the closed path is faster here.
    return {"isCloseVectorization": True}


def _launch_direct(src, dst, dtype):
    n = src.numel()
    block = _LARGE_BLOCK
    need_mask = n % block != 0
    grid = (triton.cdiv(n, block),)
    _log1p_direct_kernel[grid](
        src,
        dst,
        n,
        block,
        need_mask,
        num_warps=_LARGE_WARPS,
        buffer_size_limit=_BSL,
        unroll_num=_UNROLL,
        isCloseMemoryAsync=False,
        **_vec_kwargs(dtype),
    )


def _launch_small_group(srcs, dsts, dtype, device):
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    nt = len(srcs)
    block = min(_SMALL_MAX_BLOCK, triton.next_power_of_2(max_numel))
    num_warps = 4 if block <= 2048 else 8
    homogeneous = all(n == max_numel for n in numels)
    need_mask = (not homogeneous) or (max_numel % block != 0)
    meta_cpu = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    )
    meta = meta_cpu.to(device, non_blocking=True)
    grid = (nt, triton.cdiv(max_numel, block))
    _log1p_fused_kernel[grid](
        meta,
        nt,
        tl_dtype(dtype),
        block,
        need_mask,
        num_warps=num_warps,
        **_vec_kwargs(dtype),
    )


def _fast(tensors, inplace):
    results = [None] * len(tensors)
    small_buckets = {}  # (device, dtype) -> (srcs, dsts)
    writebacks = []

    for i, t in enumerate(tensors):
        dtype = t.dtype  # floating, non-complex (checked by caller)
        if inplace:
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if is_flat_addressable(t):
                src = dst = t
            else:
                src = dst = t.contiguous()
                writebacks.append((t, src))
            results[i] = t
        else:
            src = t if is_flat_addressable(t) else t.contiguous()
            dst = torch.empty_like(src)
            results[i] = dst

        if src.numel() >= _LARGE:
            _launch_direct(src, dst, dtype)
        else:
            bucket = small_buckets.setdefault((t.device, dtype), ([], []))
            bucket[0].append(src)
            bucket[1].append(dst)

    for (device, dtype), (srcs, dsts) in small_buckets.items():
        _launch_small_group(srcs, dsts, dtype, device)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)

    return results


def _foreach_log1p(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LOG1P")
    self = check_tensor_list(self)
    if all(t.is_floating_point() and not t.dtype.is_complex for t in self):
        return _fast(self, inplace=False)
    return apply_unary("log1p", self, inplace=False)


def _foreach_log1p_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_LOG1P_")
    self = check_tensor_list(self)
    if all(t.is_floating_point() and not t.dtype.is_complex for t in self):
        _fast(self, inplace=True)
        return None
    apply_unary("log1p", self, inplace=True)
    return None
