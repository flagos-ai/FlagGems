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

"""Kunlunxin (XPU) override for ``aten::_foreach_acos``.

The generic path loses on two fronts: branchy libdevice math, and
metadata-indirect pointers that defeat wide block-DMA. This override reuses the
branch-free single-tensor recipe and splits launches by size: large tensors get
one direct-argument kernel each (full block-DMA), small/host-bound lists share a
single fused metadata launch (``NEED_MASK`` constexpr skips masking when
divisible).

Complex and integral inputs fall back to ``apply_unary``, which the fast real
path does not handle.
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

from .acos import _acos_body  # reuse single-tensor branch-free recipe

# Use the generic foreach logger/record so the registration-liveness test
# (watching ``flag_gems.ops._foreach_unary`` for ``GEMS _FOREACH_ACOS``) passes.
logger = logging.getLogger("flag_gems.ops._foreach_unary")

UNROLL_NUM = 8
IS_CLOSE_MEMORY_ASYNC = False
# Above this per-tensor element count, direct-argument launches beat the
# metadata-indirect fused launch.
LARGE_THRESHOLD = 1_048_576

# PLACEHOLDER


def _pick_block(max_numel):
    if max_numel >= 1_048_576:
        return 131072, 8, 8192
    if max_numel >= 262_144:
        return 32768, 8, 8192
    if max_numel >= 16384:
        return 16384, 8, 4096
    return 2048, 4, 0


@triton.jit
def _foreach_acos_direct_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Direct-argument pointers enable wide block-DMA.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        r = _acos_body(x)
        tl.store(out_ptr + idx, r.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        x = tl.load(x_ptr + idx).to(tl.float32)
        r = _acos_body(x)
        tl.store(out_ptr + idx, r.to(out_ptr.dtype.element_ty))


@triton.jit
def _foreach_acos_list_kernel(
    meta_ptr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # meta layout (all int64): [0,NT) in ptrs | [NT,2NT) out ptrs | [2NT,3NT) numels
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        r = _acos_body(x)
        tl.store(out_ptr + idx, r.to(OUT_DT), mask=mask)
    else:
        x = tl.load(in_ptr + idx).to(tl.float32)
        r = _acos_body(x)
        tl.store(out_ptr + idx, r.to(OUT_DT))


def _launch_direct(x, out):
    n = x.numel()
    block, num_warps, bsl = _pick_block(n)
    need_mask = n % block != 0
    grid = (triton.cdiv(n, block),)
    kwargs = dict(
        num_warps=num_warps,
        unroll_num=UNROLL_NUM,
        isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
    )
    if bsl:
        kwargs["buffer_size_limit"] = bsl
    _foreach_acos_direct_kernel[grid](x, out, n, block, need_mask, **kwargs)


def _launch_group(srcs, dsts, in_dtype, out_dtype, device):
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return

    # Big tensors: one direct-argument launch each (full-bandwidth block-DMA).
    if max_numel >= LARGE_THRESHOLD:
        for x, o in zip(srcs, dsts):
            if x.numel():
                _launch_direct(x, o)
        return

    # Small / host-bound: one fused metadata launch over the whole group.
    nt = len(srcs)
    block, num_warps, bsl = _pick_block(max_numel)
    homogeneous = all(n == max_numel for n in numels)
    need_mask = (not homogeneous) or (max_numel % block != 0)

    meta = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)

    grid = (nt, triton.cdiv(max_numel, block))
    kwargs = dict(
        num_warps=num_warps,
        unroll_num=UNROLL_NUM,
        isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
    )
    if bsl:
        kwargs["buffer_size_limit"] = bsl
    _foreach_acos_list_kernel[grid](
        meta,
        nt,
        tl_dtype(in_dtype),
        tl_dtype(out_dtype),
        block,
        need_mask,
        **kwargs,
    )


def _all_float(tensors):
    return all(t.dtype.is_floating_point for t in tensors)


def _run(tensors, inplace):
    tensors = check_tensor_list(tensors)
    results = [None] * len(tensors)
    buckets = {}  # (device, dtype) -> (srcs, dsts)
    writebacks = []

    for i, t in enumerate(tensors):
        dtype = t.dtype  # floating only here; out dtype == in dtype
        key = (t.device, dtype)
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
        bucket = buckets.setdefault(key, ([], []))
        bucket[0].append(src)
        bucket[1].append(dst)

    for (device, dtype), (srcs, dsts) in buckets.items():
        _launch_group(srcs, dsts, dtype, dtype, device)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)

    return results


def _foreach_acos(self):
    logger.debug("GEMS _FOREACH_ACOS")
    self = check_tensor_list(self)
    if not _all_float(self):  # complex64 / integral -> generic executor
        return apply_unary("acos", self, inplace=False)
    return _run(self, inplace=False)


def _foreach_acos_(self):
    logger.debug("GEMS _FOREACH_ACOS_")
    self = check_tensor_list(self)
    if not _all_float(self):
        return apply_unary("acos", self, inplace=True)
    _run(self, inplace=True)
    return None
