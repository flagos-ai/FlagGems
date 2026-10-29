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

"""Kunlunxin (XPU) override for ``aten::_foreach_div.List`` / ``_foreach_div_.List``.

Pointers read back from a metadata table produce narrow accesses, so small
homogeneous lists are launched as one kernel whose pointers are all direct
arguments, while large tensors each get a direct-argument kernel.

The default ``x / y`` is slower than the round-to-nearest operation on this
backend, so fp32 uses the latter (bit-identical to ``x / y``) and
half-precision outputs use a reciprocal-square-root refinement, whose fp32
error is below one half-precision ulp; zero and infinity keep their IEEE
results.

Non-floating inputs are delegated to the generic executor for its INT_TO_FLOAT
promotion.
"""

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.ops._foreach_binary import _apply_list, _apply_tensor
from flag_gems.utils.foreach import (
    _dense,
    _finish_inplace,
    check_tensor_list,
    has_internal_overlap,
    tl_dtype,
)

logger = logging.getLogger(__name__)

UNROLL_NUM = 8
IS_CLOSE_MEMORY_ASYNC = False
LARGE_THRESHOLD = 1_048_576


_INF = tl.constexpr(float("inf"))


@triton.jit
def _nr_div(x, y):
    # 1/y from the reciprocal square root, refined once by Newton.
    ay = tl.abs(y)
    inv = xpu.rsqrt(ay)
    inv = inv * inv
    inv = inv * (2.0 - y * inv)
    # Refinement is undefined at 0 and inf; restore the IEEE results.
    inv = tl.where(ay == 0.0, tl.where(y < 0.0, -_INF, _INF), inv)
    inv = tl.where(ay == _INF, 0.0, inv)
    return x * inv


@triton.jit
def _fdiv(x, y, OT: tl.constexpr):
    """``x / y`` in fp32 through the fastest exact-enough sequence."""
    if OT == tl.float16:
        return _nr_div(x, y)
    if OT == tl.bfloat16:
        return x / y
    return xpu.div_rn(x, y)


@triton.jit
def _div_store_plain(
    a_ptr, b_ptr, o_ptr, idx, m, OT: tl.constexpr, NEED_MASK: tl.constexpr
):
    if NEED_MASK:
        a = tl.load(a_ptr + idx, mask=m, other=0).to(tl.float32)
        b = tl.load(b_ptr + idx, mask=m, other=1).to(tl.float32)
        tl.store(o_ptr + idx, (a / b).to(OT), mask=m)
    else:
        a = tl.load(a_ptr + idx).to(tl.float32)
        b = tl.load(b_ptr + idx).to(tl.float32)
        tl.store(o_ptr + idx, (a / b).to(OT))


@triton.jit
def _div_store(a_ptr, b_ptr, o_ptr, idx, m, OT: tl.constexpr, NEED_MASK: tl.constexpr):
    if NEED_MASK:
        a = tl.load(a_ptr + idx, mask=m, other=0).to(tl.float32)
        b = tl.load(b_ptr + idx, mask=m, other=1).to(tl.float32)
        tl.store(o_ptr + idx, _fdiv(a, b, OT).to(OT), mask=m)
    else:
        a = tl.load(a_ptr + idx).to(tl.float32)
        b = tl.load(b_ptr + idx).to(tl.float32)
        tl.store(o_ptr + idx, _fdiv(a, b, OT).to(OT))


def _pick_block(max_numel, elem_bytes):
    # Divide-bound, so block size barely matters. Cap tile at 65536: 131072
    # trips an f32->bf16 store miscompile.
    if max_numel >= 1_048_576:
        return (32768 if elem_bytes >= 4 else 65536), 8, 8192
    if max_numel >= 262_144:
        return 32768, 8, 8192
    if max_numel >= 16384:
        return 16384, 8, 4096
    return 2048, 4, 0


@triton.jit
def _foreach_div_direct_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Pointers passed as direct kernel args give wide contiguous accesses.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        _div_store(a_ptr, b_ptr, out_ptr, idx, idx < n_elements, OUT_DT, True)
    else:
        _div_store(a_ptr, b_ptr, out_ptr, idx, idx, OUT_DT, False)


@triton.jit
def _foreach_div_list_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # meta (int64): [0,NT) a ptrs | [NT,2NT) b ptrs | [2NT,3NT) out ptrs | [3NT,4NT) numels
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(B_DT))
    out_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        _div_store(a_ptr, b_ptr, out_ptr, idx, idx < n_elements, OUT_DT, True)
    else:
        _div_store(a_ptr, b_ptr, out_ptr, idx, idx, OUT_DT, False)


def _launch_direct(a, b, d, out_dt):
    n = a.numel()
    block, num_warps, bsl = _pick_block(n, d.element_size())
    need_mask = n % block != 0
    grid = (triton.cdiv(n, block),)
    kwargs = dict(
        num_warps=num_warps,
        unroll_num=UNROLL_NUM,
        isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
    )
    if bsl:
        kwargs["buffer_size_limit"] = bsl
    if out_dt == torch.bfloat16:
        kwargs["isCloseVectorization"] = True
    _foreach_div_direct_kernel[grid](
        a, b, d, n, tl_dtype(out_dt), block, need_mask, **kwargs
    )


def _launch_group(a_list, b_list, d_list, a_dt, b_dt, out_dt, device):
    numels = [t.numel() for t in a_list]
    max_numel = max(numels)
    if max_numel == 0:
        return

    homogeneous = all(n == max_numel for n in numels)

    if max_numel >= LARGE_THRESHOLD:
        for a, b, d in zip(a_list, b_list, d_list):
            if a.numel():
                _launch_direct(a, b, d, out_dt)
        return

    nt = len(a_list)
    block, num_warps, bsl = _pick_block(max_numel, d_list[0].element_size())
    need_mask = (not homogeneous) or (max_numel % block != 0)

    meta = torch.tensor(
        [t.data_ptr() for t in a_list]
        + [t.data_ptr() for t in b_list]
        + [t.data_ptr() for t in d_list]
        + numels,
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
    _foreach_div_list_kernel[grid](
        meta,
        nt,
        tl_dtype(a_dt),
        tl_dtype(b_dt),
        tl_dtype(out_dt),
        block,
        need_mask,
        **kwargs,
    )


def _all_float(tensors):
    return all(t.dtype.is_floating_point for t in tensors)


def _run(self, other, inplace):
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    results = []
    buckets = {}  # (device, a_dt, b_dt, out_dt) -> (a_list, b_list, d_list)
    writebacks = []

    for a, b in zip(self, other):
        out_dtype = torch.result_type(a, b)
        if not out_dtype.is_floating_point:  # div promotes integers to float32
            out_dtype = torch.float32
        src_a = _dense(a)
        src_b = _dense(b)
        if inplace:
            if out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone the tensor before performing the operation."
                )
            dst = src_a
            if dst is not a:
                writebacks.append((a, dst))
            results.append(a)
        else:
            dst = torch.empty_like(src_a, dtype=out_dtype)
            results.append(dst)
        key = (a.device, src_a.dtype, src_b.dtype, dst.dtype)
        bucket = buckets.setdefault(key, ([], [], []))
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(dst)

    for (device, a_dt, b_dt, out_dt), (a_list, b_list, d_list) in buckets.items():
        _launch_group(a_list, b_list, d_list, a_dt, b_dt, out_dt, device)

    _finish_inplace(results, writebacks)
    return results


def _foreach_div_List(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_DIV_LIST")
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    if not (_all_float(self) and _all_float(other)):
        return _apply_list("div", self, other, inplace=False)
    return _run(self, other, inplace=False)


def _foreach_div__List(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_DIV__LIST")
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    if not (_all_float(self) and _all_float(other)):
        return _apply_list("div", self, other, inplace=True)
    _run(self, other, inplace=True)
    return None


@triton.jit
def _foreach_div_tensor_meta_kernel(
    meta,
    scalar_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One launch: list tensors divided by a single device-resident scalar."""
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK
    a_ptr = tl.load(meta + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta + NT + t).to(tl.pointer_type(OUT_DT))
    n = tl.load(meta + 2 * NT + t)
    s = tl.load(scalar_ptr).to(tl.float32)
    idx = offset + tl.arange(0, BLOCK)
    if NEED_MASK:
        m = idx < n
        a = tl.load(a_ptr + idx, mask=m, other=0).to(tl.float32)
        tl.store(out_ptr + idx, (a / s).to(OUT_DT), mask=m)
    else:
        a = tl.load(a_ptr + idx).to(tl.float32)
        tl.store(out_ptr + idx, (a / s).to(OUT_DT))


def _run_scalar_tensor(self, other, inplace):
    """``.Tensor`` overload: one (0-dim) tensor broadcast across the list."""
    results = []
    writebacks = []
    buckets = {}  # (device, a_dt, out_dt) -> (a_list, d_list)
    for a in self:
        out_dtype = torch.result_type(a, other)
        if not out_dtype.is_floating_point:
            out_dtype = torch.float32
        src_a = _dense(a)
        if inplace:
            if out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src_a
            if dst is not a:
                writebacks.append((a, dst))
            results.append(a)
        else:
            dst = torch.empty_like(src_a, dtype=out_dtype)
            results.append(dst)
        key = (a.device, src_a.dtype, dst.dtype)
        bucket = buckets.setdefault(key, ([], []))
        bucket[0].append(src_a)
        bucket[1].append(dst)

    for (device, a_dt, out_dt), (a_list, d_list) in buckets.items():
        numels = [t.numel() for t in a_list]
        max_numel = max(numels)
        if max_numel == 0:
            continue
        nt = len(a_list)
        block, num_warps, bsl = _pick_block(max_numel, d_list[0].element_size())
        homogeneous = all(n == max_numel for n in numels)
        need_mask = (not homogeneous) or (max_numel % block != 0)
        meta = torch.tensor(
            [t.data_ptr() for t in a_list] + [t.data_ptr() for t in d_list] + numels,
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
        _foreach_div_tensor_meta_kernel[grid](
            meta,
            other,
            nt,
            tl_dtype(a_dt),
            tl_dtype(out_dt),
            block,
            need_mask,
            **kwargs,
        )
    _finish_inplace(results, writebacks)
    return results


def _foreach_div_Tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_DIV_TENSOR")
    self = check_tensor_list(self)
    if not _all_float(self):
        return _apply_tensor("div", self, other, inplace=False)
    if other is None or other.dim() != 0:
        return _apply_tensor("div", self, other, inplace=False)
    return _run_scalar_tensor(self, other, inplace=False)


def _foreach_div__Tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_DIV__TENSOR")
    self = check_tensor_list(self)
    if not _all_float(self):
        return _apply_tensor("div", self, other, inplace=True)
    if other is None or other.dim() != 0:
        return _apply_tensor("div", self, other, inplace=True)
    _run_scalar_tensor(self, other, inplace=True)
    return None
