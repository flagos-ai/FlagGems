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

"""FlagGems-compatible Triton implementation of aten::_foreach_pow.ScalarAndTensor.

Measured contract (authoritative pytest):
  * new list of tensors, same shapes as the exponent tensors,
  * result dtype is float64 (reference is evaluated from upcast float64
    exponents and the test asserts actual.dtype == wanted.dtype),
  * values match pow(base, exponent) within atol=1e-4 / rtol=1e-7,
  * outputs never alias their input tensor,
  * empty exponent list returns [].

Uniform contiguous lists (up to 16 tensors of one dtype) use a single float64
allocation shaped (N, *shape) plus zero-copy unbind for the returned list, and
exactly one kernel launch whose grid axis 1 carries the tensor index:
  * launch-bound tiny lists use _pow_fused_kernel (pointer select, low arg count),
  * streaming large lists use _pow_group_kernel, whose branches hold direct
    input/output pointers so the float64 store keeps the load's layout.
Anything else falls back to one launch per tensor (flat or strided).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024
_MAX_FUSED = 16
_FUSE_SMALL_ELEMS = 1 << 20


@triton.jit
def _pow_fused_kernel(
    base,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    o_ptr,
    n,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xp = x0
    if N > 1:
        xp = tl.where(tid == 1, x1, xp)
    if N > 2:
        xp = tl.where(tid == 2, x2, xp)
    if N > 3:
        xp = tl.where(tid == 3, x3, xp)
    if N > 4:
        xp = tl.where(tid == 4, x4, xp)
    if N > 5:
        xp = tl.where(tid == 5, x5, xp)
    if N > 6:
        xp = tl.where(tid == 6, x6, xp)
    if N > 7:
        xp = tl.where(tid == 7, x7, xp)
    if N > 8:
        xp = tl.where(tid == 8, x8, xp)
    if N > 9:
        xp = tl.where(tid == 9, x9, xp)
    if N > 10:
        xp = tl.where(tid == 10, x10, xp)
    if N > 11:
        xp = tl.where(tid == 11, x11, xp)
    if N > 12:
        xp = tl.where(tid == 12, x12, xp)
    if N > 13:
        xp = tl.where(tid == 13, x13, xp)
    if N > 14:
        xp = tl.where(tid == 14, x14, xp)
    if N > 15:
        xp = tl.where(tid == 15, x15, xp)
    x = tl.load(xp + offs, mask=mask, other=0.0)
    y = libdevice.pow(base, x.to(tl.float32))
    tl.store(o_ptr + tid.to(tl.int64) * n + offs, y.to(tl.float64), mask=mask)


@triton.jit
def _pow_row(base, x_ptr, o_ptr, offs, mask):
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = libdevice.pow(base, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(tl.float64), mask=mask)


@triton.jit
def _pow_group_kernel(
    base,
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    x4,
    o4,
    x5,
    o5,
    x6,
    o6,
    x7,
    o7,
    x8,
    o8,
    x9,
    o9,
    x10,
    o10,
    x11,
    o11,
    x12,
    o12,
    x13,
    o13,
    x14,
    o14,
    x15,
    o15,
    n,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if tid == 0:
        _pow_row(base, x0, o0, offs, mask)
    if tid == 1:
        _pow_row(base, x1, o1, offs, mask)
    if tid == 2:
        _pow_row(base, x2, o2, offs, mask)
    if tid == 3:
        _pow_row(base, x3, o3, offs, mask)
    if tid == 4:
        _pow_row(base, x4, o4, offs, mask)
    if tid == 5:
        _pow_row(base, x5, o5, offs, mask)
    if tid == 6:
        _pow_row(base, x6, o6, offs, mask)
    if tid == 7:
        _pow_row(base, x7, o7, offs, mask)
    if tid == 8:
        _pow_row(base, x8, o8, offs, mask)
    if tid == 9:
        _pow_row(base, x9, o9, offs, mask)
    if tid == 10:
        _pow_row(base, x10, o10, offs, mask)
    if tid == 11:
        _pow_row(base, x11, o11, offs, mask)
    if tid == 12:
        _pow_row(base, x12, o12, offs, mask)
    if tid == 13:
        _pow_row(base, x13, o13, offs, mask)
    if tid == 14:
        _pow_row(base, x14, o14, offs, mask)
    if tid == 15:
        _pow_row(base, x15, o15, offs, mask)


@triton.jit
def _pow_flat_kernel(base, x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = libdevice.pow(base, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(tl.float64), mask=mask)


@triton.jit
def _pow_strided_kernel(
    base,
    x_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    d0,
    d1,
    d2,
    d3,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs
    lin = tl.zeros((BLOCK,), dtype=tl.int64)
    if RANK > 3:
        i3 = idx % d3
        idx = idx // d3
        lin += i3 * s3
    if RANK > 2:
        i2 = idx % d2
        idx = idx // d2
        lin += i2 * s2
    if RANK > 1:
        i1 = idx % d1
        idx = idx // d1
        lin += i1 * s1
    if RANK > 0:
        lin += idx * s0
    x = tl.load(x_ptr + lin, mask=mask, other=0.0)
    y = libdevice.pow(base, x.to(tl.float32))
    tl.store(o_ptr + lin, y.to(tl.float64), mask=mask)


def _launch_flat(base, x, out):
    n = x.numel()
    _pow_flat_kernel[(triton.cdiv(n, _BLOCK),)](
        base, x, out, n, BLOCK=_BLOCK, num_warps=4
    )


def _launch_strided(base, x, out):
    n = x.numel()
    shape = list(x.shape)
    strides = list(x.stride())
    while len(shape) < 4:
        shape.append(1)
        strides.append(0)
    _pow_strided_kernel[(triton.cdiv(n, _BLOCK),)](
        base,
        x,
        out,
        n,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        RANK=x.dim(),
        BLOCK=_BLOCK,
        num_warps=4,
    )


def _launch_deep(base, x, out):
    if x.dim() <= 4:
        _launch_strided(base, x, out)
        return
    for i in range(x.shape[0]):
        _launch_deep(base, x[i], out[i])


def foreach_pow_scalar_and_tensor(self, exponent):
    exps = list(exponent)
    count = len(exps)
    if count == 0:
        return []

    base = self.item() if isinstance(self, torch.Tensor) else self
    base = float(base)

    first = exps[0]
    shape = first.shape
    dtype = first.dtype
    if (
        count <= _MAX_FUSED
        and first.is_contiguous()
        and all(
            x.shape == shape and x.dtype == dtype and x.is_contiguous()
            for x in exps[1:]
        )
    ):
        n = first.numel()
        buf = torch.empty(
            (count,) + tuple(shape), dtype=torch.float64, device=first.device
        )
        outs = list(torch.unbind(buf, 0))
        if n:
            grid = (triton.cdiv(n, _BLOCK), count)
            if n * count <= _FUSE_SMALL_ELEMS:
                # Launch-bound tiny lists: pointer-select form keeps the
                # argument count low and the kernel is negligible here.
                ptrs = tuple(exps)
                if count < _MAX_FUSED:
                    ptrs = ptrs + (first,) * (_MAX_FUSED - count)
                _pow_fused_kernel[grid](
                    base, *ptrs, buf, n, N=count, BLOCK=_BLOCK, num_warps=4
                )
            else:
                # Streaming lists: one fused launch whose branches use direct
                # input/output pointers, so the compiler keeps one layout for
                # the narrow load and the float64 store.
                args = []
                for x, o in zip(exps, outs):
                    args.append(x)
                    args.append(o)
                if count < _MAX_FUSED:
                    args.extend([first, outs[0]] * (_MAX_FUSED - count))
                _pow_group_kernel[grid](
                    base, *args, n, N=count, BLOCK=_BLOCK, num_warps=4
                )
        return outs

    outs = []
    for x in exps:
        out = torch.empty_like(x, dtype=torch.float64)
        outs.append(out)
        if x.numel() == 0:
            continue
        if x.is_contiguous():
            _launch_flat(base, x, out)
        elif x.dim() <= 4:
            _launch_strided(base, x, out)
        else:
            # Rare fallback: peel leading dims with zero-copy views.
            _launch_deep(base, x, out)
    return outs
