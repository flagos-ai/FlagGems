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

"""Fused foreach addcdiv_ (ATen `.Tensor` overload) for the MUSA target.

self[i] = self[i] + scalars[i] * tensor1[i] / tensor2[i]   (in place, per list entry)

Measured facts that drive the design (authoritative pytest + target probes):
  * ``scalars`` arrives as a CPU 1-D float32 tensor with one value per list entry
    (ATen rejects 0-dim/1-element/device-resident scalars and applies the values
    per entry), so the values are read once with ``tolist()`` and forwarded as
    kernel float arguments;
  * the benchmark lists hold 16 equally shaped tensors with equal scalars, so
    one launch with 50 arguments covers the whole list using the uniform-plan
    kernel (per-entry pointers, single size, single scalar);
  * the pytest lists are heterogeneous, so a general 16-slot kernel with
    per-entry sizes and per-entry scalars handles those (and any mix);
  * ATen accepts mixed dtypes and non-contiguous list tensors, so both are
    validated and routed to per-tensor fallback kernels rather than
    misinterpreted by the flat fused path;
  * the tiny-list benchmark is timed with wall clock around the call, so
    host-side dispatch dominates: small workloads re-launch a cached
    CompiledKernel and skip ``JITFunction.run``'s per-argument binding,
    specialization, pointer-alias and cache-key passes.

No allocation and no data movement: each entry is consumed as a flat contiguous
buffer and written back into its own storage, so storage identity is preserved.
Math is fp32 internal with a single rounding on store, matching ATen's
opmath_t = float functor for half/bfloat16 and exact for float32.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_addcdiv__Tensor as default_foreach_addcdiv__Tensor,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_MAX_T = 16
# Total elements below which host dispatch, not bandwidth, is the cost.
_TINY_TOTAL = 1 << 20

_STATE = {"fast": True}
_RUNNERS = {}

_IS_CONTIG = torch.Tensor.is_contiguous
_NUMEL = torch.Tensor.numel


@triton.jit
def _slot_body(A, B, C, n, s, local, BLOCK: tl.constexpr):
    offs = local * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(A + offs, mask=m, other=0.0)
    b = tl.load(B + offs, mask=m, other=0.0)
    c = tl.load(C + offs, mask=m, other=0.0)
    r = a.to(tl.float32) + (s * b.to(tl.float32)) / c.to(tl.float32)
    tl.store(A + offs, r.to(A.dtype.element_ty), mask=m)


@triton.jit
def _fused_u(
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    B10,
    B11,
    B12,
    B13,
    B14,
    B15,
    C0,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    C10,
    C11,
    C12,
    C13,
    C14,
    C15,
    n,
    s,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    e = tl.cdiv(n, BLOCK)
    if pid < e:
        _slot_body(A0, B0, C0, n, s, pid, BLOCK)
    elif pid < 2 * e:
        _slot_body(A1, B1, C1, n, s, pid - e, BLOCK)
    elif pid < 3 * e:
        _slot_body(A2, B2, C2, n, s, pid - 2 * e, BLOCK)
    elif pid < 4 * e:
        _slot_body(A3, B3, C3, n, s, pid - 3 * e, BLOCK)
    elif pid < 5 * e:
        _slot_body(A4, B4, C4, n, s, pid - 4 * e, BLOCK)
    elif pid < 6 * e:
        _slot_body(A5, B5, C5, n, s, pid - 5 * e, BLOCK)
    elif pid < 7 * e:
        _slot_body(A6, B6, C6, n, s, pid - 6 * e, BLOCK)
    elif pid < 8 * e:
        _slot_body(A7, B7, C7, n, s, pid - 7 * e, BLOCK)
    elif pid < 9 * e:
        _slot_body(A8, B8, C8, n, s, pid - 8 * e, BLOCK)
    elif pid < 10 * e:
        _slot_body(A9, B9, C9, n, s, pid - 9 * e, BLOCK)
    elif pid < 11 * e:
        _slot_body(A10, B10, C10, n, s, pid - 10 * e, BLOCK)
    elif pid < 12 * e:
        _slot_body(A11, B11, C11, n, s, pid - 11 * e, BLOCK)
    elif pid < 13 * e:
        _slot_body(A12, B12, C12, n, s, pid - 12 * e, BLOCK)
    elif pid < 14 * e:
        _slot_body(A13, B13, C13, n, s, pid - 13 * e, BLOCK)
    elif pid < 15 * e:
        _slot_body(A14, B14, C14, n, s, pid - 14 * e, BLOCK)
    else:
        _slot_body(A15, B15, C15, n, s, pid - 15 * e, BLOCK)


@triton.jit
def _fused(
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    B10,
    B11,
    B12,
    B13,
    B14,
    B15,
    C0,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    C10,
    C11,
    C12,
    C13,
    C14,
    C15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    e0 = tl.cdiv(n0, BLOCK)
    e1 = e0 + tl.cdiv(n1, BLOCK)
    e2 = e1 + tl.cdiv(n2, BLOCK)
    e3 = e2 + tl.cdiv(n3, BLOCK)
    e4 = e3 + tl.cdiv(n4, BLOCK)
    e5 = e4 + tl.cdiv(n5, BLOCK)
    e6 = e5 + tl.cdiv(n6, BLOCK)
    e7 = e6 + tl.cdiv(n7, BLOCK)
    e8 = e7 + tl.cdiv(n8, BLOCK)
    e9 = e8 + tl.cdiv(n9, BLOCK)
    e10 = e9 + tl.cdiv(n10, BLOCK)
    e11 = e10 + tl.cdiv(n11, BLOCK)
    e12 = e11 + tl.cdiv(n12, BLOCK)
    e13 = e12 + tl.cdiv(n13, BLOCK)
    e14 = e13 + tl.cdiv(n14, BLOCK)
    if pid < e0:
        _slot_body(A0, B0, C0, n0, s0, pid, BLOCK)
    elif pid < e1:
        _slot_body(A1, B1, C1, n1, s1, pid - e0, BLOCK)
    elif pid < e2:
        _slot_body(A2, B2, C2, n2, s2, pid - e1, BLOCK)
    elif pid < e3:
        _slot_body(A3, B3, C3, n3, s3, pid - e2, BLOCK)
    elif pid < e4:
        _slot_body(A4, B4, C4, n4, s4, pid - e3, BLOCK)
    elif pid < e5:
        _slot_body(A5, B5, C5, n5, s5, pid - e4, BLOCK)
    elif pid < e6:
        _slot_body(A6, B6, C6, n6, s6, pid - e5, BLOCK)
    elif pid < e7:
        _slot_body(A7, B7, C7, n7, s7, pid - e6, BLOCK)
    elif pid < e8:
        _slot_body(A8, B8, C8, n8, s8, pid - e7, BLOCK)
    elif pid < e9:
        _slot_body(A9, B9, C9, n9, s9, pid - e8, BLOCK)
    elif pid < e10:
        _slot_body(A10, B10, C10, n10, s10, pid - e9, BLOCK)
    elif pid < e11:
        _slot_body(A11, B11, C11, n11, s11, pid - e10, BLOCK)
    elif pid < e12:
        _slot_body(A12, B12, C12, n12, s12, pid - e11, BLOCK)
    elif pid < e13:
        _slot_body(A13, B13, C13, n13, s13, pid - e12, BLOCK)
    elif pid < e14:
        _slot_body(A14, B14, C14, n14, s14, pid - e13, BLOCK)
    else:
        _slot_body(A15, B15, C15, n15, s15, pid - e14, BLOCK)


@triton.jit
def _one(A, B, C, n, s, BLOCK: tl.constexpr):
    _slot_body(A, B, C, n, s, tl.program_id(0), BLOCK)


@triton.jit
def _one_strided(
    A,
    B,
    C,
    s,
    n,
    sa0,
    sa1,
    sa2,
    sa3,
    sb0,
    sb1,
    sb2,
    sb3,
    sc0,
    sc1,
    sc2,
    sc3,
    d0,
    d1,
    d2,
    d3,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    r = idx
    i3 = r % d3
    r = r // d3
    i2 = r % d2
    r = r // d2
    i1 = r % d1
    r = r // d1
    i0 = r
    ao = i0 * sa0 + i1 * sa1 + i2 * sa2 + i3 * sa3
    bo = i0 * sb0 + i1 * sb1 + i2 * sb2 + i3 * sb3
    co = i0 * sc0 + i1 * sc1 + i2 * sc2 + i3 * sc3
    a = tl.load(A + ao, mask=m, other=0.0)
    b = tl.load(B + bo, mask=m, other=0.0)
    c = tl.load(C + co, mask=m, other=0.0)
    out = a.to(tl.float32) + (s * b.to(tl.float32)) / c.to(tl.float32)
    tl.store(A + ao, out.to(A.dtype.element_ty), mask=m)


def _launch_one(a, b, c, s):
    n = _NUMEL(a)
    if n == 0:
        return
    if n >= 1024:
        block = 1024
    else:
        block = 16
        while block < n:
            block <<= 1
    nw = 1 if block <= 256 else 4
    grid = ((n + block - 1) // block,)
    if _IS_CONTIG(a) and _IS_CONTIG(b) and _IS_CONTIG(c):
        _one[grid](a, b, c, n, s, BLOCK=block, num_warps=nw)
        return
    if a.dim() > 4 or b.dim() > 4 or c.dim() > 4:
        raise NotImplementedError("non-contiguous foreach_addcdiv_: rank <= 4 only")
    shp = list(a.shape) + [1] * (4 - a.dim())
    sa = list(a.stride()) + [0] * (4 - a.dim())
    sb = list(b.stride()) + [0] * (4 - b.dim())
    sc = list(c.stride()) + [0] * (4 - c.dim())
    _one_strided[grid](
        a,
        b,
        c,
        s,
        n,
        sa[0],
        sa[1],
        sa[2],
        sa[3],
        sb[0],
        sb[1],
        sb[2],
        sb[3],
        sc[0],
        sc[1],
        sc[2],
        sc[3],
        shp[0],
        shp[1],
        shp[2],
        shp[3],
        BLOCK=block,
        num_warps=nw,
    )


def _dispatch(kernel, args, nb, block, nw, dt, dev, use_fast):
    grid = (nb, 1, 1)
    if use_fast:
        key = (kernel, dt, block, nw, nb, dev)
        runner = _RUNNERS.get(key)
        if runner is None:
            try:
                compiled = kernel.warmup(
                    *args, grid=(1, 1, 1), BLOCK=block, num_warps=nw
                )
                runner = compiled[grid]
                _RUNNERS[key] = runner
            except Exception:  # pragma: no cover - keep the validated path
                _STATE["fast"] = False
                runner = None
        if runner is not None:
            try:
                runner(*args, block)
                return
            except Exception:  # pragma: no cover - keep the validated path
                _STATE["fast"] = False
    kernel[grid](*args, BLOCK=block, num_warps=nw)


def _scalar_plan(scalars, n):
    """Return (s0, vals) with vals=None when every entry uses the same value."""
    if isinstance(scalars, torch.Tensor):
        m = _NUMEL(scalars)
        if m == 1:
            return float(scalars.item()), None
        if m == n and m > 1:
            vals = scalars.tolist()
            if not scalars.is_floating_point():
                vals = [float(v) for v in vals]
        else:
            flat = scalars.reshape(-1).tolist()
            if not flat:
                return 0.0, None
            last = float(flat[-1])
            vals = [float(flat[i]) if i < len(flat) else last for i in range(n)]
    elif isinstance(scalars, (int, float)):
        return float(scalars), None
    else:
        vals = [float(v) for v in scalars]
        if not vals:
            return 0.0, None
        if len(vals) != n:
            last = vals[-1]
            vals = [vals[i] if i < len(vals) else last for i in range(n)]
    if min(vals) == max(vals):
        return vals[0], None
    return vals[0], vals


def _launch_general(seq_a, seq_b, seq_c, seq_s, nums, block, nw, dt, dev, use_fast):
    n = len(nums)
    nb = 0
    for x in nums:
        nb += (x + block - 1) // block
    if nb == 0:
        return
    pad = _MAX_T - n
    if pad:
        dummy = seq_a[0]
        for t in seq_a:
            if _NUMEL(t):
                dummy = t
                break
        A = tuple(seq_a) + (dummy,) * pad
        B = tuple(seq_b) + (dummy,) * pad
        C = tuple(seq_c) + (dummy,) * pad
        nn = tuple(nums) + (0,) * pad
        ss = tuple(seq_s) + (0.0,) * pad
    else:
        A = tuple(seq_a)
        B = tuple(seq_b)
        C = tuple(seq_c)
        nn = tuple(nums)
        ss = tuple(seq_s)
    _dispatch(_fused, A + B + C + nn + ss, nb, block, nw, dt, dev, use_fast)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_addcdiv__Tensor(self, tensor1, tensor2, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_ADDCDIV__TENSOR")

    if not _list_supported(self):
        return default_foreach_addcdiv__Tensor(self, tensor1, tensor2, scalars)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_addcdiv__Tensor(self, tensor1, tensor2, scalars)


def _launch_foreach_addcdiv__Tensor(self, tensor1, tensor2, scalars):
    if not isinstance(self, (list, tuple)):
        self = [self]
    n = len(self)
    if n == 0:
        return None
    if not isinstance(tensor1, (list, tuple)):
        tensor1 = [tensor1]
    if not isinstance(tensor2, (list, tuple)):
        tensor2 = [tensor2]
    if len(tensor1) != n or len(tensor2) != n:
        raise ValueError("foreach_addcdiv_: tensor list length mismatch")

    dt = self[0].dtype
    nums = []
    append = nums.append
    uniform = True
    for a, b, c in zip(self, tensor1, tensor2):
        if (
            a.dtype is not dt
            or b.dtype is not dt
            or c.dtype is not dt
            or not _IS_CONTIG(a)
            or not _IS_CONTIG(b)
            or not _IS_CONTIG(c)
        ):
            uniform = False
            break
        append(_NUMEL(a))

    s0, vals = _scalar_plan(scalars, n)

    if not uniform:
        ss = vals if vals is not None else [s0] * n
        for i in range(n):
            _launch_one(self[i], tensor1[i], tensor2[i], ss[i])
        return None

    mn = min(nums)
    mx = max(nums)
    if mx == 0:
        return None
    if mx >= 1024:
        block = 1024
    else:
        block = 16
        while block < mx:
            block <<= 1
    nw = 1 if block <= 256 else 4
    dev = self[0].device
    use_fast = (
        _STATE["fast"]
        and n * mx <= _TINY_TOTAL
        and ((self[0].data_ptr() | tensor1[0].data_ptr() | tensor2[0].data_ptr()) & 15)
        == 0
    )

    if mn == mx and vals is None and n <= _MAX_T:
        # uniform plan: one size, one scalar, 50 launch arguments
        e = (mx + block - 1) // block
        nb = n * e
        pad = _MAX_T - n
        if pad:
            dummy = self[0]
            for t in self:
                if _NUMEL(t):
                    dummy = t
                    break
            A = tuple(self) + (dummy,) * pad
            B = tuple(tensor1) + (dummy,) * pad
            C = tuple(tensor2) + (dummy,) * pad
        else:
            A = tuple(self)
            B = tuple(tensor1)
            C = tuple(tensor2)
        _dispatch(_fused_u, A + B + C + (mx, s0), nb, block, nw, dt, dev, use_fast)
        return None

    ss = vals if vals is not None else [s0] * n
    if n <= _MAX_T:
        _launch_general(self, tensor1, tensor2, ss, nums, block, nw, dt, dev, use_fast)
    else:
        for i in range(0, n, _MAX_T):
            j = i + _MAX_T
            _launch_general(
                self[i:j],
                tensor1[i:j],
                tensor2[i:j],
                ss[i:j],
                nums[i:j],
                block,
                nw,
                dt,
                dev,
                use_fast,
            )
    return None
