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
from triton.language.extra import libdevice as _libdevice
from triton.runtime import driver as _driver

logger = logging.getLogger(__name__)

# MODE selects the elementwise math form, mirroring ATen's own dispatch in
# pow_tensor_scalar_kernel (0.5 -> sqrt, -0.5 -> rsqrt, -1 -> reciprocal,
# 2 -> x*x, 3 -> x*x*x, otherwise generic powf on the fp32 opmath type).
_MODE_POW = 0
_MODE_SQRT = 1
_MODE_SQUARE = 2
_MODE_RECIP = 3
_MODE_RSQRT = 4
_MODE_CUBE = 5

# DT selects the storage dtype of the buffers addressed by the fused path.
_DT_F16 = 0
_DT_BF16 = 1
_DT_F32 = 2

_BLOCK = 2048
_NUM_WARPS = 4
# Above this many total elements a single fused multi-tensor launch buys
# nothing over one launch per tensor (device work dominates), so keep the
# simple per-tensor path there.
_FUSED_MAX_ELEMS = 1 << 20
_FUSED_SLOTS = 16


@triton.jit
def _math(xf, exp, MODE: tl.constexpr):
    if MODE == 0:
        y = _libdevice.pow(xf, exp)
    elif MODE == 1:
        y = tl.sqrt(xf)
    elif MODE == 2:
        y = xf * xf
    elif MODE == 3:
        y = 1.0 / xf
    elif MODE == 4:
        y = 1.0 / tl.sqrt(xf)
    else:
        y = xf * xf * xf
    return y


@triton.jit
def _slot_body(x_ptr, n, exp, pid, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = _math(x.to(tl.float32), exp, MODE)
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _pow_scalar_1d(x_ptr, n, exp, MODE: tl.constexpr, BLOCK: tl.constexpr):
    _slot_body(x_ptr, n, exp, tl.program_id(0), MODE, BLOCK)


# Uniform-shape fused path: the buffer addresses travel in one int64 device
# table, so a whole tensor list launches with only (table, n, exponent) as
# runtime arguments instead of one pointer and one size per tensor. Each
# program handles one tensor of the list, selected by program_id(1).
@triton.jit
def _pow_table(tbl, n, exp, DT: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    addr = tl.load(tbl + slot)
    if DT == 0:
        _slot_body(addr.to(tl.pointer_type(tl.float16)), n, exp, pid, MODE, BLOCK)
    elif DT == 1:
        _slot_body(addr.to(tl.pointer_type(tl.bfloat16)), n, exp, pid, MODE, BLOCK)
    else:
        _slot_body(addr.to(tl.pointer_type(tl.float32)), n, exp, pid, MODE, BLOCK)


@triton.jit
def _pow_multi(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
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
    exp,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    if slot == 0:
        _slot_body(a0, n0, exp, pid, MODE, BLOCK)
    elif slot == 1:
        _slot_body(a1, n1, exp, pid, MODE, BLOCK)
    elif slot == 2:
        _slot_body(a2, n2, exp, pid, MODE, BLOCK)
    elif slot == 3:
        _slot_body(a3, n3, exp, pid, MODE, BLOCK)
    elif slot == 4:
        _slot_body(a4, n4, exp, pid, MODE, BLOCK)
    elif slot == 5:
        _slot_body(a5, n5, exp, pid, MODE, BLOCK)
    elif slot == 6:
        _slot_body(a6, n6, exp, pid, MODE, BLOCK)
    elif slot == 7:
        _slot_body(a7, n7, exp, pid, MODE, BLOCK)
    elif slot == 8:
        _slot_body(a8, n8, exp, pid, MODE, BLOCK)
    elif slot == 9:
        _slot_body(a9, n9, exp, pid, MODE, BLOCK)
    elif slot == 10:
        _slot_body(a10, n10, exp, pid, MODE, BLOCK)
    elif slot == 11:
        _slot_body(a11, n11, exp, pid, MODE, BLOCK)
    elif slot == 12:
        _slot_body(a12, n12, exp, pid, MODE, BLOCK)
    elif slot == 13:
        _slot_body(a13, n13, exp, pid, MODE, BLOCK)
    elif slot == 14:
        _slot_body(a14, n14, exp, pid, MODE, BLOCK)
    else:
        _slot_body(a15, n15, exp, pid, MODE, BLOCK)


# Stride-aware path for non-contiguous inputs. Shape and strides are passed as
# runtime scalars (up to five logical dimensions), so no Python-level constant
# tuple indexing is needed.
@triton.jit
def _pow_scalar_strided(
    x_ptr,
    n,
    exp,
    s1,
    s2,
    s3,
    s4,
    t0,
    t1,
    t2,
    t3,
    t4,
    ND: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = offs.to(tl.int64)
    if ND == 1:
        off = p * t0
    elif ND == 2:
        off = (p % s1) * t1 + (p // s1) * t0
    elif ND == 3:
        off = (p % s2) * t2 + ((p // s2) % s1) * t1 + (p // (s1 * s2)) * t0
    elif ND == 4:
        off = (
            (p % s3) * t3
            + ((p // s3) % s2) * t2
            + ((p // (s2 * s3)) % s1) * t1
            + (p // (s1 * s2 * s3)) * t0
        )
    else:
        off = (
            (p % s4) * t4
            + ((p // s4) % s3) * t3
            + ((p // (s3 * s4)) % s2) * t2
            + ((p // (s2 * s3 * s4)) % s1) * t1
            + (p // (s1 * s2 * s3 * s4)) * t0
        )
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    y = _math(x.to(tl.float32), exp, MODE)
    tl.store(x_ptr + off, y.to(x.dtype), mask=mask)


# Caches hold compiled-kernel launchers and address tables, both keyed by every
# fact that determines a launch: device, dtype, math mode, exponent, shapes and
# the exact buffer addresses. Nothing kept here depends on tensor values.
_RUNNERS = {}
_TABLES = {}
_DIRECT_OK = True


def _mode_for(exp):
    if exp == 0.5:
        return _MODE_SQRT
    if exp == 2.0:
        return _MODE_SQUARE
    if exp == -1.0:
        return _MODE_RECIP
    if exp == -0.5:
        return _MODE_RSQRT
    if exp == 3.0:
        return _MODE_CUBE
    return _MODE_POW


def _dt_of(dtype):
    if dtype is torch.float16:
        return _DT_F16
    if dtype is torch.bfloat16:
        return _DT_BF16
    return _DT_F32


def _prepare(kernel, payload, grid3, mode):
    """Compile once and return the low-overhead launcher for that geometry."""
    global _DIRECT_OK
    if _DIRECT_OK:
        try:
            compiled = kernel.warmup(
                *payload,
                grid=grid3,
                MODE=mode,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
            return compiled[grid3]
        except Exception:
            _DIRECT_OK = False
    return None


def _launch_one(t, exp):
    numel = t.numel()
    grid = ((numel + _BLOCK - 1) // _BLOCK,)
    mode = _mode_for(exp)
    if t.is_contiguous():
        _pow_scalar_1d[grid](
            t, numel, exp, MODE=mode, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
        return
    shape = tuple(t.shape)
    strides = tuple(t.stride())
    nd = len(shape)
    if nd > 5:
        raise RuntimeError("foreach_pow_scalar_list_: only tensors up to 5-D supported")
    padded_s = list(shape) + [1] * (5 - nd)
    padded_t = list(strides) + [0] * (5 - nd)
    _pow_scalar_strided[grid](
        t,
        numel,
        exp,
        padded_s[1],
        padded_s[2],
        padded_s[3],
        padded_s[4],
        padded_t[0],
        padded_t[1],
        padded_t[2],
        padded_t[3],
        padded_t[4],
        ND=nd,
        MODE=mode,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_uniform(tensors, n, exp, mode):
    """One launch for a same-dtype, same-shape, same-exponent tensor list."""
    k = len(tensors)
    dev = _driver.active.get_current_device()
    dt = _dt_of(tensors[0].dtype)
    addrs = tuple(t.data_ptr() for t in tensors)
    tkey = (dev, dt, n, addrs)
    tbl = _TABLES.get(tkey)
    if tbl is None:
        tbl = torch.tensor(addrs, dtype=torch.int64, device=tensors[0].device)
        _TABLES[tkey] = tbl
    grid3 = ((n + _BLOCK - 1) // _BLOCK, k, 1)
    rkey = (dev, dt, mode, exp, n, k)
    runner = _RUNNERS.get(rkey)
    if runner is None:
        runner = _prepare(_pow_table, [tbl, n, exp, dt, mode, _BLOCK], grid3, mode)
        if runner is None:
            _pow_table[grid3[:2]](
                tbl,
                n,
                exp,
                DT=dt,
                MODE=mode,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
            return
        _RUNNERS[rkey] = runner
    runner(
        tbl,
        n,
        exp,
        dt,
        mode,
        _BLOCK,
        stream=_driver.active.get_current_stream(dev),
    )


def _launch_multi(tensors, sizes, exp, mode):
    """One launch for a same-dtype, same-exponent list of differing shapes."""
    k = len(tensors)
    max_n = max(sizes)
    pad = _FUSED_SLOTS - k
    first = tensors[0]
    ptrs = list(tensors)
    full = sizes
    if pad:
        ptrs = ptrs + [first] * pad
        full = sizes + [0] * pad
    dev = _driver.active.get_current_device()
    # Triton specializes pointer arguments on 16-byte divisibility, so the
    # prepared launcher is keyed by the same facts.
    pkey = tuple(p.data_ptr() % 16 == 0 for p in ptrs)
    rkey = (dev, first.dtype, mode, exp, tuple(sizes), pkey)
    grid3 = ((max_n + _BLOCK - 1) // _BLOCK, k, 1)
    runner = _RUNNERS.get(rkey)
    if runner is None:
        payload = ptrs + full + [exp, mode, _BLOCK]
        runner = _prepare(_pow_multi, payload, grid3, mode)
        if runner is None:
            _pow_multi[grid3[:2]](
                *ptrs, *full, exp, MODE=mode, BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
            return
        _RUNNERS[rkey] = runner
    runner(
        *ptrs,
        *full,
        exp,
        mode,
        _BLOCK,
        stream=_driver.active.get_current_stream(dev),
    )


def _foreach_pow__ScalarList(self, exponent):
    logger.debug("GEMS_HYGON _FOREACH_POW__SCALARLIST")
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    n = len(tensors)
    if n == 0:
        return None
    if isinstance(exponent, (list, tuple)):
        exps = exponent
    else:
        exps = [exponent] * n
    if len(exps) != n:
        raise RuntimeError(
            "_foreach_pow_.ScalarList: exponent list length must match tensor list length"
        )
    if 1 < n <= _FUSED_SLOTS:
        e0 = exps[0]
        if isinstance(e0, torch.Tensor):
            e0 = e0.item()
        e0 = float(e0)
        dtype0 = tensors[0].dtype
        total = 0
        uniform = True
        fused = True
        for t, e in zip(tensors, exps):
            if isinstance(e, torch.Tensor):
                e = e.item()
            if (
                e != e0
                or t.dtype is not dtype0
                or not t.is_contiguous()
                or t.numel() == 0
            ):
                fused = False
                break
            total += t.numel()
            if t.numel() != tensors[0].numel():
                uniform = False
        if fused and total <= _FUSED_MAX_ELEMS:
            mode = _mode_for(e0)
            if uniform:
                _launch_uniform(tensors, tensors[0].numel(), e0, mode)
            else:
                _launch_multi(tensors, [t.numel() for t in tensors], e0, mode)
            return None
    for t, e in zip(tensors, exps):
        if t.numel() == 0:
            continue
        if isinstance(e, torch.Tensor):
            e = e.item()
        _launch_one(t, float(e))
    return None
