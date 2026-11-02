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

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_BLOCK = 1024
_RBLOCK = 1024
_HIST_BLOCK = 2048
_SMALL_BLOCK = 1024
_HIST_NB_MAX = 512
_HIST_WARPS = 2
_HIST_ACC_MIN = 1 << 22  # use register-accumulate variant above this many elements


@triton.jit
def _div_rn32(num, sp):
    # Correctly-rounded fp32 division via plain div + Markstein FMA refinement,
    # matching IEEE-RN (torch reference) while avoiding the slow precise_divf.
    q0 = num / sp
    e = tl.math.fma(-q0, sp, num)
    rc = 1.0 / sp
    return tl.math.fma(e, rc, q0)


@triton.jit
def _histc_hist_kernel(
    inp_ptr,
    out_ptr,
    mm_ptr,
    n_elements,
    mn,
    mx,
    bn,
    IS_FLOAT: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_INT: tl.constexpr,
    USE_PTR: tl.constexpr,
    NO_MASK: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if NO_MASK:
        x = tl.load(inp_ptr + offs)
    else:
        mask = offs < n_elements
        x = tl.load(inp_ptr + offs, mask=mask, other=0)
    if USE_PTR:
        mn = tl.load(mm_ptr + 0)
        mx = tl.load(mm_ptr + 1)
        bn = tl.load(mm_ptr + 2)
    if IS_INT:
        xi = x.to(tl.int64)
        mni = mn.to(tl.int64)
        mxi = mx.to(tl.int64)
        bni = bn.to(tl.int64)
        span = mxi - mni
        safe = tl.where(span == 0, 1, span)
        t = (xi - mni) * bni // safe
        idx = t.to(tl.int32)
        idx = tl.where(span == 0, (bni // 2).to(tl.int32), idx)
        idx = tl.minimum(idx, bni.to(tl.int32) - 1)
        idx = tl.maximum(idx, 0)
        if NO_MASK:
            valid = (x >= mni) & (x <= mxi)
        else:
            valid = mask & (x >= mni) & (x <= mxi)
    else:
        sp = mx - mn
        if IS_FP64:
            t = (x - mn) * bn / sp
        else:
            t = _div_rn32((x - mn) * bn, sp)
        idx = t.to(tl.int32)
        idx = tl.where(sp == 0, bn.to(tl.int32) // 2, idx)
        idx = tl.minimum(idx, bn.to(tl.int32) - 1)
        idx = tl.maximum(idx, 0)
        if NO_MASK:
            valid = (x >= mn) & (x <= mx)
        else:
            valid = mask & (x >= mn) & (x <= mx)
    hidx = tl.where(valid, idx, NB - 1)
    h = tl.histogram(hidx, num_bins=NB)
    if IS_INT:
        if IS_FP64:
            hc = h.to(tl.int64)
        else:
            hc = h.to(tl.int32)
    else:
        if IS_FP64:
            hc = h.to(tl.float64)
        else:
            hc = h.to(tl.float32)
    tl.atomic_add(
        out_ptr + tl.arange(0, NB), hc, mask=tl.arange(0, NB) < bn.to(tl.int32)
    )


@triton.jit
def _histc_acc_kernel(
    inp_ptr,
    out_ptr,
    mm_ptr,
    n_elements,
    mn,
    mx,
    bn,
    IS_FLOAT: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_INT: tl.constexpr,
    USE_PTR: tl.constexpr,
    NO_MASK: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERS: tl.constexpr,
):
    # Register-accumulate variant: each block covers ITERS*BLOCK elements with
    # per-block histograms accumulated in registers, then one atomic merge.
    pid = tl.program_id(0).to(tl.int64)
    if USE_PTR:
        mn = tl.load(mm_ptr + 0)
        mx = tl.load(mm_ptr + 1)
        bn = tl.load(mm_ptr + 2)
    if IS_INT:
        if IS_FP64:
            acc = tl.zeros([NB], dtype=tl.int64)
        else:
            acc = tl.zeros([NB], dtype=tl.int32)
    else:
        if IS_FP64:
            acc = tl.zeros([NB], dtype=tl.float64)
        else:
            acc = tl.zeros([NB], dtype=tl.float32)
    for i in tl.range(0, ITERS, num_stages=2):
        base = pid * (ITERS * BLOCK) + i * BLOCK
        offs = base + tl.arange(0, BLOCK).to(tl.int64)
        if NO_MASK:
            x = tl.load(inp_ptr + offs)
        else:
            mask = offs < n_elements
            x = tl.load(inp_ptr + offs, mask=mask, other=0)
        if IS_INT:
            xi = x.to(tl.int64)
            mni = mn.to(tl.int64)
            mxi = mx.to(tl.int64)
            bni = bn.to(tl.int64)
            span = mxi - mni
            safe = tl.where(span == 0, 1, span)
            t = (xi - mni) * bni // safe
            idx = t.to(tl.int32)
            idx = tl.where(span == 0, (bni // 2).to(tl.int32), idx)
            idx = tl.minimum(idx, bni.to(tl.int32) - 1)
            idx = tl.maximum(idx, 0)
            if NO_MASK:
                valid = (x >= mni) & (x <= mxi)
            else:
                valid = mask & (x >= mni) & (x <= mxi)
        else:
            sp = mx - mn
            if IS_FP64:
                t = (x - mn) * bn / sp
            else:
                t = _div_rn32((x - mn) * bn, sp)
            idx = t.to(tl.int32)
            idx = tl.where(sp == 0, bn.to(tl.int32) // 2, idx)
            idx = tl.minimum(idx, bn.to(tl.int32) - 1)
            idx = tl.maximum(idx, 0)
            if NO_MASK:
                valid = (x >= mn) & (x <= mx)
            else:
                valid = mask & (x >= mn) & (x <= mx)
        hidx = tl.where(valid, idx, NB - 1)
        h = tl.histogram(hidx, num_bins=NB)
        acc += h
    tl.atomic_add(
        out_ptr + tl.arange(0, NB), acc, mask=tl.arange(0, NB) < bn.to(tl.int32)
    )


@triton.jit
def _histc_atomic_kernel(
    inp_ptr,
    out_ptr,
    mm_ptr,
    n_elements,
    mn,
    mx,
    bn,
    IS_FLOAT: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_INT: tl.constexpr,
    USE_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask, other=0)
    if USE_PTR:
        mn = tl.load(mm_ptr + 0)
        mx = tl.load(mm_ptr + 1)
        bn = tl.load(mm_ptr + 2)
    if IS_INT:
        xi = x.to(tl.int64)
        mni = mn.to(tl.int64)
        mxi = mx.to(tl.int64)
        bni = bn.to(tl.int64)
        span = mxi - mni
        safe = tl.where(span == 0, 1, span)
        t = (xi - mni) * bni // safe
        idx = t.to(tl.int32)
        idx = tl.where(span == 0, (bni // 2).to(tl.int32), idx)
        idx = tl.minimum(idx, bni.to(tl.int32) - 1)
        idx = tl.maximum(idx, 0)
        valid = mask & (x >= mni) & (x <= mxi)
    else:
        sp = mx - mn
        if IS_FP64:
            t = (x - mn) * bn / sp
        else:
            t = _div_rn32((x - mn) * bn, sp)
        idx = t.to(tl.int32)
        idx = tl.where(sp == 0, bn.to(tl.int32) // 2, idx)
        idx = tl.minimum(idx, bn.to(tl.int32) - 1)
        idx = tl.maximum(idx, 0)
        valid = mask & (x >= mn) & (x <= mx)
    one = tl.zeros_like(x) + 1
    tl.atomic_add(out_ptr + idx, one, mask=valid)


@triton.jit
def _block_minmax_kernel(
    inp_ptr, mn_ptr, mx_ptr, n_elements, IS_FLOAT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask, other=0)
    if IS_FLOAT:
        xmn = tl.where(mask, x, float("inf"))
        xmx = tl.where(mask, x, float("-inf"))
    else:
        xmn = tl.where(mask, x.to(tl.int64), 2**63 - 1)
        xmx = tl.where(mask, x.to(tl.int64), -(2**63))
    mn = tl.min(xmn, axis=0)
    mx = tl.max(xmx, axis=0)
    tl.store(mn_ptr + pid, mn)
    tl.store(mx_ptr + pid, mx)


@triton.jit
def _final_minmax_kernel(
    mn_ptr,
    mx_ptr,
    out_ptr,
    G,
    IS_FLOAT: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if IS_FLOAT:
        if IS_FP64:
            hi = tl.full((), float("inf"), tl.float64)
            lo = tl.full((), float("-inf"), tl.float64)
        else:
            hi = float("inf")
            lo = float("-inf")
    else:
        hi = 2**63 - 1
        lo = -(2**63)
    mn_acc = hi
    mx_acc = lo
    for start in tl.range(0, G, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < G
        v = tl.load(mn_ptr + offs, mask=m, other=hi)
        mn_acc = tl.minimum(mn_acc, tl.min(v, axis=0))
        v = tl.load(mx_ptr + offs, mask=m, other=lo)
        mx_acc = tl.maximum(mx_acc, tl.max(v, axis=0))
    tl.store(out_ptr + 0, mn_acc)
    tl.store(out_ptr + 1, mx_acc)


def _scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return v


def histc(inp, bins=100, min=0, max=0):
    bins_i = int(_scalar(bins))
    min_v = float(_scalar(min))
    max_v = float(_scalar(max))

    if not inp.is_contiguous():
        inp = inp.contiguous()
    dt = inp.dtype
    is_float = dt.is_floating_point
    is_fp64 = dt == torch.float64
    n = inp.numel()

    out = torch.zeros(bins_i, dtype=dt, device=inp.device)
    if n == 0 or bins_i <= 0:
        return out

    mm = None
    use_ptr = False
    if min_v == max_v:
        G = triton.cdiv(n, _RBLOCK)
        pdtype = dt if is_float else torch.int64
        if is_fp64:
            mm = torch.tensor(
                [min_v, max_v, float(bins_i)], dtype=torch.float64, device=inp.device
            )
        elif is_float:
            mm = torch.tensor(
                [min_v, max_v, float(bins_i)], dtype=torch.float32, device=inp.device
            )
        else:
            mm = torch.tensor(
                [int(min_v), int(max_v), int(bins_i)],
                dtype=torch.int64,
                device=inp.device,
            )
        mn_part = torch.empty(G, dtype=pdtype, device=inp.device)
        mx_part = torch.empty(G, dtype=pdtype, device=inp.device)
        _block_minmax_kernel[(G,)](
            inp, mn_part, mx_part, n, IS_FLOAT=is_float, BLOCK=_RBLOCK
        )
        _final_minmax_kernel[(1,)](
            mn_part, mx_part, mm, G, IS_FLOAT=is_float, IS_FP64=is_fp64, BLOCK=_RBLOCK
        )
        use_ptr = True
    elif is_float and not is_fp64:
        pass
    else:
        if is_fp64:
            mm = torch.tensor(
                [min_v, max_v, float(bins_i)], dtype=torch.float64, device=inp.device
            )
        else:
            mm = torch.tensor(
                [int(min_v), int(max_v), int(bins_i)],
                dtype=torch.int64,
                device=inp.device,
            )
        use_ptr = True

    if mm is None:
        mm = inp

    if is_float and not is_fp64:
        mn_arg = float(np.float32(min_v))
        mx_arg = float(np.float32(max_v))
        bn_arg = float(np.float32(bins_i))
    else:
        mn_arg = 0.0
        mx_arg = 0.0
        bn_arg = 0

    nb = 1
    while nb <= bins_i:
        nb <<= 1

    if nb <= _HIST_NB_MAX:
        if n > _HIST_ACC_MIN:
            span = 4 * _HIST_BLOCK
            grid = (triton.cdiv(n, span),)
            _histc_acc_kernel[grid](
                inp,
                out,
                mm,
                n,
                mn_arg,
                mx_arg,
                bn_arg,
                IS_FLOAT=is_float,
                IS_FP64=is_fp64,
                IS_INT=not is_float,
                USE_PTR=use_ptr,
                NO_MASK=(n % span == 0),
                NB=nb,
                BLOCK=_HIST_BLOCK,
                num_warps=_HIST_WARPS,
                ITERS=4,
            )
        elif n > 65536:
            grid = (triton.cdiv(n, _HIST_BLOCK),)
            _histc_hist_kernel[grid](
                inp,
                out,
                mm,
                n,
                mn_arg,
                mx_arg,
                bn_arg,
                IS_FLOAT=is_float,
                IS_FP64=is_fp64,
                IS_INT=not is_float,
                USE_PTR=use_ptr,
                NO_MASK=(n % _HIST_BLOCK == 0),
                NB=nb,
                BLOCK=_HIST_BLOCK,
                num_warps=4,
            )
        else:
            grid = (triton.cdiv(n, _SMALL_BLOCK),)
            _histc_hist_kernel[grid](
                inp,
                out,
                mm,
                n,
                mn_arg,
                mx_arg,
                bn_arg,
                IS_FLOAT=is_float,
                IS_FP64=is_fp64,
                IS_INT=not is_float,
                USE_PTR=use_ptr,
                NO_MASK=(n % _SMALL_BLOCK == 0),
                NB=nb,
                BLOCK=_SMALL_BLOCK,
                num_warps=_HIST_WARPS,
            )
    else:
        grid = (triton.cdiv(n, _BLOCK),)
        _histc_atomic_kernel[grid](
            inp,
            out,
            mm,
            n,
            mn_arg,
            mx_arg,
            bn_arg,
            IS_FLOAT=is_float,
            IS_FP64=is_fp64,
            IS_INT=not is_float,
            USE_PTR=use_ptr,
            BLOCK=_BLOCK,
        )
    return out
