# Copyright 2026 FlagOS Contributors
# Copyright (c) 2024 FlagGems. All rights reserved.
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

from flag_gems.ops.contiguous import contiguous as _gems_contiguous

"""Bounded compiled-code cache using public Triton launchers.

Only argument type, alignment and scalar metadata specialize code; tensor data
and returned values are never retained. Current stream and runtime hooks are
resolved by each public CompiledKernel grid launch.
"""
_hygon_launch_cached__CACHE = {}


def _hygon_launch_cached__launch(kernel, grid, *args, **kwargs):
    names = kernel.arg_names
    full_args = args + tuple((kwargs[name] for name in names[len(args) :]))
    options = {name: value for (name, value) in kwargs.items() if name not in names}
    signature = tuple(
        (
            (
                (value.dtype, value.data_ptr() % 16 == 0)
                if isinstance(value, torch.Tensor)
                else value
            )
            for value in full_args
        )
    )
    key = (
        kernel,
        torch.cuda.current_device(),
        signature,
        tuple(sorted(options.items())),
    )
    compiled = _hygon_launch_cached__CACHE.get(key)
    if compiled is None:
        compiled = kernel[grid](*full_args, **options)
        if compiled is not None:
            if len(_hygon_launch_cached__CACHE) >= 256:
                _hygon_launch_cached__CACHE.pop(next(iter(_hygon_launch_cached__CACHE)))
            _hygon_launch_cached__CACHE[key] = compiled
    else:
        dims = tuple(grid) + (1,) * (3 - len(grid))
        compiled[dims](*full_args, stream=torch.cuda.current_stream().cuda_stream)


launch = _hygon_launch_cached__launch
logger = logging.getLogger(__name__)
_BN_FUSED = 1024
_BN_MAX = 4096
_BN_TWO = 1024
_TARGET_PROG = 160
_PROG_FUSED = 128
_FOLD_TILE = 8
_SCR = {}


def _scratch(device):
    """Per-device zero-initialised scratch: a program counter, an fp32
    grad_weight accumulator, and the program count a launch reports about
    itself.  Every kernel restores or overwrites what it uses before returning,
    so each launch starts from the same state; nothing here depends on the
    inputs and no value is carried between calls as a result."""
    key = (str(device), torch.cuda.current_stream(device).cuda_stream)
    scr = _SCR.get(key)
    if scr is None:
        scr = (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(4096, dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        _SCR[key] = scr
    return scr


@triton.jit
def _k_reduce(PART, DW, N, NPROG, BLOCK: tl.constexpr, BLOCK_R: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cm = offs < N
    tot = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(0, NPROG, BLOCK_R):
        rows = k + tl.arange(0, BLOCK_R)
        rm = rows < NPROG
        v = tl.load(
            PART + rows[:, None] * N + offs[None, :],
            mask=rm[:, None] & cm[None, :],
            other=0.0,
        )
        tot += tl.sum(v.to(tl.float32), axis=0)
    tl.store(DW + offs, tot.to(DW.dtype.element_ty), mask=cm)


@triton.jit
def _k_row(
    X,
    DY,
    RSTD,
    W,
    DX,
    DW,
    ACC,
    CNT,
    M,
    N,
    rs,
    cs,
    NPROG,
    BLOCK_N: tl.constexpr,
    HAS_W: tl.constexpr,
    DO_DX: tl.constexpr,
    DO_DW: tl.constexpr,
    DIRECT: tl.constexpr,
    DO_FOLD: tl.constexpr,
):
    """One normalized row per step; a program strides over its row subset.
    grad_weight is accumulated in registers and written once.  With DO_FOLD the
    per-program vectors are combined by device-scope fp32 atomics into a
    zero-maintained accumulator, and the program that arrives last casts it to
    the output dtype and re-zeroes it, so the operator takes a single launch."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    cm = offs < N
    if HAS_W:
        wv = tl.load(W + offs, mask=cm, other=0.0).to(tl.float32)
    else:
        wv = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m in range(pid, M, NPROG):
        p = m * rs + offs * cs
        x = tl.load(X + p, mask=cm, other=0.0).to(tl.float32)
        dy = tl.load(DY + p, mask=cm, other=0.0).to(tl.float32)
        r = tl.load(RSTD + m).to(tl.float32)
        xn = x * r
        if DO_DX:
            if HAS_W:
                g = dy * wv
            else:
                g = dy
            s = tl.sum(xn * g, axis=0) * (1.0 / N)
            dx = (g - xn * s) * r
            tl.store(DX + p, dx.to(DX.dtype.element_ty), mask=cm)
        if DO_DW:
            acc += dy * xn
    if DO_DW:
        if DIRECT:
            tl.store(DW + offs, acc.to(DW.dtype.element_ty), mask=cm)
        elif DO_FOLD:
            tl.atomic_add(ACC + offs, acc, mask=cm)
            one = tl.arange(0, 1)
            tl.debug_barrier()
            old = tl.atomic_add(
                CNT + one, tl.full([1], 1, tl.int32), sem="acq_rel", scope="gpu"
            )
            if tl.sum(old) == NPROG - 1:
                v = tl.load(ACC + offs, mask=cm, other=0.0, cache_modifier=".cg")
                tl.store(DW + offs, v.to(DW.dtype.element_ty), mask=cm)
                tl.store(ACC + offs, tl.zeros([BLOCK_N], dtype=tl.float32), mask=cm)
                tl.store(CNT + one, tl.zeros([1], tl.int32), mask=one < 1)
        else:
            tl.store(ACC + pid * N + offs, acc, mask=cm)


@triton.jit
def _k_reduce_np(PART, DW, NG, N, BLOCK: tl.constexpr, BLOCK_R: tl.constexpr):
    """Fold of the wide-row partials; the number of partial rows is read from
    NG, which the producing kernel wrote itself, so the launch shape of the main
    kernel stays free to be autotuned."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cm = offs < N
    nprog = tl.load(NG)
    tot = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(0, nprog, BLOCK_R):
        rows = k + tl.arange(0, BLOCK_R)
        rm = rows < nprog
        v = tl.load(
            PART + rows[:, None] * N + offs[None, :],
            mask=rm[:, None] & cm[None, :],
            other=0.0,
        )
        tot += tl.sum(v.to(tl.float32), axis=0)
    tl.store(DW + offs, tot.to(DW.dtype.element_ty), mask=cm)


_R2_CONFIGS = [
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
]
_R2_MIN_BM = 2


@triton.autotune(configs=_R2_CONFIGS, key=["N", "ES"])
@triton.jit
def _k_rows2(
    X,
    DY,
    RSTD,
    W,
    DX,
    DW,
    PART,
    NG,
    ES,
    M,
    N,
    rs,
    cs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_W: tl.constexpr,
    DO_DX: tl.constexpr,
    DO_DW: tl.constexpr,
):
    """Wide rows (N > _BN_MAX): one column pass accumulating the per-row dot
    product, then one column pass writing grad_input and the per-program
    grad_weight partial.  ES only widens the autotune key by element size."""
    pid = tl.program_id(0)
    tl.store(NG + tl.arange(0, 1), tl.num_programs(0) + tl.zeros([1], tl.int32))
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = rows < M
    r = tl.load(RSTD + rows, mask=rm, other=0.0).to(tl.float32)
    base = rows[:, None] * rs
    nt = tl.cdiv(N, BLOCK_N)
    s = tl.zeros([BLOCK_M], dtype=tl.float32)
    if DO_DX:
        for t in range(0, nt):
            offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
            cm = offs < N
            ptr = base + offs[None, :] * cs
            mk = rm[:, None] & cm[None, :]
            x = tl.load(X + ptr, mask=mk, other=0.0).to(tl.float32)
            dy = tl.load(DY + ptr, mask=mk, other=0.0).to(tl.float32)
            if HAS_W:
                g = dy * tl.load(W + offs, mask=cm, other=0.0).to(tl.float32)[None, :]
            else:
                g = dy
            s += tl.sum(x * r[:, None] * g, axis=1)
        s = s * (1.0 / N)
    for t in range(0, nt):
        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
        cm = offs < N
        ptr = base + offs[None, :] * cs
        mk = rm[:, None] & cm[None, :]
        x = tl.load(X + ptr, mask=mk, other=0.0).to(tl.float32)
        dy = tl.load(DY + ptr, mask=mk, other=0.0).to(tl.float32)
        xn = x * r[:, None]
        if DO_DX:
            if HAS_W:
                g = dy * tl.load(W + offs, mask=cm, other=0.0).to(tl.float32)[None, :]
            else:
                g = dy
            dx = (g - xn * s[:, None]) * r[:, None]
            tl.store(DX + ptr, dx.to(DX.dtype.element_ty), mask=mk)
        if DO_DW:
            tl.store(PART + pid * N + offs, tl.sum(dy * xn, axis=0), mask=cm)


def _pow2_le(v):
    v = int(v)
    return 1 << v.bit_length() - 1 if v > 0 else 1


def _fused_rms_norm_backward(
    grad_out, input, normalized_shape, rstd, weight=None, output_mask=(True, True)
):
    logger.debug("GEMS_HYGON _FUSED_RMS_NORM_BACKWARD")
    do_dx = bool(output_mask[0])
    has_w = weight is not None
    do_dw = bool(output_mask[1]) and has_w
    if not do_dx and (not do_dw):
        return (None, None)
    x = input
    if not x.is_contiguous():
        x = _gems_contiguous(x)
    dy = grad_out
    if not dy.is_contiguous():
        dy = _gems_contiguous(dy)
    ns = tuple((int(v) for v in normalized_shape))
    N = 1
    for v in ns:
        N *= v
    M = x.numel() // N
    xv = x.reshape(M, N)
    dyv = dy.reshape(M, N)
    rs = xv.stride(0)
    cs = xv.stride(1)
    rv = rstd.reshape(-1)
    wv = weight.reshape(-1) if has_w else xv
    dx = torch.empty_like(x) if do_dx else None
    dxv = dx.reshape(M, N) if do_dx else xv
    dw_flat = torch.empty(N, dtype=x.dtype, device=x.device) if do_dw else None
    bn = triton.next_power_of_2(N)
    part = None
    if bn <= _BN_FUSED:
        nprog = min(M, _PROG_FUSED if do_dw else _TARGET_PROG)
        direct = nprog == 1
        do_fold = do_dw and (not direct)
        if do_fold:
            cnt, acc_buf, _ng = _scratch(x.device)
            part = acc_buf
        else:
            cnt = xv
        launch(
            _k_row,
            (nprog,),
            xv,
            dyv,
            rv,
            wv,
            dxv,
            dw_flat if do_dw else xv,
            part if part is not None else xv,
            cnt,
            M,
            N,
            rs,
            cs,
            nprog,
            bn,
            has_w,
            do_dx,
            do_dw,
            direct,
            do_fold,
            num_warps=max(1, min(8, bn // 128)),
        )
    elif bn <= _BN_MAX:
        cap = max(1, int(M * x.element_size()) // 8)
        nprog = min(M, _TARGET_PROG, cap)
        direct = nprog == 1
        if do_dw and (not direct):
            part = torch.empty((nprog, N), dtype=torch.float32, device=x.device)
        launch(
            _k_row,
            (nprog,),
            xv,
            dyv,
            rv,
            wv,
            dxv,
            dw_flat if do_dw else xv,
            part if part is not None else xv,
            xv,
            M,
            N,
            rs,
            cs,
            nprog,
            bn,
            has_w,
            do_dx,
            do_dw,
            direct,
            False,
            num_warps=max(1, min(8, bn // 128)),
        )
        if do_dw and (not direct):
            launch(
                _k_reduce,
                (triton.cdiv(N, 512),),
                part,
                dw_flat,
                N,
                nprog,
                512,
                _FOLD_TILE,
            )
    else:
        ng = _scratch(x.device)[2]
        maxrow = -(-M // _R2_MIN_BM)
        if do_dw:
            part = torch.empty((maxrow, N), dtype=torch.float32, device=x.device)
        else:
            part = xv
        _k_rows2[lambda META: (triton.cdiv(M, META["BLOCK_M"]),)](
            xv,
            dyv,
            rv,
            wv,
            dxv,
            dw_flat if do_dw else xv,
            part,
            ng,
            x.element_size(),
            M,
            N,
            rs,
            cs,
            HAS_W=has_w,
            DO_DX=do_dx,
            DO_DW=do_dw,
        )
        if do_dw:
            launch(
                _k_reduce_np,
                (triton.cdiv(N, 512),),
                part,
                dw_flat,
                ng,
                N,
                512,
                _FOLD_TILE,
            )
    dw = dw_flat.view(ns) if do_dw else None
    return (dx, dw)
