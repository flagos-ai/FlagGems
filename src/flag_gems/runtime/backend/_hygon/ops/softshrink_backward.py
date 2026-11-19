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

"""Triton implementation of softshrink_backward.

out = grad_output  where  not (|self| <= lambd)   (i.e. |self| > lambd, NaN -> pass through)
    = 0          otherwise
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _keep(x, lambd, WIDE: tl.constexpr):
    """ATen predicate: grad flows where not(|self| <= lambd) (NaN passes through)."""
    if WIDE:
        xc = x
    else:
        xc = x.to(tl.float32)
    return ~(tl.abs(xc) <= lambd)


@triton.jit
def _ssb_flat_kernel(
    go_ptr,
    x_ptr,
    lambd_ptr,
    out_ptr,
    n_elements,
    lambd,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    LAMBDA_PTR: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    g = tl.load(go_ptr + offs, mask=mask, other=0)
    if LAMBDA_PTR:
        lam = tl.load(lambd_ptr)
    else:
        lam = lambd
    keep = _keep(x, lam, WIDE)
    res = tl.where(keep, g, tl.zeros_like(g))
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _ssb_strided_kernel(
    go_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    lambd,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    rem = offs
    go_off = tl.zeros((BLOCK,), dtype=tl.int64)
    x_off = tl.zeros((BLOCK,), dtype=tl.int64)
    if NDIM >= 8:
        i = rem % s7
        rem = rem // s7
        go_off += i * g7
        x_off += i * x7
    if NDIM >= 7:
        i = rem % s6
        rem = rem // s6
        go_off += i * g6
        x_off += i * x6
    if NDIM >= 6:
        i = rem % s5
        rem = rem // s5
        go_off += i * g5
        x_off += i * x5
    if NDIM >= 5:
        i = rem % s4
        rem = rem // s4
        go_off += i * g4
        x_off += i * x4
    if NDIM >= 4:
        i = rem % s3
        rem = rem // s3
        go_off += i * g3
        x_off += i * x3
    if NDIM >= 3:
        i = rem % s2
        rem = rem // s2
        go_off += i * g2
        x_off += i * x2
    if NDIM >= 2:
        i = rem % s1
        rem = rem // s1
        go_off += i * g1
        x_off += i * x1
    go_off += rem * g0
    x_off += rem * x0
    x = tl.load(x_ptr + x_off, mask=mask, other=0)
    g = tl.load(go_ptr + go_off, mask=mask, other=0)
    keep = _keep(x, lambd, WIDE)
    res = tl.where(keep, g, tl.zeros_like(g))
    tl.store(out_ptr + offs, res, mask=mask)


_BLOCK = 1024
_NUM_WARPS = 4
_I64_LIMIT = 2**31 - 1


def _bcast_strides(t, shape):
    st = t.stride()
    sh = t.shape
    off = len(shape) - len(sh)
    res = []
    for j in range(len(shape)):
        sj = j - off
        if sj < 0:
            res.append(0)
        elif sh[sj] == shape[j]:
            res.append(st[sj])
        elif sh[sj] == 1:
            res.append(0)
        else:
            raise RuntimeError("incompatible shapes for broadcast")
    return res


def _collapse(shape, gs, xs):
    dims = []
    for j in range(len(shape)):
        if shape[j] == 1:
            continue
        dims.append([shape[j], gs[j], xs[j]])
    merged = []
    for d in dims:
        if merged:
            prev = merged[-1]
            if prev[1] == d[1] * d[0] and prev[2] == d[2] * d[0]:
                prev[0] = prev[0] * d[0]
                prev[1] = d[1]
                prev[2] = d[2]
                continue
        merged.append(d)
    return merged


def softshrink_backward(grad_output, self, lambd):
    logger.debug("GEMS_HYGON SOFTSHRINK_BACKWARD")
    go = grad_output
    x = self

    if isinstance(lambd, torch.Tensor):
        lambd_is_ptr = True
        lam_ptr = lambd
        lam_val = 0.0
    else:
        lambd_is_ptr = False
        lam_ptr = go
        lam_val = float(lambd)

    wide = x.dtype == torch.float64
    same_dtype = go.dtype == x.dtype

    if go.shape == x.shape and go.is_contiguous() and x.is_contiguous():
        if same_dtype:
            out = torch.empty_like(go)
        else:
            out = torch.empty(
                go.shape, dtype=torch.promote_types(go.dtype, x.dtype), device=go.device
            )
        n = out.numel()
        if n == 0:
            return out
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _ssb_flat_kernel[grid](
            go,
            x,
            lam_ptr,
            out,
            n,
            lam_val,
            BLOCK=_BLOCK,
            WIDE=wide,
            LAMBDA_PTR=lambd_is_ptr,
            I64=(n > _I64_LIMIT),
            num_warps=_NUM_WARPS,
        )
        return out

    shape = torch.broadcast_shapes(tuple(go.shape), tuple(x.shape))
    if same_dtype:
        out = torch.empty(shape, dtype=go.dtype, device=go.device)
    else:
        out = torch.empty(
            shape, dtype=torch.promote_types(go.dtype, x.dtype), device=go.device
        )
    n = out.numel()
    if n == 0:
        return out

    gs = _bcast_strides(go, shape)
    xs = _bcast_strides(x, shape)
    merged = _collapse(shape, gs, xs)
    ndim = len(merged)
    if ndim > 8:
        raise RuntimeError("softshrink_backward: unsupported rank after collapse")

    sm = [1] * 8
    gm = [0] * 8
    xm = [0] * 8
    for j, d in enumerate(merged):
        sm[j] = d[0]
        gm[j] = d[1]
        xm[j] = d[2]

    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _ssb_strided_kernel[grid](
        go,
        x,
        out,
        n,
        lam_val,
        sm[0],
        sm[1],
        sm[2],
        sm[3],
        sm[4],
        sm[5],
        sm[6],
        sm[7],
        gm[0],
        gm[1],
        gm[2],
        gm[3],
        gm[4],
        gm[5],
        gm[6],
        gm[7],
        xm[0],
        xm[1],
        xm[2],
        xm[3],
        xm[4],
        xm[5],
        xm[6],
        xm[7],
        NDIM=ndim,
        BLOCK=_BLOCK,
        WIDE=wide,
        num_warps=_NUM_WARPS,
    )
    return out
