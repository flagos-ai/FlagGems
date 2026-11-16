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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.adaptive_avg_pool3d_backward import (
    _adaptive_avg_pool3d_backward as default_adaptive_avg_pool3d_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _aap3d_bwd_kernel(
    go_ptr,
    out_ptr,
    C,
    Di,
    Hi,
    Wi,
    Do,
    Ho,
    Wo,
    sN,
    sC,
    sD,
    sH,
    sW,
    KD,
    KH,
    KW,
    I64: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    # program ids: plane = (n, c, id) on axis 0, tile over (ih, iw) on axes 1/2
    z = tl.program_id(0)
    ph = tl.program_id(1)
    pw = tl.program_id(2)
    if I64:
        z = z.to(tl.int64)

    id = z % Di
    nc = z // Di
    n = nc // C
    c = nc % C
    base = n * sN + c * sC

    ih = ph * BH + tl.arange(0, BH)
    iw = pw * BW + tl.arange(0, BW)
    mh = ih < Hi
    mw = iw < Wi
    ihs = tl.where(mh, ih, 0)
    iws = tl.where(mw, iw, 0)

    # backward incidence ranges per spatial dimension
    sd = (id * Do) // Di
    ed = ((id + 1) * Do + Di - 1) // Di
    cd = ed - sd

    sh = (ihs * Ho) // Hi
    eh = ((ihs + 1) * Ho + Hi - 1) // Hi
    ch = eh - sh
    sw = (iws * Wo) // Wi
    ew = ((iws + 1) * Wo + Wi - 1) // Wi
    cw = ew - sw

    acc = tl.zeros((BH, BW), dtype=tl.float32)

    for kd in range(KD):
        od = sd + kd
        md = (kd < cd) & (od < Do)
        ad = (od * Di) // Do
        bd = ((od + 1) * Di + Do - 1) // Do
        inv_d = 1.0 / (bd - ad).to(tl.float32)
        doff = base + od * sD
        for kh in range(KH):
            oh = sh + kh
            mdh = md & mh & (kh < ch)
            ah = (oh * Hi) // Ho
            bh = ((oh + 1) * Hi + Ho - 1) // Ho
            inv_h = 1.0 / (bh - ah).to(tl.float32)
            rowoff = doff + oh * sH
            rowm = mdh[:, None]
            for kw in range(KW):
                ow = sw + kw
                aw = (ow * Wi) // Wo
                bw = ((ow + 1) * Wi + Wo - 1) // Wo
                inv_w = 1.0 / (bw - aw).to(tl.float32)
                scale = (inv_d * inv_h)[:, None] * inv_w[None, :]
                m3 = rowm & (mw & (kw < cw))[None, :]
                off = rowoff[:, None] + (ow * sW)[None, :]
                g = tl.load(go_ptr + off, mask=m3, other=0.0).to(tl.float32)
                acc += g * scale

    out_off = (z * Hi + ihs[:, None]) * Wi + iws[None, :]
    tl.store(
        out_ptr + out_off,
        acc.to(out_ptr.dtype.element_ty),
        mask=mh[:, None] & mw[None, :],
    )


def _next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _max_count(inner, outer):
    """Exact maximum of (end(i) - start(i)) over input positions i."""
    if inner <= 0 or outer <= 0:
        return 0
    g = math.gcd(inner, outer)
    return (inner - g + outer + inner - 1) // inner


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _adaptive_avg_pool3d_backward(grad_output, self=None, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _ADAPTIVE_AVG_POOL3D_BACKWARD")

    if not _use_triton_kernel(grad_output):
        return default_adaptive_avg_pool3d_backward(grad_output, self, *args, **kwargs)

    with torch_device_fn.device(grad_output.device):
        return _launch_aap3d_bwd(grad_output, self, *args, **kwargs)


def _launch_aap3d_bwd(grad_output, self=None, *args, **kwargs):
    if self is None:
        for key in ("self", "input", "input_tensor", "inp", "src"):
            if key in kwargs:
                self = kwargs[key]
                break
    if self is None and len(args) >= 1:
        self = args[-1]
    if self is None:
        raise TypeError("run() missing required argument 'self'")

    go = grad_output
    inp = self

    go5 = go if go.dim() == 5 else go.unsqueeze(0)
    in5 = inp if inp.dim() == 5 else inp.unsqueeze(0)

    N, C, Di, Hi, Wi = (int(v) for v in in5.shape)
    Do, Ho, Wo = (int(v) for v in go5.shape[2:])

    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if out.numel() == 0 or N * C * Di == 0:
        return out

    KD = _max_count(Di, Do)
    KH = _max_count(Hi, Ho)
    KW = _max_count(Wi, Wo)

    sN, sC, sD, sH, sW = (int(s) for s in go5.stride())

    max_off = (
        max(N - 1, 0) * abs(sN)
        + max(C - 1, 0) * abs(sC)
        + max(Do - 1, 0) * abs(sD)
        + max(Ho - 1, 0) * abs(sH)
        + max(Wo - 1, 0) * abs(sW)
    )
    out_elems = N * C * Di * Hi * Wi
    planes = N * C * Di
    i64 = (out_elems >= 2**31) or (max_off >= 2**31) or (planes >= 2**31)

    bw = min(_next_pow2(Wi), 64)
    bh = max(1, min(_next_pow2(Hi), max(1, 1024 // bw)))
    tile = bh * bw
    if tile <= 256:
        num_warps = 1
    elif tile <= 512:
        num_warps = 2
    else:
        num_warps = 4

    grid = (planes, triton.cdiv(Hi, bh), triton.cdiv(Wi, bw))
    _aap3d_bwd_kernel[grid](
        go5,
        out,
        C,
        Di,
        Hi,
        Wi,
        Do,
        Ho,
        Wo,
        sN,
        sC,
        sD,
        sH,
        sW,
        KD,
        KH,
        KW,
        I64=i64,
        BH=bh,
        BW=bw,
        num_warps=num_warps,
    )
    return out
