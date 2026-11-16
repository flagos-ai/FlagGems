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

"""Kunlunxin (XPU3) override for ``aten::_slow_conv2d_forward``.

The generic implementation uses an implicit-GEMM ``tl.dot`` kernel whose
operands are built with ``tl.load(ptr, mask=..., other=0.0)``.  On XPU3 the
``other=`` merge lowers through ``rect_select`` (unsupported on arch3) and
masked 2-D dot operands silently miscompile: measured baseline 72/72 functional
cases failed with 98.6% mismatches.

This override keeps the generic argument validation (imported, read-only) and
computes with a mask-free scalar kernel: every gather clamps its address and
zeroes invalid lanes with ``tl.where``; the reduction is an outer-product FMA
chain over ``(ic, kh, kw)`` — no ``tl.dot``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._slow_conv2d_forward import _validate

logger = logging.getLogger(__name__)


@triton.jit(
    do_not_specialize=["N", "C_IN", "H", "W", "C_OUT", "H_OUT", "W_OUT",
                       "SH", "SW", "PH", "PW"]
)
def _sc2df_fma_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    N,
    C_IN,
    H,
    W,
    C_OUT,
    H_OUT,
    W_OUT,
    SH,
    SW,
    PH,
    PW,
    xsn,
    xsc,
    xsh,
    xsw,
    won,
    woc,
    wok,
    wol,
    osn,
    osc,
    osh,
    osw,
    HAS_BIAS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    OC_TILES: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_w = tl.program_id(0)
    oh = tl.program_id(1)
    pid_nz = tl.program_id(2)
    pid_n = pid_nz // OC_TILES
    oc = (pid_nz % OC_TILES) * BLOCK_OC + tl.arange(0, BLOCK_OC)

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_valid = ow < W_OUT
    oc_valid = oc < C_OUT
    occ = tl.minimum(oc, C_OUT - 1)

    acc = tl.zeros((BLOCK_W, BLOCK_OC), dtype=tl.float32)
    x_n = x_ptr + pid_n * xsn
    w_oc = w_ptr + occ * won
    for ic_i in range(0, C_IN):
        for kh_i in tl.static_range(0, KH):
            ih = oh * SH - PH + kh_i
            ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
            for kw_i in tl.static_range(0, KW):
                iw = ow * SW - PW + kw_i
                iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
                m = ow_valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xv = tl.load(x_n + ic_i * xsc + ihc * xsh + iwc * xsw)
                xv = tl.where(m, xv, 0.0).to(tl.float32)
                wv = tl.load(w_oc + ic_i * woc + (kh_i * wok + kw_i * wol))
                wv = tl.where(oc_valid, wv, 0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(b_ptr + occ)
        acc += tl.where(oc_valid, bv, 0.0).to(tl.float32)[None, :]

    optr = (
        o_ptr
        + pid_n * osn
        + (oh * osh + ow * osw)[:, None]
        + oc[None, :] * osc
    )
    tl.store(optr, acc.to(o_ptr.dtype.element_ty), mask=ow_valid[:, None] & oc_valid[None, :])




def _slow_conv2d_forward(
    self, weight, kernel_size, bias=None, stride=None, padding=None, output=None
):
    """aten::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding)."""
    logger.debug("GEMS _SLOW_CONV2D_FORWARD")
    x = self
    if stride is None:
        stride = (1, 1)
    if padding is None:
        padding = (0, 0)
    dims = _validate(x, weight, kernel_size, bias, stride, padding, output)
    k_h, k_w, s_h, s_w, p_h, p_w, h_out, w_out = dims

    if (x.requires_grad or weight.requires_grad
            or (bias is not None and bias.requires_grad)):
        raise RuntimeError(
            "_slow_conv2d_forward (kunlunxin): autograd is not supported by this "
            "backend override; call with requires_grad=False operands"
        )

    n, c_in, h, w = x.shape
    c_out = weight.shape[0]
    if output is not None:
        out = output
    else:
        out = torch.empty((n, c_out, h_out, w_out), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    block_w = 64 if w_out >= 64 else max(8, min(32, 1 << (max(1, int(w_out)) - 1).bit_length()))
    block_oc = min(32, 1 << (max(1, int(c_out)) - 1).bit_length())
    num_warps = 8 if (w_out >= 64 and c_in * k_h * k_w > 1) else 4
    oc_tiles = triton.cdiv(c_out, block_oc)
    grid = (triton.cdiv(w_out, block_w), h_out, n * oc_tiles)
    _sc2df_fma_kernel[grid](
        x, weight, bias if bias is not None else x, out,
        n, c_in, h, w, c_out, h_out, w_out,
        s_h, s_w, p_h, p_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        HAS_BIAS=(bias is not None),
        BLOCK_W=block_w, BLOCK_OC=block_oc, OC_TILES=oc_tiles, KH=k_h, KW=k_w,
        num_warps=num_warps,
    )
    return out
