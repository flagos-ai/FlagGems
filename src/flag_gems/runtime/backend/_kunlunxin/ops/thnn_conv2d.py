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

"""Kunlunxin (XPU3) override for ``aten::thnn_conv2d``.

The generic implementation builds the implicit-GEMM tile with
``tl.load(ptr, mask=..., other=0.0)`` and feeds it to ``tl.dot``.  On XPU3 the
``other=`` merge lowers through ``rect_select`` (unsupported on arch3, silently
returns garbage) and masked 2-D operands poison the dot lowering; measured
baseline: 44/55 functional cases failed with ``nan``.

This override keeps the generic host contract (operand materialisation, shape
checks, dtype dispatch, empty shapes) and computes with a mask-free scalar
kernel: every gather clamps its address and zeroes invalid lanes with
``tl.where``, and the reduction is an outer-product FMA chain over
``(ic, kh, kw)`` — no ``tl.dot`` anywhere.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(
    do_not_specialize=["N", "C_IN", "H", "W", "C_OUT", "H_OUT", "W_OUT",
                       "SH", "SW", "PH", "PW"]
)
def _thnn_fma_kernel(
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


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def thnn_conv2d(self, weight, kernel_size, bias=None, stride=1, padding=0):
    logger.debug("GEMS THNN_CONV2D")
    if not torch.is_tensor(weight):
        raise TypeError("thnn_conv2d(): weight must be a Tensor")
    x = self
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    if weight.dim() != 4:
        raise RuntimeError(
            f"thnn_conv2d(): non-empty 4D weight tensor expected, but got: "
            f"{list(weight.shape)}"
        )
    N, C_in, H, W = x.shape
    if weight.shape[1] != C_in:
        raise RuntimeError(
            f"thnn_conv2d(): Expected input tensor to have channel dimension "
            f"{weight.shape[1]} at index 1, but got channel dimension {C_in} instead"
        )
    C_out = weight.shape[0]

    KH, KW = _pair(kernel_size)
    SH, SW = _pair(stride)
    PH, PW = _pair(padding)

    H_out = (H + 2 * PH - KH) // SH + 1
    W_out = (W + 2 * PW - KW) // SW + 1

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("thnn_conv2d: unsupported dtype %s" % x.dtype)

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    block_w = 64 if W_out >= 64 else max(8, min(32, 1 << (max(1, int(W_out)) - 1).bit_length()))
    block_oc = min(32, 1 << (max(1, int(C_out)) - 1).bit_length())
    num_warps = 8 if (W_out >= 64 and C_in * KH * KW > 1) else 4
    oc_tiles = triton.cdiv(C_out, block_oc)
    grid = (triton.cdiv(W_out, block_w), H_out, N * oc_tiles)
    _thnn_fma_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        N,
        C_in,
        H,
        W,
        C_out,
        H_out,
        W_out,
        SH,
        SW,
        PH,
        PW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        HAS_BIAS=(bias is not None),
        BLOCK_W=block_w,
        BLOCK_OC=block_oc,
        OC_TILES=oc_tiles,
        KH=KH,
        KW=KW,
        num_warps=num_warps,
    )
    return out
