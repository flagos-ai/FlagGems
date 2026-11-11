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

import torch
import triton
import triton.language as tl

# Bounded search over per-thread output elements (BLOCK / threads) so each
# thread can keep several independent tap-load streams in flight.  The kernel
# body, masks and fp32 accumulation order are identical for every config.
_CONFIGS = [
    triton.Config({"BLOCK": 64}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
]

_AUTOTUNE_KEY = [
    "C_OUT",
    "outHW",
    "H",
    "W",
    "outW",
    "KH",
    "KW",
    "SH",
    "SW",
    "PH",
    "PW",
    "DH",
    "DW",
    "MULT",
    "HAS_BIAS",
]

# Narrow blocks for the flat variant: when a whole output plane fits in one
# block, the only way to use the machine is to pack many planes per block.
_CONFIGS_FLAT = [
    triton.Config({"BLOCK": 64}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _conv_depthwise2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    C_OUT,
    outHW,
    H,
    W,
    outW,
    sn,
    sc,
    sh,
    sw,
    on,
    oc,
    oh_s,
    ow_s,
    wc,
    wk,
    wl,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    MULT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    blocks_per_plane = tl.cdiv(outHW, BLOCK)
    pid = tl.program_id(0)
    sb = pid % blocks_per_plane
    nc = pid // blocks_per_plane
    n = nc // C_OUT
    co = nc % C_OUT
    ci = co // MULT

    offs = sb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < outHW
    oh = offs // outW
    ow = offs - oh * outW

    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    x_base = x_ptr + n.to(tl.int64) * sn + ci.to(tl.int64) * sc

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kh in tl.static_range(KH):
        ih = ih0 + kh * DH
        mh = mask & (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = iw0 + kw * DW
            m = mh & (iw >= 0) & (iw < W)
            xv = tl.load(x_base + ih * sh + iw * sw, mask=m, other=0.0)
            wv = tl.load(w_ptr + co * wc + kh * wk + kw * wl)
            acc += xv.to(tl.float32) * wv.to(tl.float32)

    if HAS_BIAS:
        acc += tl.load(b_ptr + co).to(tl.float32)

    o_base = o_ptr + n.to(tl.int64) * on + co.to(tl.int64) * oc
    tl.store(o_base + oh * oh_s + ow * ow_s, acc.to(o_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_CONFIGS_FLAT, key=_AUTOTUNE_KEY + ["total"])
@triton.jit
def _conv_depthwise2d_flat(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    C_OUT,
    outHW,
    total,
    H,
    W,
    outW,
    sn,
    sc,
    sh,
    sw,
    wc,
    wk,
    wl,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    MULT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    plane = offs // outHW
    sp = offs - plane * outHW
    n = plane // C_OUT
    co = plane % C_OUT
    ci = co // MULT

    oh = sp // outW
    ow = sp - oh * outW
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    x_base = x_ptr + n.to(tl.int64) * sn + ci.to(tl.int64) * sc

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kh in tl.static_range(KH):
        ih = ih0 + kh * DH
        mh = mask & (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = iw0 + kw * DW
            m = mh & (iw >= 0) & (iw < W)
            xv = tl.load(x_base + ih * sh + iw * sw, mask=m, other=0.0)
            wv = tl.load(w_ptr + co * wc + kh * wk + kw * wl)
            acc += xv.to(tl.float32) * wv.to(tl.float32)

    if HAS_BIAS:
        acc += tl.load(b_ptr + co).to(tl.float32)

    tl.store(o_ptr + offs, acc.to(o_ptr.dtype.element_ty), mask=mask)


def _to_pair(v):
    if isinstance(v, int):
        return (int(v), int(v))
    seq = list(v)
    if len(seq) == 1:
        return (int(seq[0]), int(seq[0]))
    if len(seq) != 2:
        raise RuntimeError(
            "Expected 2 elements in the parameter list, but got {}".format(len(seq))
        )
    return (int(seq[0]), int(seq[1]))


def _to_kernel_size(v):
    # kernel_size must have two entries and must agree with the weight spatial
    # size; the actual kernel shape is taken from weight.shape[2:].
    if isinstance(v, int):
        return (int(v), int(v))
    seq = list(v)
    if len(seq) != 2:
        raise RuntimeError(
            "Expected kernel_size.size() == 2 to be true, but got false.  "
            "(Could this error message be improved?  If so, please report an "
            "enhancement request to PyTorch.)"
        )
    return (int(seq[0]), int(seq[1]))


def _conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation):
    x = self
    if x.numel() == 0 or x.dim() != 4:
        raise RuntimeError(
            "Expected input.numel() > 0 && input.dim() == 4 to be true, but got false.  "
            "(Could this error message be improved?  If so, please report an "
            "enhancement request to PyTorch.)"
        )
    if weight.dim() != 4:
        raise RuntimeError("Expected 4D weight (got {}D weight)".format(weight.dim()))

    ksize = _to_kernel_size(kernel_size)
    if ksize[0] != weight.shape[2] or ksize[1] != weight.shape[3]:
        raise RuntimeError(
            "Expected kernel size {} to match weight spatial size {}".format(
                ksize, (weight.shape[2], weight.shape[3])
            )
        )
    if weight.shape[1] != 1:
        raise RuntimeError(
            "Expected weight to have 1 input channel per group, got {}".format(
                weight.shape[1]
            )
        )

    stride_hw = _to_pair(stride)
    pad_hw = _to_pair(padding)
    dil_hw = _to_pair(dilation)

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    KH = weight.shape[2]
    KW = weight.shape[3]
    if KH <= 0 or KW <= 0:
        raise RuntimeError(
            "Expected kernel size to be positive, got {}".format((KH, KW))
        )
    SH, SW = stride_hw
    PH, PW = pad_hw
    DH, DW = dil_hw

    if C_in == 0 or C_out == 0:
        raise RuntimeError("Expected non-zero channels")
    if C_out % C_in != 0:
        raise RuntimeError(
            "Expected out_channels {} to be a multiple of in_channels {}".format(
                C_out, C_in
            )
        )
    MULT = C_out // C_in

    outH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    outW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
    if outH <= 0 or outW <= 0:
        raise RuntimeError(
            "Calculated output size: ({}x{}x{}x{}). Output size is too small".format(
                N, C_out, outH, outW
            )
        )

    if bias is not None and bias.numel() != C_out:
        raise RuntimeError(
            "Expected bias to have {} elements, got {}".format(C_out, bias.numel())
        )

    out = torch.empty((N, C_out, outH, outW), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    outHW = outH * outW
    n_planes = N * C_out

    sn, sc, sh, sw = x.stride()
    on, oc, oh_s, ow_s = out.stride()
    wc, _, wk, wl = weight.stride()

    b_ptr = bias if bias is not None else x
    total = n_planes * outHW
    # A plane that fits in the narrowest block leaves most lanes of a per-plane
    # block idle; packing whole planes across the grid is worth it only when it
    # reduces the block count by at least 2x.
    if n_planes * triton.cdiv(outHW, 64) >= 2 * triton.cdiv(total, 64):
        flat_grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)
        _conv_depthwise2d_flat[flat_grid](
            x,
            weight,
            b_ptr,
            out,
            C_out,
            outHW,
            total,
            H,
            W,
            outW,
            sn,
            sc,
            sh,
            sw,
            wc,
            wk,
            wl,
            KH=KH,
            KW=KW,
            SH=SH,
            SW=SW,
            PH=PH,
            PW=PW,
            DH=DH,
            DW=DW,
            MULT=MULT,
            HAS_BIAS=bias is not None,
        )
        return out

    def grid(meta):
        return (n_planes * triton.cdiv(outHW, meta["BLOCK"]),)

    _conv_depthwise2d_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        C_out,
        outHW,
        H,
        W,
        outW,
        sn,
        sc,
        sh,
        sw,
        on,
        oc,
        oh_s,
        ow_s,
        wc,
        wk,
        wl,
        KH=KH,
        KW=KW,
        SH=SH,
        SW=SW,
        PH=PH,
        PW=PW,
        DH=DH,
        DW=DW,
        MULT=MULT,
        HAS_BIAS=bias is not None,
    )
    return out
