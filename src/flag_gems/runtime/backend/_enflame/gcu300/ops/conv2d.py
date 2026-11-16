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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Fixed-grid SIP budget (same as conv1d): grid = cdiv(MAX_GRID_DIM, num_warps);
# with num_warps == 4 this is 6 programs x 4 warps == 24 warps, exactly
# saturating the 24 SIPs of GCU300.  A direct sweep of grid strides
# {6,12,24,48} (MAX_GRID_DIM 24/48/96/192) on b2/b3/b5/b6 confirmed the
# runtime is FLAT -- conv2d is DMA-descriptor-overhead-bound (each output tile
# issues KH*KW*Cg strided row transfers), not parallelism-bound, so larger
# grids neither help nor hurt.  MAX_GRID_DIM stays 24 (matches conv1d).
MAX_GRID_DIM = 24
# Host guard on the left-padding edge width (in OUTPUT columns) the fast kernel
# handles; larger padding falls back to the generic path.  Benchmark paddings
# are <= 2; 8 leaves headroom for realistic cases.
BLOCK_EDGE = 8


# ---------------------------------------------------------------------------
# im2col-free 2D convolution written as a batched GEMM per output row:
#
#     out[n, oc, oh, ow] =
#         sum_{kh, kw, ci} w2[oc, kh, kw, ci] *
#             x[n, ci, oh + kh*DH - PH_top, ow + kw*DW - PW_left]
#
# (fast path: stride == 1; the requested stride/dilation asymmetry is absorbed
# into PH_top/PW_left).  M = OCg (weight A tile), N = a contiguous run of ow
# within ONE output row (input B tile -- unit stride along W rides the DMA fast
# path), reduction = (kh, kw, ci) walked as nested runtime loops.
#
# The design reuses every measured GCU300 fact from the conv1d kernel:
#
#   1. ``tl.make_block_ptr`` rides the DMA fast path; explicit-pointer loads of
#      the same tile are ~15-20x slower.  Weight (A), input row window (B) and
#      output all use block_ptr in the main region.
#   2. Negative block_ptr offsets are unreliable (``boundary_check`` only guards
#      the upper bound; small tensors silently read garbage, large ones raise a
#      hardware ``Sdte`` fault).  Eliminated *by construction*:
#        - vertically: the input row ``r = oh + kh*DH - PH_top`` is a runtime
#          SCALAR; taps whose row falls outside [0, IH) contribute zero, so the
#          whole (kh) iteration is skipped by a scalar branch -- no negative
#          base pointer is ever formed;
#        - horizontally: **shifted tiling** -- main tile j covers output columns
#          ``ow in [OL0c + j*BLOCK_N, ...)`` where ``OL0c == PW_left`` (stride
#          1), so the load offset ``j*BLOCK_N + kw*DW >= 0`` always.  Right
#          padding is an upper-bound overflow which boundary_check zero-fills.
#   3. A block_ptr whose fast dimension has stride > 1 corrupts memory, hence
#      the stride==1 fast-path guard (stride>1 falls back to the generic
#      kernel for now).
#
# The leading ``OL0c = PW_left`` output columns of every row form a thin edge
# region the shifted main tiling cannot reach.  It is computed with a
# matrix-vector reduction and NO ``tl.dot`` -- the conv1d-proven fix
# (experience #22) transplanted to 2D.  A second ``tl.dot`` in this kernel is
# "poisoned" on GCU300: every gather-dot edge variant (both M/N orientations,
# block_ptr or gathered operands, N up to 128) corrupted every accumulator
# M-lane past the first ~4, bit-identically, even though the identical code is
# correct as a standalone kernel.  The matrix-vector edge instantiates exactly
# ONE MMA config (the main region's), the same structure conv1d ships.  One
# tile covers one (n, oh, g) and BLOCK_M channels; the OL0c edge columns are
# the BLOCK_E = next_pow2(OL0c) accumulator lanes (host-sized, <= BLOCK_EDGE).
#
# Both regions live in ONE kernel: a single launch per conv2d call, and no
# large-tensor PyTorch work on the forward path (only the tiny weight is
# permuted, OC*Cg*KH*KW elements, to make the A tile row-major for the MMA).
# ---------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conv2d"),
    # KH/KW (tap loop trip counts), Cg/OCg (reduction/M widths), OL0c (edge
    # width -> HAS_EDGE/BLOCK_E) and G all change the code the kernel actually
    # runs, so they use the exact ("default") strategy; N/OH/OW are pure size
    # dims and may be log-bucketed.  OL0c can be 0 -> "log" would be undefined
    # (lesson #24 from conv1d: libtuner log buckets collided pad=0/3/6 shapes
    # and cost 1.6x).
    key=["N", "OH", "OW", "OCg", "Cg", "KH", "KW", "OL0c", "G"],
    strategy=[
        "log",
        "log",
        "log",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
)
@triton.jit
def conv2d_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    N,
    OH,
    OW,
    OCg,
    G,
    IH,
    IW,
    PH_top: tl.constexpr,
    PW_left: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oc: tl.constexpr,
    stride_oh: tl.constexpr,
    Cg: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    OL0c: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_EDGE: tl.constexpr,
    GRID_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_num = tl.cdiv(GRID_DIM, num_warps)
    grid_m = tl.cdiv(OCg, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    n_kblocks = tl.cdiv(Cg, BLOCK_K)

    # ===================== main region (shifted tiling) =====================
    # tile linearization: tile = (((n*G + g)*OH + oh)*grid_n + pid_n)*grid_m
    #                        + pid_m
    grid_n = tl.cdiv(OW - OL0c, BLOCK_N)
    total = N * G * OH * grid_n * grid_m
    for tile in tl.range(pid, total, grid_num):
        pid_m = tile % grid_m
        t1 = tile // grid_m
        pid_n = t1 % grid_n
        t2 = t1 // grid_n
        oh = t2 % OH
        ng = t2 // OH
        n = ng // G
        g = ng % G

        # load offset == ow0 + kw*DW - PW_left == pid_n*BLOCK_N + kw*DW  (>= 0)
        ow0 = OL0c + pid_n * BLOCK_N
        x_base = x_ptr + n * stride_xn + g * Cg * stride_xc
        w_base = w_ptr + g * OCg * stride_wm
        o_base = o_ptr + n * stride_on + g * OCg * stride_oc + oh * stride_oh

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kh in range(0, KH):
            r = oh + kh * DH - PH_top
            # vertical padding: the whole tap row contributes zero.  Scalar
            # branch -- never forms a negative base pointer (lesson: negative
            # block_ptr offsets fault / read garbage on GCU300).
            if (r >= 0) & (r < IH):
                xb = x_base + r * stride_xh
                for kw in range(0, KW):
                    for ci0 in range(0, n_kblocks):
                        a_block_ptr = tl.make_block_ptr(
                            base=w_base + (kh * KW + kw) * Cg,
                            shape=(OCg, Cg),
                            strides=(stride_wm, 1),
                            offsets=(pid_m * BLOCK_M, ci0 * BLOCK_K),
                            block_shape=(BLOCK_M, BLOCK_K),
                            order=(1, 0),
                        )
                        a = tl.load(
                            a_block_ptr, boundary_check=(0, 1), padding_option="zero"
                        )
                        b_block_ptr = tl.make_block_ptr(
                            base=xb,
                            shape=(Cg, IW),
                            strides=(stride_xc, 1),
                            offsets=(ci0 * BLOCK_K, pid_n * BLOCK_N + kw * DW),
                            block_shape=(BLOCK_K, BLOCK_N),
                            order=(1, 0),
                        )
                        b = tl.load(
                            b_block_ptr, boundary_check=(0, 1), padding_option="zero"
                        )
                        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

        if HAS_BIAS:
            bias_block_ptr = tl.make_block_ptr(
                base=b_ptr + g * OCg,
                shape=(OCg,),
                strides=(1,),
                offsets=(pid_m * BLOCK_M,),
                block_shape=(BLOCK_M,),
                order=(0,),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(0,), padding_option="zero")
            acc += bias.to(tl.float32)[:, None]

        o_block_ptr = tl.make_block_ptr(
            base=o_base,
            shape=(OCg, OW),
            strides=(stride_oc, 1),
            offsets=(pid_m * BLOCK_M, ow0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))

    # ======= edge region: output columns [0, OL0c) of every (n, g, oh) =======
    # The shifted main tiling cannot reach the leading OL0c = PW_left output
    # columns (their input window starts before x[..., 0]).  They are computed
    # here with a matrix-vector reduction and NO ``tl.dot`` -- the conv1d-proven
    # fix (experience #22) transplanted to 2D.
    #
    # Why not a dot: a second ``tl.dot`` in this kernel is "poisoned" on GCU300.
    # Every gather-dot variant was tried for the edge -- M=pixels/N=channels
    # (the generic kernel's orientation) and M=channels/N=pixels (the main
    # region's orientation), each with block_ptr or gathered operands, with
    # ``other=``/``out_dtype=``/``num_stages=1`` permutations -- and ALL
    # corrupted every accumulator M-lane past the first ~4 (oc 0..3 correct,
    # oc 4.. garbage; n>=1 denormal ~7.8e-41), bit-identically, even at N=128
    # (>64) and even though the identical edge code is correct as a standalone
    # kernel.  The fault is the gathered B operand sharing the kernel with the
    # main region's block_ptr DMA dot: the MMA only computes ~4 M-rows.  The
    # matrix-vector edge has no dot at all, so the kernel instantiates exactly
    # ONE MMA config (the main region's) -- the same structure conv1d ships.
    #
    # One tile covers ONE (n, oh, g) and BLOCK_M output channels; the OL0c edge
    # columns are the BLOCK_E accumulator columns (BLOCK_E = next_pow2(OL0c),
    # host-sized, <= BLOCK_EDGE so at most half the lanes are surplus).  For each
    # contributing tap the body does a strided vector load xv[ci] = x[n,ci,r,wpos]
    # (r, wpos are runtime SCALARS here -- oh is fixed per tile, ol is unrolled)
    # broadcast against the (BLOCK_M, BLOCK_K) weight tile and reduced with
    # ``tl.sum`` in fp32, then scattered into acc_e[:, ol].  Taps whose row r
    # falls outside [0, IH) are skipped by a scalar branch (never forms a
    # negative pointer); taps whose wpos < 0 are eliminated by a CONSTEXPR loop
    # start _kw0 = ceil((PW_left - ol)/DW) (experience #23), so non-contributing
    # bodies are never generated.
    if HAS_EDGE:
        offs_e = tl.arange(0, BLOCK_E)
        total_e = N * OH * G * grid_m
        for tile in tl.range(pid, total_e, grid_num):
            pid_m = tile % grid_m
            t1e = tile // grid_m
            g = t1e % G
            noh = t1e // G
            oh = noh % OH
            n = noh // OH

            x_base = x_ptr + n * stride_xn + g * Cg * stride_xc
            w_base = w_ptr + g * OCg * stride_wm
            o_base = o_ptr + n * stride_on + g * OCg * stride_oc + oh * stride_oh
            oc = pid_m * BLOCK_M + offs_m
            oc_ok = oc < OCg

            acc_e = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
            for ol in tl.static_range(OL0c):
                # first kw whose wpos = ol + kw*DW - PW_left >= 0 (constexpr).
                _kw0raw = (PW_left - ol + DW - 1) // DW
                _kw0 = _kw0raw if _kw0raw > 0 else 0
                for kh in range(0, KH):
                    r = oh + kh * DH - PH_top
                    # vertical padding: skip the whole tap row by scalar branch
                    # (never forms a negative base pointer).
                    if (r >= 0) & (r < IH):
                        for kw in range(_kw0, KW):
                            wpos = ol + kw * DW - PW_left
                            for ci0 in range(0, n_kblocks):
                                a_block_ptr = tl.make_block_ptr(
                                    base=w_base + (kh * KW + kw) * Cg,
                                    shape=(OCg, Cg),
                                    strides=(stride_wm, 1),
                                    offsets=(pid_m * BLOCK_M, ci0 * BLOCK_K),
                                    block_shape=(BLOCK_M, BLOCK_K),
                                    order=(1, 0),
                                )
                                a = tl.load(
                                    a_block_ptr,
                                    boundary_check=(0, 1),
                                    padding_option="zero",
                                )
                                ci = ci0 * BLOCK_K + offs_k
                                ci_ok = ci < Cg
                                # wpos >= 0 is guaranteed by _kw0; clamp only the
                                # upper bound and mask the (rare) wpos >= IW tap.
                                wpos_c = tl.minimum(wpos, IW - 1)
                                xv = tl.load(
                                    x_base + ci * stride_xc + r * stride_xh + wpos_c,
                                    mask=ci_ok & (wpos < IW),
                                    other=0.0,
                                )
                                col = tl.sum(
                                    a.to(tl.float32) * xv[None, :].to(tl.float32),
                                    axis=1,
                                )
                                acc_e += tl.where(
                                    offs_e[None, :] == ol, col[:, None], 0.0
                                )

            if HAS_BIAS:
                bias = tl.load(b_ptr + g * OCg + oc, mask=oc_ok, other=0.0)
                acc_e += bias.to(tl.float32)[:, None]

            o_ptrs = o_base + oc[:, None] * stride_oc + offs_e[None, :]
            tl.store(
                o_ptrs,
                acc_e.to(o_ptr.dtype.element_ty),
                mask=oc_ok[:, None] & (offs_e < OL0c)[None, :],
            )


def _resolve_padding(padding, dilation_h, dilation_w, kernel_h, kernel_w):
    """Return ``(pad_top, pad_bottom, pad_left, pad_right)`` (stride==1).

    Matches torch's "same" convention: total pad per dim is
    ``dilation*(kernel-1)`` and the odd remainder goes to the bottom/right.
    """
    if isinstance(padding, str):
        if padding == "same":
            tot_h = dilation_h * (kernel_h - 1)
            tot_w = dilation_w * (kernel_w - 1)
            return tot_h // 2, tot_h - tot_h // 2, tot_w // 2, tot_w - tot_w // 2
        elif padding == "valid":
            return 0, 0, 0, 0
        raise ValueError(
            f"Unsupported padding mode: {padding}, only 'valid' or 'same' are allowed."
        )
    if isinstance(padding, (list, tuple)):
        padding_h, padding_w = padding
    else:
        padding_h = padding_w = padding
    return padding_h, padding_h, padding_w, padding_w


def _conv2d_forward(
    input,
    weight,
    bias,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
    dilation_h,
    dilation_w,
    groups,
):
    batch, _, in_h, in_w = input.shape
    out_c, in_c_g, kernel_h, kernel_w = weight.shape
    out_c_g = out_c // groups
    out_h = in_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1)
    out_w = in_w + pad_left + pad_right - dilation_w * (kernel_w - 1)
    # output columns whose input window starts before x[..., 0]; the shifted
    # main tiling cannot reach them, the edge loop covers exactly these.
    edge_cols = min(pad_left, out_w)

    # (OC, KH, KW, Cg) makes the weight A-tile row-major (ci contiguous) for
    # the MMA.  The weight is tiny, so this host-side reshape is not a
    # large-tensor op.
    w = weight.permute(0, 2, 3, 1).contiguous()
    out = torch.empty(
        (batch, out_c, out_h, out_w), dtype=input.dtype, device=input.device
    )
    has_bias = bias is not None
    b = bias.contiguous() if has_bias else weight

    edge_pix = batch * out_h * edge_cols  # per group

    def _total_tiles(meta):
        grid_m = triton.cdiv(out_c_g, meta["BLOCK_M"])
        grid_n = triton.cdiv(max(out_w - edge_cols, 0), meta["BLOCK_N"])
        main = batch * groups * out_h * grid_n * grid_m
        if edge_pix > 0:
            # matrix-vector edge: one tile per (n, oh, g, m-tile); the OL0c
            # edge columns are the BLOCK_E accumulator lanes of that tile.
            return main + batch * out_h * groups * grid_m
        return main

    grid = lambda META: (  # noqa: E731
        min(triton.cdiv(MAX_GRID_DIM, META["num_warps"]), _total_tiles(META)),
    )
    with torch_device_fn.device(input.device):
        conv2d_fwd_kernel[grid](
            input,
            w,
            b,
            out,
            batch,
            out_h,
            out_w,
            out_c_g,
            groups,
            in_h,
            in_w,
            pad_top,
            pad_left,
            stride_xn=input.stride(0),
            stride_xc=input.stride(1),
            stride_xh=input.stride(2),
            stride_wm=w.stride(0),
            stride_on=out.stride(0),
            stride_oc=out.stride(1),
            stride_oh=out.stride(2),
            Cg=in_c_g,
            KH=kernel_h,
            KW=kernel_w,
            DH=dilation_h,
            DW=dilation_w,
            OL0c=edge_cols,
            HAS_BIAS=has_bias,
            HAS_EDGE=edge_pix > 0,
            GRID_DIM=MAX_GRID_DIM,
            BLOCK_E=max(triton.next_power_of_2(edge_cols), 1),
        )
    return out


class _Conv2dFn(torch.autograd.Function):
    """Fast Triton forward; backward reuses the generic differentiable path."""

    @staticmethod
    def forward(ctx, input, weight, bias, pads, dilation_h, dilation_w, groups):
        out = _conv2d_forward(
            input, weight, bias, *pads, dilation_h, dilation_w, groups
        )
        ctx.save_for_backward(input, weight, bias)
        ctx.meta = (pads, dilation_h, dilation_w, groups)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from flag_gems.ops.conv2d import conv2d as conv2d_default

        input, weight, bias = ctx.saved_tensors
        pads, dilation_h, dilation_w, groups = ctx.meta
        with torch.enable_grad():
            i2 = input.detach().requires_grad_(True)
            w2 = weight.detach().requires_grad_(True)
            b2 = bias.detach().requires_grad_(True) if bias is not None else None
            # materialize the (possibly asymmetric) padding, then run the
            # generic differentiable conv2d with padding=0.  F.pad is
            # differentiable, so grads slice back to i2 automatically.  This is
            # the BACKWARD path -- the forward fast path stays single-launch
            # with no large-tensor Torch work.
            i_pad = torch.nn.functional.pad(i2, (pads[2], pads[3], pads[0], pads[1]))
            ref = conv2d_default(i_pad, w2, b2, 1, 0, (dilation_h, dilation_w), groups)
            inputs = [i2, w2] if b2 is None else [i2, w2, b2]
            grads = torch.autograd.grad(ref, inputs, grad_output)
        grad_bias = None if bias is None else grads[2]
        return grads[0], grads[1], grad_bias, None, None, None, None


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_ENFLAME_GCU300 CONV2D")
    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride
    if isinstance(dilation, (list, tuple)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    if isinstance(padding, str) and padding == "same":
        assert stride == 1, (
            "Doesn't support any stride values other than 1 "
            f"in padding = 'same' mode, received stride value {stride}"
        )

    if input.dim() == 3:
        out = conv2d(
            input.unsqueeze(0), weight, bias, stride, padding, dilation, groups
        )
        return out.squeeze(0)

    pad_top, pad_bottom, pad_left, pad_right = _resolve_padding(
        padding, dilation_h, dilation_w, weight.shape[-2], weight.shape[-1]
    )

    out_c, in_c_g, kernel_h, kernel_w = weight.shape
    in_h, in_w = input.shape[-2], input.shape[-1]
    out_h = in_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1)
    out_w = in_w + pad_left + pad_right - dilation_w * (kernel_w - 1)
    supported = (
        input.dim() == 4
        and weight.dim() == 4
        and out_h > 0
        and out_w > 0
        and in_c_g * groups == input.shape[1]
        and out_c % groups == 0
        and pad_top >= 0
        and pad_bottom >= 0
        and pad_left >= 0
        and pad_right >= 0
        # v1 fast path: stride 1 only.  stride>1 needs the conv1d-style
        # full-span compute + tl.split down-sample on BOTH axes; until then the
        # generic kernel handles it (accuracy-proven 72/72 on GCU300).
        and stride_h == 1
        and stride_w == 1
        # the matrix-vector edge holds the OL0c edge columns in a BLOCK_E =
        # next_pow2(pad_left) accumulator; keep it small so the per-tile
        # div/mod decomposition and the scatter store stay cheap.
        and pad_left <= BLOCK_EDGE
        # the DMA fast path needs a unit-stride width axis
        and input.stride(3) == 1
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        # int32 offset safety: max linear offset == N * stride_xn
        and input.numel() < 2**31
        and out_h * out_w * out_c * input.shape[0] < 2**31
    )
    if not supported:
        from flag_gems.ops.conv2d import conv2d as conv2d_default

        return conv2d_default(input, weight, bias, stride, padding, dilation, groups)

    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return _Conv2dFn.apply(
            input,
            weight,
            bias,
            (pad_top, pad_bottom, pad_left, pad_right),
            dilation_h,
            dilation_w,
            groups,
        )
    return _conv2d_forward(
        input,
        weight,
        bias,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        dilation_h,
        dilation_w,
        groups,
    )
