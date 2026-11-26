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

from flag_gems.runtime.backend._enflame.gcu400.ops.conv2d__opt import Conv2d, conv2d

logger = logging.getLogger(__name__)


# Forward kernel: 1-D implicit GEMM (M = N*OL, K = C_in/groups * KW,
# Ndim = C_out/groups).  Unlike the conv2d form this is a dedicated 1-D
# decode (no H-axis arithmetic) and the K loop can be fully unrolled with
# tl.static_range: conv1d's KVOL is the kernel width alone (3-15), so the
# per-iteration DSM intermediates (~30-50KB/iter) stay inside the budget
# where the 2-D/3-D siblings blow the 917504B limit at KVOL>=27.  Measured
# on the conv1d bench core shapes (Triton GCU 4.1.6, do_bench median,
# w1000/r100): static unroll wins 1.03-1.21x over the conv2d delegation on
# every shape whose unroll budget fits, while BK=64 loses 2.5-3x and
# BM=64/BN=16 tiles lose 10-60% (tile table reuses the conv2d-measured
# num_warps=1 everywhere).
@triton.jit
def conv1d_implicit_gemm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    C_in_pg,
    C_out_pg,
    OL,
    IL,
    KDIM: tl.constexpr,
    KW: tl.constexpr,
    S,
    P,
    DL,
    x_ns,
    x_cs,
    x_ls,
    w_os,
    w_cs,
    w_ks,
    out_ns,
    out_cs,
    out_ls,
    F32: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    bidx = offs_m // OL
    o = offs_m % OL
    # Clamp every masked-lane address component: GCU touches the address of
    # a masked load (unmasked-load+select lowering), so an unclamped index
    # reads a neighbouring allocation instead of contributing zeros.
    bidx_v = tl.where(mask_m, bidx, 0)

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < C_out_pg
    co_g = pid_g * C_out_pg + offs_n
    ocv = tl.where(mask_n, co_g, 0)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_off = tl.arange(0, BK)

    for kstart in tl.static_range(0, KDIM, BK):
        ks = kstart + k_off
        ci_loc = ks // KW
        kw = ks % KW
        # x channels are global (group offset), weight channels are local
        # to the group: two different clamped vectors.
        ci_g = pid_g * C_in_pg + ci_loc
        if EVEN_K:
            ci_loc_v = ci_loc
            ci_g_v = ci_g
        else:
            kmask = ks < KDIM
            ci_loc_v = tl.where(kmask, ci_loc, 0)
            ci_g_v = tl.where(kmask, ci_g, 0)
            kw = tl.where(kmask, kw, 0)

        ipos = o[:, None] * S - P + kw[None, :] * DL
        v = (ipos >= 0) & (ipos < IL)
        ipos_v = tl.where(v, ipos, 0)
        aoff = (bidx_v * x_ns)[:, None] + (ci_g_v * x_cs)[None, :] + ipos_v * x_ls
        amask = mask_m[:, None] & v
        if not EVEN_K:
            amask = amask & kmask[None, :]
        a = tl.load(x_ptr + aoff, mask=amask, other=0.0)

        woff = ci_loc_v[:, None] * w_cs + ocv[None, :] * w_os + kw[:, None] * w_ks
        bmask = mask_n[None, :]
        if not EVEN_K:
            bmask = bmask & kmask[:, None]
        b = tl.load(w_ptr + woff, mask=bmask, other=0.0)

        if F32:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        # One fp32 rounding at the store: keep the conv2d-owned bias
        # semantics (single rounding against the fp32-upcast reference).
        bv = tl.load(b_ptr + ocv, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bv[None, :]

    ooff = (bidx * out_ns)[:, None] + (co_g * out_cs)[None, :] + (o * out_ls)[:, None]
    smask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + ooff, acc.to(out_ptr.dtype.element_ty), mask=smask)


def _next_pow2(v, lo, hi):
    """Clamp ``v`` up to a power of two inside ``[lo, hi]``."""
    p = lo
    while p < v and p < hi:
        p *= 2
    return p


# Direct (im2col-free) form: every operand tile is a strided 2-D block, so the
# GCU DTE can move it as a bulk transfer instead of one gather per element.
# The kernel walks kw in an unrolled loop and the channel block inside it; the
# X tile is addressed as (L, C) with the L axis contiguous, which keeps the
# fast axis on dim 0 and lets ``boundary_check`` express both the zero padding
# and the tail masking (negative offsets are masked to zero as well).
@triton.jit
def conv1d_direct_dte_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C_in_pg,
    C_out_pg,
    OL,
    IL,
    KW: tl.constexpr,
    GROUPS: tl.constexpr,
    S,
    P,
    DL,
    x_ns,
    x_cs,
    x_ls,
    w_os,
    w_cs,
    w_ks,
    out_ns,
    out_cs,
    out_ls,
    F32: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BKC: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_ng = tl.program_id(2)
    img = pid_ng // GROUPS
    grp = pid_ng % GROUPS

    o0 = pid_o * BM
    co0 = grp * C_out_pg + pid_n * BN
    ci0 = grp * C_in_pg

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    offs_m = o0 + tl.arange(0, BM)
    offs_k = tl.arange(0, BKC)
    for kw in tl.range(0, KW):
        # Output row o reads input position o * S + shift, so the shift is
        # folded into the tile base and the tile row axis stays the fast axis.
        shift = kw * DL - P
        for c0 in tl.range(0, C_in_pg, BKC):
            if shift >= 0:
                # Rows is the number of output positions whose input window
                # start is still inside the input row.
                rows = (IL - shift + S - 1) // S
                a_bp = tl.make_block_ptr(
                    base=x_ptr + img * x_ns + (ci0 + c0) * x_cs + shift * x_ls,
                    shape=(rows, C_in_pg - c0),
                    strides=(S * x_ls, x_cs),
                    offsets=(o0, 0),
                    block_shape=(BM, BKC),
                    order=(0, 1),
                )
                a = tl.load(a_bp, boundary_check=(0, 1), padding_option="zero")
                # The block transfer clamps the tail rows instead of zero
                # filling them, so they are cleared explicitly.
                a = tl.where(
                    (offs_m * S + shift < IL)[:, None] & (offs_m < OL)[:, None],
                    a,
                    0.0,
                )
            else:
                # A block descriptor cannot start before the tensor, so the
                # left zero-padding border is fetched with an explicit mask
                # (this only happens for the taps that reach into the padding).
                ipos = offs_m * S + shift
                a_ptrs = (
                    x_ptr
                    + img * x_ns
                    + (ci0 + c0) * x_cs
                    + ipos[:, None] * x_ls
                    + offs_k[None, :] * x_cs
                )
                a_mask = (ipos >= 0)[:, None] & (ipos < IL)[:, None]
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            w_bp = tl.make_block_ptr(
                base=w_ptr + kw * w_ks + c0 * w_cs + co0 * w_os,
                shape=(C_in_pg - c0, C_out_pg - pid_n * BN),
                strides=(w_cs, w_os),
                offsets=(0, 0),
                block_shape=(BKC, BN),
                order=(0, 1),
            )
            b = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")

            if F32:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        offs_n = tl.arange(0, BN)
        bmask = offs_n + pid_n * BN < C_out_pg
        bv = tl.load(b_ptr + co0 + offs_n, mask=bmask, other=0.0).to(tl.float32)
        acc = acc + bv[None, :]

    o_bp = tl.make_block_ptr(
        base=out_ptr + img * out_ns + co0 * out_cs,
        shape=(OL, C_out_pg - pid_n * BN),
        strides=(out_ls, out_cs),
        offsets=(o0, 0),
        block_shape=(BM, BN),
        order=(0, 1),
    )
    tl.store(o_bp, acc.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


def _launch_conv1d_direct_dte(
    x, w, bias, stride_width, padding_width, dilation_width, groups
):
    N, C, L = x.shape
    C_out, C_in_pg, KW = w.shape
    C_out_pg = C_out // groups
    OL = (L + 2 * padding_width - dilation_width * (KW - 1) - 1) // stride_width + 1
    if OL <= 0 or C_out_pg <= 0 or C_in_pg <= 0:
        return None

    is_f32 = x.dtype == torch.float32
    tiles = _select_direct_tiles(OL, C_out_pg, C_in_pg, is_f32)
    if tiles is None:
        return None
    BM, BN, BKC = tiles

    out = torch.empty((N, C_out, OL), device=x.device, dtype=x.dtype)
    has_bias = bias is not None and int(bias.numel()) > 0
    b_ptr = bias if has_bias else x

    grid = (triton.cdiv(OL, BM), triton.cdiv(C_out_pg, BN), N * groups)
    conv1d_direct_dte_kernel[grid](
        x,
        w,
        b_ptr,
        out,
        C_in_pg,
        C_out_pg,
        OL,
        L,
        KW=KW,
        GROUPS=groups,
        S=stride_width,
        P=padding_width,
        DL=dilation_width,
        x_ns=x.stride(0),
        x_cs=x.stride(1),
        x_ls=x.stride(2),
        w_os=w.stride(0),
        w_cs=w.stride(1),
        w_ks=w.stride(2),
        out_ns=out.stride(0),
        out_cs=out.stride(1),
        out_ls=out.stride(2),
        F32=is_f32,
        HAS_BIAS=has_bias,
        BM=BM,
        BN=BN,
        BKC=BKC,
        num_warps=1,
    )
    return out


# Tile table for the direct form. The channel tiles must divide the channel
# counts exactly: a partially covered channel tile would need a boundary check
# on the slow axis of the block transfer, which the backend handles by clamping
# instead of zero filling (it corrupts the last covered lane).
def _pow2_divisor(value, lo, hi):
    p = lo
    best = None
    while p <= hi:
        if value % p == 0:
            best = p
        p *= 2
    return best


def _select_direct_tiles(OL, C_out_pg, C_in_pg, is_f32):
    if OL < 32 or OL * C_out_pg * C_in_pg < (1 << 16):
        return None
    bn = _pow2_divisor(C_out_pg, 16, 128)
    bkc = _pow2_divisor(C_in_pg, 16, 64)
    if bn is None or bkc is None:
        return None
    # A wide output-position tile amortises the per-block prologue; the tile is
    # then shrunk until the two operand tiles fit the shared-memory budget.
    bm = 512 if bkc == C_in_pg else 256
    if bm > OL:
        bm = _next_pow2(OL, 32, bm)
    itemsize = 4 if is_f32 else 2
    while bm > 64 and bm * bkc * itemsize + bkc * bn * 8 > 150000:
        bm //= 2
    return bm, bn, bkc


# Tile table for the 1-D implicit GEMM, reusing the conv2d-measured shapes
# (num_warps=1 everywhere: w2/w4 split the operand tiles across warps and
# serialize the dot issue path, 10-13x measured).  fp32 keeps the shallow
# BM=64 tile of the convolution op family; 16-bit keeps BM=128 with the
# masked BN half-tile costing nothing.
def _select_tiles(M, C_out_pg, KDIM, is_f32):
    if M * C_out_pg * KDIM < (1 << 20):
        return 32, 16, 32
    if is_f32:
        if KDIM <= 64:
            return 128, 64, 32
        return 64, 32, 32
    return 128, _next_pow2(C_out_pg, 32, 128), 32


# DSM budget for the static K unroll: each unrolled iteration materializes
# its own 2-D index/mask/feed buffers (~30-50KB, no cross-iteration reuse).
# Measured on the bench core shapes: 216KB fits and wins, 384KB (8 iters x
# (128+128)-wide tiles) pushes past the SP concurrency threshold and loses
# ~13%.  Over-budget shapes keep the conv2d delegation path, which measures
# level with the runtime-range 1-D form there.
_STATIC_UNROLL_DSM_BUDGET = 300 * 1024


def _launch_conv1d_implicit_gemm(
    x, w, bias, stride_width, padding_width, dilation_width, groups
):
    if not _fast_path_eligible(x, w, bias, groups):
        return None
    N, C, L = x.shape
    C_out, C_in_pg, KW = w.shape
    C_out_pg = C_out // groups
    OL = (L + 2 * padding_width - dilation_width * (KW - 1) - 1) // stride_width + 1
    M = N * OL
    KDIM = C_in_pg * KW

    out = torch.empty((N, C_out, OL), device=x.device, dtype=x.dtype)
    if M <= 0 or C_out_pg <= 0 or KDIM <= 0:
        return out

    has_bias = bias is not None and int(bias.numel()) > 0
    b_ptr = bias if has_bias else x

    is_f32 = x.dtype == torch.float32
    BM, BN, BK = _select_tiles(M, C_out_pg, KDIM, is_f32)
    iters = triton.cdiv(KDIM, BK)
    itemsize = x.element_size()
    dsm_est = iters * (BM * BK + BK * BN) * (4 + itemsize)
    if dsm_est > _STATIC_UNROLL_DSM_BUDGET:
        return None

    grid = (triton.cdiv(M, BM), triton.cdiv(C_out_pg, BN), groups)
    conv1d_implicit_gemm_kernel[grid](
        x,
        w,
        b_ptr,
        out,
        M,
        C_in_pg,
        C_out_pg,
        OL,
        L,
        KDIM,
        KW,
        stride_width,
        padding_width,
        dilation_width,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        F32=is_f32,
        HAS_BIAS=has_bias,
        EVEN_K=(KDIM % BK) == 0,
        BM=BM,
        BN=BN,
        BK=BK,
        num_warps=1,
    )
    return out


class _Conv2dBackwardShim:
    """Expose the 1-D ctx to ``Conv2d.backward`` as its 4-D equivalent."""

    def __init__(self, ctx):
        self._ctx = ctx

    @property
    def saved_tensors(self):
        weight, input, bias = self._ctx.saved_tensors
        return (weight.unsqueeze(-1), input.unsqueeze(-1), bias)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _select_conv1d_path(
    out_len, out_channels_pg, in_channels_pg, kernel_width, is_f32, stride_width
):
    """Choose the execution path that measured fastest for this shape family.

    The 4-D conv2d delegation runs a full matrix-engine implicit GEMM over
    (out positions x out channels) and wins on almost every benchmarked shape:
    measured 3.3x faster on average (geomean) than the dedicated 1-D forms,
    which spend their time streaming strided A tiles for a (K = in_channels *
    kernel_width) contraction. Two narrow shape families still prefer a
    dedicated kernel, both fp32-only:

    * in_channels_pg <= 8: conv2d pads the channel axis up to the matrix tile
      and wastes half of the contraction, so the 1-D implicit GEMM wins
      (measured 1.35x on (8, 8, 8192, 16, 11)).  A 12-channel group keeps
      conv2d (measured 1.40x the other way on (16, 24, 2048, 96, 7, groups=2)),
      so the cut is at 8 rather than at the 16-wide matrix tile.
    * kernel_width * in_channels_pg <= 128 with out_channels_pg <= 32: the
      contraction is short and narrow, so the position-tiled direct form
      amortises better than conv2d's channel padding (measured 1.47x on
      (16, 16, 1024, 32, 7)).

    fp16 shapes of the second family keep conv2d: there the 16-bit matrix path
    is already wide enough that the direct form loses (measured 0.88x).
    """
    if is_f32 and in_channels_pg <= 8:
        return "gemm"
    if (
        is_f32
        and stride_width == 1
        and out_channels_pg <= 32
        and in_channels_pg >= 16
        and kernel_width * in_channels_pg <= 128
        and 32 <= out_len <= 2048
    ):
        return "direct"
    return "conv2d"


class Conv1d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, stride_width, padding_width, dilation_width, groups
    ):
        logger.debug("GEMS_ENFLAME CONV1D")
        output = None
        if _fast_path_eligible(input, weight, bias, groups):
            _, _, in_len = input.shape
            out_channels, in_channels_pg, kernel_width = weight.shape
            out_len = (
                in_len + 2 * padding_width - dilation_width * (kernel_width - 1) - 1
            ) // stride_width + 1
            path = _select_conv1d_path(
                out_len,
                out_channels // groups,
                in_channels_pg,
                kernel_width,
                input.dtype == torch.float32,
                stride_width,
            )
            if path == "direct":
                output = _launch_conv1d_direct_dte(
                    input,
                    weight,
                    bias,
                    stride_width,
                    padding_width,
                    dilation_width,
                    groups,
                )
            elif path == "gemm":
                output = _launch_conv1d_implicit_gemm(
                    input,
                    weight,
                    bias,
                    stride_width,
                    padding_width,
                    dilation_width,
                    groups,
                )
        if output is None:
            output = conv2d(
                input.unsqueeze(-1),
                weight.unsqueeze(-1),
                bias,
                (stride_width, 1),
                (padding_width, 0),
                (dilation_width, 1),
                groups,
            ).squeeze(-1)

        C_out, C_in_pg, KW = weight.shape
        N, _, L = input.shape
        OL = output.shape[-1]

        # Save ctx in the layout Conv2d.backward expects.
        ctx.save_for_backward(weight, input, bias)
        ctx.stride = (stride_width, 1)
        ctx.padding = (padding_width, 0)
        ctx.dilation = (dilation_width, 1)
        # Convention of the conv1d -> conv2d delegation: L maps to the H
        # axis (input (N, C, L, 1), kernel (C_out, C_in_pg, KW, 1)), so the
        # ctx stores the H==L layout Conv2d.backward expects.
        ctx.weight_info = (C_out // groups, C_in_pg, KW, 1)
        ctx.input_info = (N, L, 1)
        ctx.out_info = (OL, 1)
        ctx.device = input.device
        ctx.groups = groups
        return output

    @staticmethod
    def backward(ctx, out_grad):
        shim = _Conv2dBackwardShim(ctx)
        input_back, weight_back, bias_grad, _, _, _, _ = Conv2d.backward(
            shim, out_grad.unsqueeze(-1)
        )
        input_back = input_back.squeeze(-1)
        weight_back = weight_back.squeeze(-1)
        return (input_back, weight_back, bias_grad, None, None, None, None)


# dtypes the dedicated kernel handles; everything else keeps the conv2d
# delegation (which owns the assert messages and the fp64 failure mode).
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _fast_path_eligible(input, weight, bias, groups):
    if input.dim() != 3 or weight.dim() != 3:
        return False
    if input.dtype not in _FAST_DTYPES or weight.dtype != input.dtype:
        return False
    C_in = input.shape[1]
    C_out, C_in_pg, _ = weight.shape
    if C_in != groups * C_in_pg or C_out % groups != 0:
        return False
    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != C_out or bias.dtype != input.dtype:
            return False
    return input.shape[-1] > 0


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_ENFLAME CONV1D")
    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(dilation, (list, tuple)):
        dilation_width = dilation[0]
    else:
        dilation_width = dilation

    if isinstance(padding, str):
        if padding == "same":
            assert stride == 1, "Doesn't support any stride values other than 1 \
                in padding = 'same' mode, received stride value {stride}"
            il = input.shape[-1]
            kernel_size = weight.shape[-1]
            padding_width = math.ceil(
                (stride_width * (il - 1) + 1 + dilation_width * (kernel_size - 1) - il)
                / 2
            )
            ol = int(
                (il + 2 * padding_width - dilation_width * (kernel_size - 1) - 1)
                / stride_width
                + 1
            )
            out = Conv1d.apply(
                input, weight, bias, stride_width, padding_width, dilation_width, groups
            )
            return out[..., (ol - il) :]
        elif padding == "valid":
            padding_width = 0
        else:
            raise ValueError(
                f"Unsupported padding mode: {padding}, only 'valid' or 'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding
    return Conv1d.apply(
        input, weight, bias, stride_width, padding_width, dilation_width, groups
    )
