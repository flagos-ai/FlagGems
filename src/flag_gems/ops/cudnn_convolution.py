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

import contextlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv1d import conv1d
from flag_gems.ops.conv2d import conv2d
from flag_gems.ops.conv3d import conv3d

try:
    from triton.knobs import autotuning as _autotuning_knobs
except (ImportError, ModuleNotFoundError):
    # Triton < 3.6 has no triton.knobs module, and so no AABS either.
    _autotuning_knobs = None

logger = logging.getLogger(__name__)

_UNSET = object()

# Whether the running Triton fork ships FlagTree's AABS knob. Probed once at
# import: it is a property of the installed fork, not of any individual call.
_AABS_PRESENT = _autotuning_knobs is not None and hasattr(
    _autotuning_knobs, "adjust_block_size"
)


@contextlib.contextmanager
def _aabs_disabled():
    """Hold FlagTree AABS off for the duration of an operator dispatch.

    AABS (auto_adjust_block_sizes) rewrites the autotuned block sizes at launch
    time, shrinking any BLOCK that exceeds the tensor extent it was paired with
    to next_power_of_2(extent). It pairs a block to an extent by matching the
    ``tl.cdiv(extent, BLOCK)`` it sees against the kernel's argument names, and
    that pairing is wrong for every convolution kernel this operator launches:
    they tile the *flattened* spatial volume
    ``in_n * out_d * out_h * out_w``, but the analyzer keys the block to ``in_n``
    alone, so the tile collapses to next_power_of_2(in_n). Measured on the
    shipped kernels, AABS on -> off::

        (16, 32, 56, 56)    k3 g32  fp16     0.165 -> 1.622   (9.9x)
        (2, 4, 16, 16, 16)  k3      fp32     0.048 -> 0.945   (19.7x)
        (8, 256, 64, 64)    k3      fp16     0.038 -> 0.307   (8.1x)
        (32, 64, 128, 128)  k3      fp32     0.229 -> 0.631   (2.8x)
        (16, 32, 1024)      k3 g32  fp16     0.138 -> 0.330   (2.4x)
        (8, 3, 224, 224)    k3      fp16     1.083 -> 1.103   (unchanged)

    The collapse is fatal rather than merely slow on backends whose tensor-core
    tiles need M >= 16 rows: the shrunk M = 2 dot cannot be lowered and the
    kernel aborts while lowering the dot result's ``ttg.convert_layout``,
    surfacing as "RuntimeError: PassManager::run failed" from the mlir->llir
    stage. On the 3D core case (2, 16, 16, 16, 16) ``BLOCK_NI_DO_HO_WO`` goes
    512 -> 2 that way. Only some backends have a branch in
    triton/runtime/adjust_kernel_param.py that bumps such a tile back up to the
    dot floor, so on the others whatever AABS shrinks stays shrunk.

    The pairing lives in the Triton fork, not in any backend, which makes this a
    property of *this operator's kernels* rather than of a platform -- hence the
    guard below is a capability probe on the knob, never a check on which backend
    is running. Vanilla Triton >= 3.6 ships ``triton.knobs`` but no AABS at all,
    and there the probe short-circuits.

    The knob is global state read at launch time, so holding it off around the
    whole dispatch is what covers every kernel the operator launches -- including
    the nested conv1d/conv2d/conv3d autotuners and conv2d's backward pass, which
    passes ``out_grad.shape[0]`` as the same ``in_n`` argument. The save/restore
    keeps every other operator's autotuning untouched.
    """
    if not _AABS_PRESENT:
        # No AABS to hold off: Triton < 3.6 has no knobs module, and vanilla
        # Triton >= 3.6 has the module but not the knob. Callers that launch
        # kernels in bulk skip the manager outright (see the dispatch below).
        yield
        return

    # The knob is a descriptor whose __set__ also writes os.environ on every
    # store, making a set/restore pair ~5.6us; its __get__ prefers the instance
    # dictionary, so writing there instead is enough (~0.3us) and is what the
    # autotuner reads. Any FLAGTREE_AABS in the environment is left alone and
    # takes over again as soon as the entry is removed.
    state = _autotuning_knobs.__dict__
    previous = state.get("adjust_block_size", _UNSET)
    state["adjust_block_size"] = False
    try:
        yield
    finally:
        if previous is _UNSET:
            state.pop("adjust_block_size", None)
        else:
            state["adjust_block_size"] = previous


# ---------------------------------------------------------------------------
# Optimized paths, self-contained inside this operator.
#
# The generic conv1d/conv2d/conv3d kernels are an implicit-GEMM that tiles the
# output as (spatial, out_channel) and reduces over (in_channel * kernel). Three
# cases are disproportionately slow with that strategy and get dedicated
# kernels here:
#
#   * depthwise (groups == C_in, 1D/2D): the generic path pads the per-group
#     channel (== 1) up to 16 to satisfy tl.dot's K >= 16, wasting 15/16 of the
#     matmul FLOPs and adding a tensor copy. A depthwise convolution has no
#     cross-channel reduction at all, so a plain elementwise
#     multiply-accumulate kernel is strictly cheaper and removes the padding.
#
#   * 1x1 / pointwise (groups == 1, no padding): a pointwise convolution is a
#     plain GEMM (weight: OC x C_in) x (input: C_in x spatial) batched over N,
#     so it is implemented as a tiled matmul rather than routed through the
#     generic implicit-GEMM gather.
#
#   * small C_in (groups == 1, C_in < 16, 1D/2D): the generic path pads the
#     channel dim up to 16 to satisfy the same K >= 16 constraint, which wastes
#     (16 - C_in)/16 of the FLOPs plus a tensor copy on every call. With so few
#     channels a direct multiply-accumulate kernel (no im2col, no tensor core)
#     is cheaper than the padded dot.
#
# Everything else falls through to conv1d/conv2d/conv3d unchanged. Rewriting the
# generic GEMM itself was tried and rejected: a coalesced (out_channel, spatial)
# tile, an NHWC input with a transposed weight, and a full config sweep over the
# existing kernel each measured *slower* than the current implicit GEMM, whose
# uncoalesced loads are absorbed by L2. The remaining gap to cuDNN on the large
# conv2d shapes is structural (hand-tuned kernels and shared-memory staging that
# Triton cannot express here), not a tiling or configuration problem.
# ---------------------------------------------------------------------------


# Above this per-group channel count the padded tensor-core path beats the
# FMA-based direct kernel on bf16/f16 (see the dispatch comment below).
_DIRECT_MAX_C = 8

# tl.dot's minimum contraction length (K). Backends enforce it strictly - the
# CoreX/iluvatar compiler rejects any dot with K < 16 - so a GEMM path must
# never be handed fewer than this many reduction elements.
_DOT_MIN_K = 16


def _to_list(param, ndim):
    """Broadcast a scalar spatial parameter to one entry per spatial dimension."""
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


def _output_size(in_size, kernel, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


# --- depthwise: no cross-channel reduction, pure multiply-accumulate ---------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SP": 128}, num_warps=2),
        triton.Config({"BLOCK_SP": 256}, num_warps=2),
        triton.Config({"BLOCK_SP": 128}, num_warps=4),
        triton.Config({"BLOCK_SP": 256}, num_warps=4),
        triton.Config({"BLOCK_SP": 512}, num_warps=4),
    ],
    key=["N", "D", "H", "W", "OD", "OH", "OW", "KD", "KH", "KW"],
)
@triton.jit
def _depthwise_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_d_stride,
    in_h_stride,
    in_w_stride,
    w_c_stride,
    w_d_stride,
    w_h_stride,
    w_w_stride,
    out_n_stride,
    out_c_stride,
    out_d_stride,
    out_h_stride,
    out_w_stride,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # One program covers a single output channel and a contiguous run of output
    # pixels. Keeping the tile 1D means each tap is a flat gather multiplied by
    # a *scalar* weight, with no (BLOCK_C, BLOCK_SP) outer product to broadcast
    # and no channel dimension in the mask; that formulation measured ~1.6x
    # faster on 2D and ~5x faster on 3D than the equivalent (BLOCK_C, BLOCK_SP)
    # tile on A100.
    #
    # 1D and 2D are handled by the same kernel: the caller views them as 5D with
    # unit depth (and unit trailing width for 1D), so only the 3D addressing
    # exists here.
    pid_sp = tl.program_id(0)
    c = tl.program_id(1)

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    ow = sp_off % OW
    tmp = sp_off // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    # The tap loops re-read the input KD*KH*KW times (Triton cannot slice a
    # shifted window out of registers), so per-tap cost is dominated by mask and
    # address arithmetic, not by memory. Everything independent of the tap index
    # is hoisted: ow/oh/od live in range by construction, so only the batch bound
    # is left; the depth and row bounds depend only on kd/kh; and the pointer
    # advances by a scalar stride instead of being rebuilt from a full offset
    # vector.
    sp_ok = n < N
    iw0 = ow * SW - PW
    base = input_ptr + c * in_c_stride + n * in_n_stride + iw0 * in_w_stride

    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)
    for kd in tl.static_range(KD):
        idd = od * SD + kd * DD - PD
        depth_ok = (idd >= 0) & (idd < D)
        d_off = idd * in_d_stride
        for kh in tl.static_range(KH):
            ih = oh * SH + kh * DH - PH
            row_ok = sp_ok & depth_ok & (ih >= 0) & (ih < H)
            p = base + d_off + ih * in_h_stride
            for kw in tl.static_range(KW):
                iw = iw0 + kw * DW
                in_mask = row_ok & (iw >= 0) & (iw < W)
                x = tl.load(p, mask=in_mask, other=0.0).to(tl.float32)
                w = tl.load(
                    weight_ptr
                    + c * w_c_stride
                    + kd * w_d_stride
                    + kh * w_h_stride
                    + kw * w_w_stride
                ).to(tl.float32)
                acc += x * w
                p += DW * in_w_stride

    tl.store(
        output_ptr
        + c * out_c_stride
        + n * out_n_stride
        + od * out_d_stride
        + oh * out_h_stride
        + ow * out_w_stride,
        acc,
        mask=sp_ok,
    )


def _depthwise_conv(input, weight, padding, stride, dilation, ndim):
    # Normalize every dimensionality to a 5D (N, C, D, H, W) view: 1D gets unit
    # depth and width, 2D gets unit depth. Padding/stride/dilation are expanded
    # the same way so the kernel sees a single 3D case.
    if ndim == 1:
        input = input.unsqueeze(2).unsqueeze(-1)
        weight = weight.unsqueeze(2).unsqueeze(-1)
        padding = [0, padding[0], 0]
        stride = [1, stride[0], 1]
        dilation = [1, dilation[0], 1]
        squeeze = (2, 4)
    elif ndim == 2:
        input = input.unsqueeze(2)
        weight = weight.unsqueeze(2)
        padding = [0, padding[0], padding[1]]
        stride = [1, stride[0], stride[1]]
        dilation = [1, dilation[0], dilation[1]]
        squeeze = (2,)
    else:
        squeeze = ()

    N, C, D, H, W = input.shape
    OC, _, KD, KH, KW = weight.shape
    PD, PH, PW = padding
    SD, SH, SW = stride
    DD, DH, DW = dilation
    OD = _output_size(D, KD, SD, PD, DD)
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    output = torch.empty((N, OC, OD, OH, OW), device=input.device, dtype=input.dtype)

    in_s = input.stride()
    w_s = weight.stride()
    out_s = output.stride()

    # Depthwise means groups == C_in and weight.shape[1] == 1, so the output
    # channel count equals the input channel count and channel c maps to c.
    grid = lambda meta: (triton.cdiv(N * OD * OH * OW, meta["BLOCK_SP"]), C)
    _depthwise_conv_kernel[grid](
        input,
        weight,
        output,
        N,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        in_s[4],
        w_s[0],
        w_s[2],
        w_s[3],
        w_s[4],
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        out_s[4],
        KD,
        KH,
        KW,
        SD,
        SH,
        SW,
        PD,
        PH,
        PW,
        DD,
        DH,
        DW,
    )
    # Descending order so the remaining indices stay valid after each removal.
    for dim in sorted(squeeze, reverse=True):
        output = output.squeeze(dim)
    return output


# --- pointwise (1x1, groups == 1): plain GEMM batched over N -----------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _pointwise_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M,
    N,
    K,
    in_n_stride,
    in_c_stride,
    in_sp_stride,
    w_m_stride,
    w_k_stride,
    out_n_stride,
    out_m_stride,
    out_sp_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # weight is (OC, C_in) -> A of the GEMM (M=OC, K=C_in)
    a_ptrs = weight_ptr + offs_m[:, None] * w_m_stride + offs_k[None, :] * w_k_stride
    # input is (N, C_in, spatial) -> B (K=C_in, N=spatial)
    b_ptrs = (
        input_ptr
        + pid_b * in_n_stride
        + offs_k[:, None] * in_c_stride
        + offs_n[None, :] * in_sp_stride
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        a_mask = (offs_m < M)[:, None] & (offs_k < k_remain)[None, :]
        b_mask = (offs_k < k_remain)[:, None] & (offs_n < N)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * w_k_stride
        b_ptrs += BLOCK_K * in_c_stride

    out_ptrs = (
        output_ptr
        + pid_b * out_n_stride
        + offs_m[:, None] * out_m_stride
        + offs_n[None, :] * out_sp_stride
    )
    out_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def _pointwise_conv(input, weight, ndim):
    # input (N, C_in, *spatial), weight (OC, C_in, 1, 1, ...)
    N = input.shape[0]
    C_in = input.shape[1]
    OC = weight.shape[0]
    spatial = 1
    for s in input.shape[2:]:
        spatial *= s

    # GEMM: output[n] (OC, spatial) = weight (OC, C_in) @ input[n] (C_in, spatial)
    # Requires the spatial dims to be contiguous in memory (NCHW / NCDHW), which
    # holds for the tensors produced here; make it explicit.
    input = input.contiguous()
    weight = weight.contiguous()

    output = torch.empty((N, OC, spatial), device=input.device, dtype=input.dtype)

    in_s = input.stride()
    w_s = weight.stride()
    out_s = output.stride()

    grid = lambda meta: (
        triton.cdiv(OC, meta["BLOCK_M"]),
        triton.cdiv(spatial, meta["BLOCK_N"]),
        N,
    )
    _pointwise_conv_kernel[grid](
        input,
        weight,
        output,
        OC,
        spatial,
        C_in,
        in_s[0],
        in_s[1],
        1,
        w_s[0],
        w_s[1],
        out_s[0],
        out_s[1],
        1,
    )
    # restore the spatial shape
    return output.view((N, OC, *input.shape[2:]))


# --- small channel count (weight_c < 16): direct multiply-accumulate ---------
#
# The generic conv2d path pads the input/weight channel dim up to 16 so tl.dot's
# K >= 16 holds, but when C_in (groups == 1) is already < 16 that wastes
# (16 - C_in)/16 of the FLOPs plus an extra tensor copy on every call. For such
# small channel counts a plain direct convolution (no im2col, no tensor core)
# that accumulates over C_in * KH * KW taps is strictly cheaper.


@triton.jit
def _direct_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    H,
    W,
    OC,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_h_stride,
    in_w_stride,
    w_oc_stride,
    w_c_stride,
    w_h_stride,
    w_w_stride,
    out_n_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    # Grouped convolutions reuse the same kernel: each group owns C_IN input
    # channels and OC/GROUPS output channels, both offset by their group index.
    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ow = sp_off % OW
    tmp = sp_off // OW
    oh = tmp % OH
    n = tmp // OH
    in_c_base = pid_group * C_IN

    # Tile is (BLOCK_OC, BLOCK_SP) with the spatial dimension innermost so the
    # contiguous (unit-stride) spatial access coalesces. Accumulate in fp32
    # regardless of the input dtype. No tensor core is used (K = C_IN may be
    # below tl.dot's minimum), so this is pure multiply-accumulate.
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    for c in tl.static_range(C_IN):
        for kh in tl.static_range(KH):
            ih = oh * SH + kh * DH - PH
            for kw in tl.static_range(KW):
                iw = ow * SW + kw * DW - PW
                in_mask = (n < N) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x = tl.load(
                    input_ptr
                    + n * in_n_stride
                    + (in_c_base + c) * in_c_stride
                    + ih * in_h_stride
                    + iw * in_w_stride,
                    mask=in_mask,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    weight_ptr
                    + oc_glob[:, None] * w_oc_stride
                    + c * w_c_stride
                    + kh * w_h_stride
                    + kw * w_w_stride,
                    mask=(oc_off < oc_per_group)[:, None],
                    other=0.0,
                ).to(tl.float32)
                acc += x[None, :] * w

    out_mask = (
        (oc_off < oc_per_group)[:, None]
        & (n < N)[None, :]
        & (oh < OH)[None, :]
        & (ow < OW)[None, :]
    )
    out_ptr = (
        output_ptr
        + oc_glob[:, None] * out_c_stride
        + n[None, :] * out_n_stride
        + oh[None, :] * out_h_stride
        + ow[None, :] * out_w_stride
    )
    tl.store(out_ptr, acc, mask=out_mask)


def _direct_conv2d(input, weight, padding, stride, dilation, groups):
    N, _, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    output = torch.empty((N, OC, OH, OW), device=input.device, dtype=input.dtype)

    in_s = input.stride()
    w_s = weight.stride()
    out_s = output.stride()

    BLOCK_OC = 8
    BLOCK_SP = 256
    grid = (
        triton.cdiv(N * OH * OW, BLOCK_SP),
        triton.cdiv(OC // groups, BLOCK_OC),
        groups,
    )
    _direct_conv2d_kernel[grid](
        input,
        weight,
        output,
        N,
        H,
        W,
        OC,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        w_s[0],
        w_s[1],
        w_s[2],
        w_s[3],
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        KH,
        KW,
        SH,
        SW,
        PH,
        PW,
        DH,
        DW,
        C_IN=weight_c,
        GROUPS=groups,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=2,
    )
    return output


# The 3D small-channel case uses the opposite tiling from 2D. Here a 1D tile
# (one output channel per program, scalar weights per tap) wins by a wide
# margin, because a 3D kernel has KD*KH*KW times as many taps to unroll and the
# (BLOCK_OC, BLOCK_SP) outer product blows up the register pressure. Measured on
# (2,4,16,16,16) k3: 1D tile 16.4us vs 38.9us for the outer-product tile. The
# trade-off reverses in 2D (RGB: 136us vs 72us), where the outer-product tile
# amortizes the input load across BLOCK_OC output channels.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SP": 128}, num_warps=2),
        triton.Config({"BLOCK_SP": 128}, num_warps=4),
        triton.Config({"BLOCK_SP": 256}, num_warps=4),
    ],
    key=["N", "D", "H", "W", "OD", "OH", "OW", "KD", "KH", "KW"],
)
@triton.jit
def _direct_conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    OC,
    in_n_stride,
    in_c_stride,
    in_d_stride,
    in_h_stride,
    in_w_stride,
    w_oc_stride,
    w_c_stride,
    w_d_stride,
    w_h_stride,
    w_w_stride,
    out_n_stride,
    out_c_stride,
    out_d_stride,
    out_h_stride,
    out_w_stride,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    oc_per_group = OC // GROUPS
    oc = pid_group * oc_per_group + pid_oc

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ow = sp_off % OW
    tmp = sp_off // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    sp_ok = n < N
    iw0 = ow * SW - PW

    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)
    for c in tl.static_range(C_IN):
        in_base = (
            input_ptr
            + (pid_group * C_IN + c) * in_c_stride
            + n * in_n_stride
            + iw0 * in_w_stride
        )
        w_base = weight_ptr + oc * w_oc_stride + c * w_c_stride
        for kd in tl.static_range(KD):
            idd = od * SD + kd * DD - PD
            depth_ok = (idd >= 0) & (idd < D)
            d_off = idd * in_d_stride
            for kh in tl.static_range(KH):
                ih = oh * SH + kh * DH - PH
                row_ok = sp_ok & depth_ok & (ih >= 0) & (ih < H)
                p = in_base + d_off + ih * in_h_stride
                for kw in tl.static_range(KW):
                    iw = iw0 + kw * DW
                    in_mask = row_ok & (iw >= 0) & (iw < W)
                    x = tl.load(p, mask=in_mask, other=0.0).to(tl.float32)
                    w = tl.load(
                        w_base + kd * w_d_stride + kh * w_h_stride + kw * w_w_stride
                    ).to(tl.float32)
                    acc += x * w
                    p += DW * in_w_stride

    tl.store(
        output_ptr
        + oc * out_c_stride
        + n * out_n_stride
        + od * out_d_stride
        + oh * out_h_stride
        + ow * out_w_stride,
        acc,
        mask=sp_ok,
    )


def _direct_conv3d(input, weight, padding, stride, dilation, groups):
    N, _, D, H, W = input.shape
    OC, weight_c, KD, KH, KW = weight.shape
    PD, PH, PW = padding
    SD, SH, SW = stride
    DD, DH, DW = dilation
    OD = _output_size(D, KD, SD, PD, DD)
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    output = torch.empty((N, OC, OD, OH, OW), device=input.device, dtype=input.dtype)

    in_s = input.stride()
    w_s = weight.stride()
    out_s = output.stride()

    grid = lambda meta: (
        triton.cdiv(N * OD * OH * OW, meta["BLOCK_SP"]),
        OC // groups,
        groups,
    )
    _direct_conv3d_kernel[grid](
        input,
        weight,
        output,
        N,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        OC,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        in_s[4],
        w_s[0],
        w_s[1],
        w_s[2],
        w_s[3],
        w_s[4],
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        out_s[4],
        KD,
        KH,
        KW,
        SD,
        SH,
        SW,
        PD,
        PH,
        PW,
        DD,
        DH,
        DW,
        C_IN=weight_c,
        GROUPS=groups,
    )
    return output


def _direct_conv(input, weight, padding, stride, dilation, groups, ndim):
    if ndim == 1:
        # Lift to 2D with a unit trailing dim (same trick as conv1d -> conv2d).
        out = _direct_conv2d(
            input.unsqueeze(-1),
            weight.unsqueeze(-1),
            [padding[0], 0],
            [stride[0], 1],
            [dilation[0], 1],
            groups,
        )
        return out.squeeze(-1)
    if ndim == 2:
        return _direct_conv2d(input, weight, padding, stride, dilation, groups)
    raise ValueError(f"unsupported direct ndim {ndim}")


# --- public entry point ------------------------------------------------------


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """
    CUDNN convolution operation.

    This is a lower-level convolution operation that does not include bias.
    It supports 1D, 2D, and 3D convolutions based on the input dimensionality.

    Args:
        input: Input tensor of shape (N, C_in, *spatial_dims)
        weight: Weight tensor of shape (C_out, C_in/groups, *kernel_dims)
        padding: Padding for each spatial dimension
        stride: Stride for each spatial dimension
        dilation: Dilation for each spatial dimension
        groups: Number of groups for grouped convolution
        benchmark: cuDNN benchmark flag (ignored in Triton implementation)
        deterministic: cuDNN deterministic flag (ignored in Triton implementation)
        allow_tf32: Allow TF32 computation flag (ignored in Triton implementation)

    Returns:
        Output tensor after convolution
    """
    logger.debug("GEMS CUDNN_CONVOLUTION")

    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )

    padding = _to_list(padding, ndim)
    stride = _to_list(stride, ndim)
    dilation = _to_list(dilation, ndim)

    # AABS mispairs the block sizes of every kernel this operator launches, and
    # the knob is read at launch time, so it has to be held off around the whole
    # dispatch rather than around any single launch (see _aabs_disabled). Where
    # the fork has no such knob the branch below skips the manager outright: its
    # ~1.2us of host time is itself visible in the latency of the small
    # convolutions this op mostly runs.
    if _AABS_PRESENT:
        with _aabs_disabled():
            return _conv_forward(input, weight, padding, stride, dilation, groups)
    return _conv_forward(input, weight, padding, stride, dilation, groups)


def _conv_forward(input, weight, padding, stride, dilation, groups):
    """Pick and run the convolution path. ``padding``/``stride``/``dilation``
    are per-spatial-dimension lists by the time this runs."""
    ndim = input.ndim - 2

    # Dedicated depthwise kernel: no cross-channel reduction, avoids the
    # channel-to-16 padding that the generic tl.dot path would otherwise do.
    if weight.shape[1] == 1 and groups == input.shape[1]:
        return _depthwise_conv(input, weight, padding, stride, dilation, ndim)

    # Dedicated pointwise GEMM for 1x1 convolutions with a single group. Only
    # valid when there is no padding: a padded 1x1 conv has a larger output
    # spatial extent (zero-padded border), which the plain GEMM does not model.
    # A 1x1 kernel makes that GEMM's K exactly C_in (the kernel taps are all 1),
    # so a small C_in would hand tl.dot fewer than its minimum K; below that
    # floor fall through to the FMA direct kernel below, which reduces over
    # C_in * KH * KW without a dot.
    if (
        groups == 1
        and all(k == 1 for k in weight.shape[2:])
        and all(s == 1 for s in stride)
        and all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
        and weight.shape[1] >= _DOT_MIN_K
    ):
        return _pointwise_conv(input, weight, ndim)

    # Small per-group input-channel count (C_in/groups < 16): the generic conv2d
    # path pads the channel dim up to 16 to satisfy tl.dot's K >= 16, wasting the
    # extra FLOPs plus a tensor copy. A direct multiply-accumulate kernel avoids
    # that. Grouped convolutions are handled too, since a small channel count
    # wastes just as much whether or not the channels are split across groups.
    #
    # The direct kernel uses FMA, not tensor cores, so it only pays off when the
    # tensor-core path is not far ahead: either it is fp32 (where the exact-fp32
    # dot cannot use TF32 tensor cores at all) or the padding would waste at
    # least half the FLOPs (C_in <= 8 -> padded to 16 is a >= 2x waste).
    # Measured: for a grouped C_in=12 case the direct kernel wins at fp32
    # (0.44 -> 1.14) but loses badly at bf16/f16 (1.67 -> 1.18, 1.46 -> 0.85),
    # so the C_in <= 8 bound is where FMA overtakes the padded tensor-core dot.
    if weight.shape[1] < _DOT_MIN_K:
        if input.dtype == torch.float32 or weight.shape[1] <= _DIRECT_MAX_C:
            if ndim == 3:
                return _direct_conv3d(input, weight, padding, stride, dilation, groups)
            if ndim <= 2:
                return _direct_conv(
                    input, weight, padding, stride, dilation, groups, ndim
                )

    if ndim == 1:
        stride_val = stride[0]
        padding_val = padding[0]
        dilation_val = dilation[0]
        return conv1d(
            input,
            weight,
            bias=None,
            stride=stride_val,
            padding=padding_val,
            dilation=dilation_val,
            groups=groups,
        )
    elif ndim == 2:
        return conv2d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:  # ndim == 3, the only remaining value the caller admits
        return conv3d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
