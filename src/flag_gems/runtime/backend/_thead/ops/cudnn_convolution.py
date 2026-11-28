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

"""T-Head PPU (thead) implementation of ``cudnn_convolution``.

Vendored from the generic implementation (``flag_gems/ops/cudnn_convolution.py``,
the code path NVIDIA dispatches) with two changes:

* the ``conv2d`` / ``conv3d`` implicit-GEMM kernels the operator dispatches to
  are inlined here, so this file is self-contained. Only the forward kernels
  are reachable from ``cudnn_convolution``; the ``conv2d`` autograd backward is
  not part of this operator and stays in ``flag_gems/ops/conv2d.py`` for
  direct ``flag_gems.conv2d`` users.
* every ``@triton.autotune`` in this file is replaced by :func:`select`, which
  picks one launch configuration from the same search space the autotuners
  cover using only the problem shape.

Why: the weekly CI wipes ``~/.triton/cache`` before every run, so the first
case to touch a shape compiles every autotune candidate once per dtype -- 20
for conv2d and 11 for conv3d on thead. That compile cost alone does not fit in
the hard 1800s per-operator timeout of tools/run_tests.py (the generic
operator on a cold NV cache is 825s for 219 cases, ~92% of it compilation).
``select`` compiles one variant per (tile class, dtype) instead of the whole
candidate list -- no benchmarking, no LibTuner cache, no config-cache wipe.
"""

import contextlib
import logging

import torch
import triton
import triton.language as tl

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

    AABS (auto_adjust_block_sizes) rewrites block sizes at launch time, pairing
    a block to an extent by matching the ``tl.cdiv(extent, BLOCK)`` it sees
    against the kernel's argument names. Every convolution kernel below tiles
    the *flattened* spatial volume ``in_n * out_d * out_h * out_w``, but the
    analyzer keys the block to ``in_n`` alone, so the tile collapses to
    next_power_of_2(in_n) -- a measured 2.4x-19.7x slowdown, and a lowering
    failure on backends whose tensor-core tiles need M >= 16 rows.

    The knob is global state read at launch time, so holding it off around the
    whole dispatch is what covers every kernel this operator launches. The
    save/restore keeps every other operator's launch alone.
    """
    if not _AABS_PRESENT:
        yield
        return

    # The knob is a descriptor whose __set__ also writes os.environ on every
    # store, making a set/restore pair ~5.6us; its __get__ prefers the instance
    # dictionary, so writing there instead is enough (~0.3us) and is what the
    # launcher reads. Any FLAGTREE_AABS in the environment is left alone and
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
# Optimized paths, self-contained inside this operator (same dispatch as the
# generic implementation).
#
# The generic conv1d/conv2d/conv3d kernels are an implicit-GEMM that tiles the
# output as (spatial, out_channel) and reduces over (in_channel * kernel).
# Three cases get dedicated kernels:
#
#   * depthwise (groups == C_in, 1D/2D): no cross-channel reduction, so a
#     plain multiply-accumulate kernel avoids padding the per-group channel
#     (== 1) up to tl.dot's K >= 16 floor.
#
#   * 1x1 / pointwise (groups == 1, no padding): a plain GEMM batched over N.
#
#   * small C_in (groups == 1, C_in < 16, 1D/2D): a direct FMA kernel instead
#     of padding the channel dim up to 16 on every call.
#
# Everything else falls through to the inlined implicit-GEMM kernels below.
# ---------------------------------------------------------------------------


# Above this per-group channel count the padded tensor-core path beats the
# FMA-based direct kernel on bf16/f16.
_DIRECT_MAX_C = 8

# tl.dot's minimum contraction length (K). The inlined forward kernels pad the
# channel dim up to this before launching, so BLOCK_CI never goes below it.
_DOT_MIN_K = 16

# The generic conv2d kernel picks "bf16x6" only on Radeon, where an ieee fp32
# dot unrolls into scalar FMAs. thead has no such branch, so every dot here is
# ieee, i.e. the generic allow_tf32=False behaviour.
_DOT_PRECISION = tl.constexpr("ieee")


def _to_list(param, ndim):
    """Broadcast a scalar spatial parameter to one entry per spatial dimension."""
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


def _output_size(in_size, kernel, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


# ---------------------------------------------------------------------------
# select: deterministic replacement for @triton.autotune
# ---------------------------------------------------------------------------


def _cover(extent, choices):
    """Smallest tile that covers ``extent``; the largest when it exceeds all.

    A tile smaller than the extent costs an extra program per extent/tile
    boundary with its last program masked; a tile much larger wastes lanes.
    Capping at the largest tuned tile keeps a huge extent from requesting a
    block no tuned config ever used.
    """
    for choice in choices:
        if choice >= extent:
            return choice
    return choices[-1]


def select(kind, *, out_spatial, out_channels, in_channels=1, kernel_volume=1):
    """Pick one launch configuration from the tuned search space.

    Stands in for ``@triton.autotune`` on every kernel in this file. The
    candidate lists stay what they are -- ``_thead/tune_configs.yaml`` for the
    implicit-GEMM convolutions, the literals of the generic kernels for the
    dedicated ones -- the point is only to *choose* from that space instead of
    compiling and timing every entry on a cold cache.

    Args:
        kind: "conv2d", "conv3d", "depthwise", "pointwise" or "direct3d".
        out_spatial: number of output positions the grid spans
            (``in_n * out_height * out_width`` for 2D, the same with depth for
            3D, and the per-batch spatial extent for the pointwise GEMM).
        out_channels: output channels per group.
        in_channels: channels per group, i.e. the reduction length of the
            implicit-GEMM kernels. Unused by "depthwise" and "direct3d".
        kernel_volume: spatial kernel taps (``kh * kw`` or ``kd * kh * kw``).
            Unused by the non-GEMM kinds.

    Returns:
        Dict of kernel meta-parameters and launch options.
    """
    if kind == "conv2d":
        block_sp = _cover(out_spatial, (32, 64, 128, 256))
        block_co = _cover(out_channels, (32, 64, 128, 256))
        # The tuned space tops out at 256x128 accumulators (32768); shrink the
        # larger side until the pair fits that bound instead of leaving the
        # search space into tiles no tuned config ever allocated.
        while block_sp * block_co > 32768:
            if block_sp >= block_co:
                block_sp //= 2
            else:
                block_co //= 2
        # weight_c is padded to _DOT_MIN_K before launch, so 16 is a legal K.
        block_ci = 16 if in_channels <= 16 else 32
        # 256x128 is the widest tuned tile and uses 8 warps; everything up to
        # 128x128 uses 4.
        num_warps = 8 if block_sp * block_co >= 32768 else 4
        # Staging deeper than two only pays when the reduction loop is long
        # enough to keep the pipeline full (5x5/7x7 kernels, or 3x3 over more
        # than one channel block); a short single-block loop stays at 2.
        trips = kernel_volume * ((in_channels + block_ci - 1) // block_ci)
        num_stages = 3 if trips >= 16 else 2
        return {
            "BLOCK_NI_HO_WO": block_sp,
            "BLOCK_CI": block_ci,
            "BLOCK_CO": block_co,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    if kind == "conv3d":
        block_sp = _cover(out_spatial, (128, 256, 512))
        block_co = _cover(out_channels, (16, 32))
        # The widest tuned 3D tile is 512x16 (8192 accumulators).
        while block_sp * block_co > 8192:
            if block_sp >= block_co:
                block_sp //= 2
            else:
                block_co //= 2
        block_ci = 16 if in_channels <= 16 else 32
        # A 3D tile covers depth, height and width at once, so it is far
        # wider in registers; the tuned list keeps everything at 2 or 4 warps
        # and 4 is the majority for the common tiles.
        num_warps = 4 if block_sp * block_co >= 4096 else 2
        return {
            "BLOCK_NI_DO_HO_WO": block_sp,
            "BLOCK_CI": block_ci,
            "BLOCK_CO": block_co,
            "num_warps": num_warps,
            "num_stages": 2,
        }
    if kind == "depthwise":
        block_sp = _cover(out_spatial, (128, 256, 512))
        return {
            "BLOCK_SP": block_sp,
            "num_warps": 4 if block_sp >= 256 else 2,
        }
    if kind == "pointwise":
        block_m = _cover(out_channels, (64, 128))
        block_n = _cover(out_spatial, (64, 128))
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": 32,
            "num_warps": 8 if block_m * block_n >= 16384 else 4,
        }
    if kind == "direct3d":
        block_sp = _cover(out_spatial, (128, 256))
        return {
            "BLOCK_SP": block_sp,
            "num_warps": 4,
        }
    raise ValueError(f"unknown kernel kind {kind!r}")


# --- depthwise: no cross-channel reduction, pure multiply-accumulate ---------


@triton.jit
def _depthwise_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    D,
    H,
    W,
    MULT,
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
    # and no channel dimension in the mask.
    #
    # 1D and 2D are handled by the same kernel: the caller views them as 5D with
    # unit depth (and unit trailing width for 1D), so only the 3D addressing
    # exists here.
    pid_sp = tl.program_id(0)
    oc = tl.program_id(1)
    # A depthwise conv repeats each input channel across MULT consecutive output
    # channels, so output channel ``oc`` reads input channel ``oc // MULT``.
    ic = oc // MULT

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
    base = input_ptr + ic * in_c_stride + n * in_n_stride + iw0 * in_w_stride

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
                    + oc * w_c_stride
                    + kd * w_d_stride
                    + kh * w_h_stride
                    + kw * w_w_stride
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

    # Depthwise means groups == C_in and weight.shape[1] == 1, so each input
    # channel owns MULT = OC // C consecutive output channels.
    mult = OC // C
    launch = select("depthwise", out_spatial=N * OD * OH * OW, out_channels=OC)
    grid = (triton.cdiv(N * OD * OH * OW, launch["BLOCK_SP"]), OC)
    _depthwise_conv_kernel[grid](
        input,
        weight,
        output,
        N,
        D,
        H,
        W,
        mult,
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
        **launch,
    )
    # Descending order so the remaining indices stay valid after each removal.
    for dim in sorted(squeeze, reverse=True):
        output = output.squeeze(dim)
    return output


# --- pointwise (1x1, groups == 1): plain GEMM batched over N -----------------


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
    # Requires the spatial dims to be contiguous in memory (NCHW / NCDHW).
    input = input.contiguous()
    weight = weight.contiguous()

    output = torch.empty((N, OC, spatial), device=input.device, dtype=input.dtype)

    in_s = input.stride()
    w_s = weight.stride()
    out_s = output.stride()

    launch = select("pointwise", out_spatial=spatial, out_channels=OC, in_channels=C_in)
    grid = (
        triton.cdiv(OC, launch["BLOCK_M"]),
        triton.cdiv(spatial, launch["BLOCK_N"]),
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
        **launch,
    )
    # restore the spatial shape
    return output.view((N, OC, *input.shape[2:]))


# --- small channel count (weight_c < 16): direct multiply-accumulate ---------


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


# The 2D small-channel kernel has no autotune in the generic implementation;
# these are its measured launch parameters, kept as-is.
_DIRECT_BLOCK_OC = 8
_DIRECT_BLOCK_SP = 256
_DIRECT_NUM_WARPS = 2


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

    grid = (
        triton.cdiv(N * OH * OW, _DIRECT_BLOCK_SP),
        triton.cdiv(OC // groups, _DIRECT_BLOCK_OC),
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
        BLOCK_OC=_DIRECT_BLOCK_OC,
        BLOCK_SP=_DIRECT_BLOCK_SP,
        num_warps=_DIRECT_NUM_WARPS,
    )
    return output


# The 3D small-channel case uses the opposite tiling from 2D. Here a 1D tile
# (one output channel per program, scalar weights per tap) wins by a wide
# margin, because a 3D kernel has KD*KH*KW times as many taps to unroll and the
# (BLOCK_OC, BLOCK_SP) outer product blows up the register pressure.
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

    launch = select("direct3d", out_spatial=N * OD * OH * OW, out_channels=OC)
    grid = (
        triton.cdiv(N * OD * OH * OW, launch["BLOCK_SP"]),
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
        **launch,
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


# --- implicit-GEMM conv2d (inlined from flag_gems/ops/conv2d.py) -------------


@triton.jit
def _conv2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_height out_weight value in kernel
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

        accum += tl.dot(input_block, weight_block, input_precision=_DOT_PRECISION)
    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
    accum += bias
    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


def _conv2d_forward(input, weight, padding, stride, dilation, groups):
    """Forward of ``flag_gems.ops.conv2d`` minus the autograd bookkeeping.

    ``padding``/``stride``/``dilation`` are per-dimension pairs. Bias is not a
    parameter of ``cudnn_convolution``, so the zero bias the generic kernel is
    launched with here mirrors ``conv2d(..., bias=None)``.
    """
    in_n, _, input_height, input_width = input.shape
    out_c, weight_c, weight_height, weight_width = weight.shape
    padding_height, padding_width = padding
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation

    # tl.dot requires K = BLOCK_CI >= 16, so pad a smaller channel dim up to it
    # instead of letting the kernel issue a sub-16 dot.
    if weight_c < _DOT_MIN_K:
        pad_c = _DOT_MIN_K - weight_c
        if groups == 1:
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
        else:
            # For grouped conv, pad each group's channels independently.
            n, _, h, w = input.shape
            input = input.reshape(n, groups, weight_c, h, w)
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
            input = input.reshape(n, groups * (weight_c + pad_c), h, w)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_c))
        weight_c = weight_c + pad_c

    out_height = _output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = _output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output = torch.empty(
        (in_n, out_c, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )
    bias_pointer = torch.zeros(out_c, device=input.device, dtype=input.dtype)

    launch = select(
        "conv2d",
        out_spatial=in_n * out_height * out_width,
        out_channels=out_c // groups,
        in_channels=weight_c,
        kernel_volume=weight_height * weight_width,
    )
    grid = (
        triton.cdiv(in_n * out_height * out_width, launch["BLOCK_NI_HO_WO"]),
        triton.cdiv(out_c // groups, launch["BLOCK_CO"]),
        groups,
    )
    _conv2d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups=groups,
        **launch,
    )
    return output


# --- implicit-GEMM conv3d (inlined from flag_gems/ops/conv3d.py) -------------


@triton.jit
def _conv3d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_do_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_depth out_height out_weight value in kernel
    ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(
        0, BLOCK_NI_DO_HO_WO
    )
    ni_do_ho_offset = ni_do_ho_wo_offset // out_width
    ni_do_offset = ni_do_ho_offset // out_height
    in_n_point_value = ni_do_offset // out_depth
    output_depth_point_value = ni_do_offset % out_depth
    output_height_point_value = ni_do_ho_offset % out_height
    output_width_point_value = ni_do_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_depth, input_height, input_width] and
    # [groups, out_c, in_c, weight_depth, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for dhwc in range(weight_depth * weight_height * weight_width * BLOCK_CI_COUNT):
        c = (dhwc % BLOCK_CI_COUNT) * BLOCK_CI
        dhw = dhwc // BLOCK_CI_COUNT
        dh = dhw // weight_width
        d = dh // weight_height
        h = dh % weight_height
        w = dhw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_depth_offset = (
            d * dilation_depth - padding_depth + stride_depth * output_depth_point_value
        )
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_depth_stride * input_depth_offset)[:, None]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_depth_stride * d)
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_depth_offset)[:, None]
            & (input_depth_offset < input_depth)[:, None]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

        accum += tl.dot(input_block, weight_block, input_precision=_DOT_PRECISION)
    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
    accum += bias
    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_depth_stride * output_depth_point_value)[:, None]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_depth_point_value < out_depth)[:, None]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


def _conv3d_forward(input, weight, padding, stride, dilation, groups):
    """Forward of ``flag_gems.ops.conv3d`` with the channel padding applied.

    ``padding``/``stride``/``dilation`` are per-dimension triples. The generic
    ``conv3d`` has no autograd wrapper, so there is nothing to preserve beyond
    the forward numerics.
    """
    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape
    padding_depth, padding_height, padding_width = padding
    stride_depth, stride_height, stride_width = stride
    dilation_depth, dilation_height, dilation_width = dilation

    # tl.dot requires K = BLOCK_CI >= 16; pad the channel dim up to it.
    if weight_c < _DOT_MIN_K:
        pad_c = _DOT_MIN_K - weight_c
        if groups == 1:
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 0, 0, pad_c))
        else:
            # For grouped conv, pad each group's channels independently.
            n, _, d, h, w = input.shape
            input = input.reshape(n, groups, weight_c, d, h, w)
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 0, 0, pad_c))
            input = input.reshape(n, groups * (weight_c + pad_c), d, h, w)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 0, 0, pad_c))
        weight_c = weight_c + pad_c

    out_depth = _output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )
    out_height = _output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = _output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output = torch.empty(
        (in_n, out_c, out_depth, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )
    bias_pointer = torch.zeros(out_c, device=input.device, dtype=input.dtype)

    launch = select(
        "conv3d",
        out_spatial=in_n * out_depth * out_height * out_width,
        out_channels=out_c // groups,
        in_channels=weight_c,
        kernel_volume=weight_depth * weight_height * weight_width,
    )
    grid = (
        triton.cdiv(
            in_n * out_depth * out_height * out_width, launch["BLOCK_NI_DO_HO_WO"]
        ),
        triton.cdiv(out_c // groups, launch["BLOCK_CO"]),
        groups,
    )
    _conv3d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_depth,
        input_height,
        input_width,
        out_c,
        out_depth,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_depth,
        weight_height,
        weight_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        groups=groups,
        **launch,
    )
    return output


def _conv1d_forward(input, weight, padding, stride, dilation, groups):
    """Lift 1D to 2D with a unit trailing dim, as ``flag_gems.ops.conv1d``."""
    return _conv2d_forward(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        [padding[0], 0],
        [stride[0], 1],
        [dilation[0], 1],
        groups,
    ).squeeze(-1)


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
    logger.debug("GEMS_THEAD CUDNN_CONVOLUTION")

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
    # dispatch rather than around any single launch (see _aabs_disabled).
    if _AABS_PRESENT:
        with _aabs_disabled():
            return _conv_forward(input, weight, padding, stride, dilation, groups)
    return _conv_forward(input, weight, padding, stride, dilation, groups)


def _conv_forward(input, weight, padding, stride, dilation, groups):
    """Pick and run the convolution path. ``padding``/``stride``/``dilation``
    are per-spatial-dimension lists by the time this runs."""
    ndim = input.ndim - 2

    # Dedicated depthwise kernel: no cross-channel reduction, avoids the
    # channel-to-16 padding that the implicit-GEMM path would otherwise do.
    if weight.shape[1] == 1 and groups == input.shape[1]:
        return _depthwise_conv(input, weight, padding, stride, dilation, ndim)

    # Dedicated pointwise GEMM for 1x1 convolutions with a single group. Only
    # valid with no padding (a padded 1x1 conv has a larger output spatial
    # extent) and with K = C_in >= the dot floor.
    if (
        groups == 1
        and all(k == 1 for k in weight.shape[2:])
        and all(s == 1 for s in stride)
        and all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
        and weight.shape[1] >= _DOT_MIN_K
    ):
        return _pointwise_conv(input, weight, ndim)

    # Small per-group input-channel count: a direct FMA kernel beats padding
    # the channel dim up to 16, but only when the tensor-core path is not far
    # ahead -- fp32 (the ieee dot cannot use TF32 tensor cores) or a padding
    # waste of at least 2x (C_in <= 8).
    if weight.shape[1] < _DOT_MIN_K:
        if input.dtype == torch.float32 or weight.shape[1] <= _DIRECT_MAX_C:
            if ndim == 3:
                return _direct_conv3d(input, weight, padding, stride, dilation, groups)
            if ndim <= 2:
                return _direct_conv(
                    input, weight, padding, stride, dilation, groups, ndim
                )

    if ndim == 1:
        return _conv1d_forward(input, weight, padding, stride, dilation, groups)
    elif ndim == 2:
        return _conv2d_forward(input, weight, padding, stride, dilation, groups)
    else:  # ndim == 3, the only remaining value the caller admits
        return _conv3d_forward(input, weight, padding, stride, dilation, groups)
