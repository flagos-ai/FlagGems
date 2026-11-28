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

from flag_gems.ops.avg_pool2d import _parse_pool_params, pool2d_output_size
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Tuning knobs (adjusted one variable at a time during the optimization rounds,
# mirroring the adaptive_avg_pool2d recipe on GCU400/L600):
# - GRID_LIMIT keeps the grid-stride walk inside one <=48-CTA wave (larger
#   grids measured monotonically worse: waves serialize, no CTA interleaving).
# - BLOCK_CAP bounds each spatial block; plane batching (P) fills the tile.
# - Forward tiles use the same 8K-element / 1-warp configuration for every
#   output-plane size. A per-shape sweep over the five benchmark core shapes
#   measured the size-dispatched 16K/1-warp (big planes) and 4K/2-warp (small
#   planes) pair at 0.327x mean speedup of the ATen leg, while a single 8K tile
#   at 1 warp reached 0.49x: the 16K tile halves the CTA count on the 28x28 and
#   14x14 plane cases and the 2-warp small tile loses the 4x4-plane tail. The
#   backward candidate walk ((<= cdiv(K-1,S)+1)^2 loads per input element)
#   still prefers the 16K tile at 1 warp on every core shape.
POOL_GRID_LIMIT = 48
POOL_BLOCK_CAP = 64
POOL_TILE_FLOOR = 1024
POOL_FWD_BIG_PLANE_OUT = 128
POOL_FWD_TILE_BIG = 8192
POOL_FWD_WARPS_BIG = 1
POOL_FWD_TILE_SMALL = 8192
POOL_FWD_WARPS_SMALL = 1
POOL_BWD_TILE_LIMIT = 16384
POOL_BWD_WARPS = 1


@libentry()
@triton.jit
def avg_pool2d_forward_kernel(
    X,
    Y,
    NC,
    IH,
    IW,
    OH,
    OW,
    UNITS,
    OH_BLOCKS,
    OW_BLOCKS,
    INV_DIV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    UNIFORM_DIV: tl.constexpr,
    P: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    """Plane-batched sliding-window forward.

    One unit owns a (P * BLOCK_OH, BLOCK_OW) output tile whose row axis mixes P
    consecutive (n, c) planes with BLOCK_OH output rows, so a single 2D tile op
    replaces the per-plane serial loop of the previous implementation. The
    KH * KW window loads are constexpr-unrolled computed-offset gathers (the
    layout proven on GCU400 by the adaptive_avg_pool2d general arm), the
    accumulation stays in fp32, and the uniform divisor branches multiply by a
    host-precomputed reciprocal instead of dividing per element. A grid-stride
    walk keeps the grid inside one <=48-CTA wave regardless of N*C.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    ps_in = IH * IW
    ps_out = OH * OW
    rows = tl.arange(0, P * BLOCK_OH)
    p_local = rows // BLOCK_OH
    oh_local = rows - p_local * BLOCK_OH
    cols = tl.arange(0, BLOCK_OW)
    for unit in tl.range(pid, UNITS, num_progs):
        wb = unit % OW_BLOCKS
        rest = unit // OW_BLOCKS
        hb = rest % OH_BLOCKS
        pb = rest // OH_BLOCKS

        plane = pb * P + p_local
        oh = hb * BLOCK_OH + oh_local
        ow = wb * BLOCK_OW + cols

        row_ok = plane < NC
        oh_ok = oh < OH
        col_ok = ow < OW
        out_ok = row_ok & oh_ok

        acc = tl.zeros((P * BLOCK_OH, BLOCK_OW), dtype=tl.float32)
        for kh in range(KH):
            h_in = oh * SH - PH + kh
            h_ok = (h_in >= 0) & (h_in < IH) & row_ok & oh_ok
            # Scalar tensor base plus an integer offset tile keeps the load on
            # the vectorized path; a pointer-vector base degrades to
            # per-lane scalar loads on GCU400.
            row_off = plane * ps_in + h_in * IW
            for kw in range(KW):
                w_in = ow * SW - PW + kw
                w_ok = (w_in >= 0) & (w_in < IW) & col_ok
                vals = tl.load(
                    X + row_off[:, None] + w_in[None, :],
                    mask=h_ok[:, None] & w_ok[None, :],
                    other=0.0,
                )
                acc += vals.to(tl.float32)

        if UNIFORM_DIV:
            out_vals = acc * INV_DIV
        else:
            # count_include_pad=False: analytic valid-element count per window
            # (dilation is always 1), identical to the ATen divisor.
            h0 = oh * SH - PH
            w0 = ow * SW - PW
            vh = tl.maximum(0, tl.minimum(IH, h0 + KH) - tl.maximum(0, h0))
            vw = tl.maximum(0, tl.minimum(IW, w0 + KW) - tl.maximum(0, w0))
            divisor = (vh[:, None] * vw[None, :]).to(tl.float32)
            out_vals = tl.where(divisor != 0, acc / divisor, 0.0)

        tl.store(
            Y + (plane * ps_out + oh * OW)[:, None] + ow[None, :],
            out_vals.to(Y.dtype.element_ty),
            mask=out_ok[:, None] & col_ok[None, :],
        )


@libentry()
@triton.jit
def avg_pool2d_backward_kernel(
    GO,
    GI,
    NC,
    IH,
    IW,
    OH,
    OW,
    UNITS,
    IH_BLOCKS,
    IW_BLOCKS,
    INV_DIV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    UNIFORM_DIV: tl.constexpr,
    P: tl.constexpr,
    BLOCK_IH: tl.constexpr,
    BLOCK_IW: tl.constexpr,
    NUM_H_STEPS: tl.constexpr,
    NUM_W_STEPS: tl.constexpr,
):
    """Scatter-as-gather backward over plane-batched input tiles.

    Each grad_input element receives from at most
    (ceil((KH-1)/SH)+1) * (ceil((KW-1)/SW)+1) output elements, so instead of
    sweeping all KH*KW window offsets the kernel visits only those candidates:
    oh = (ih + PH) // SH - jh for jh in [0, NUM_H_STEPS). The unit grid covers
    every input element exactly once (each lane is stored, zeros when no
    candidate hits), which makes an empty output buffer safe without atomics
    or a zero-init pass. Uniform divisor branches multiply the fp32
    accumulator by a host-precomputed reciprocal once per tile.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    ps_in = IH * IW
    ps_out = OH * OW
    rows = tl.arange(0, P * BLOCK_IH)
    p_local = rows // BLOCK_IH
    ih_local = rows - p_local * BLOCK_IH
    cols = tl.arange(0, BLOCK_IW)
    for unit in tl.range(pid, UNITS, num_progs):
        wb = unit % IW_BLOCKS
        rest = unit // IW_BLOCKS
        hb = rest % IH_BLOCKS
        pb = rest // IH_BLOCKS

        plane = pb * P + p_local
        ih = hb * BLOCK_IH + ih_local
        iw = wb * BLOCK_IW + cols

        row_ok = (plane < NC) & (ih < IH)
        col_ok = iw < IW

        base_h = (ih + PH) // SH
        rem_h = (ih + PH) - base_h * SH
        base_w = (iw + PW) // SW
        rem_w = (iw + PW) - base_w * SW

        acc = tl.zeros((P * BLOCK_IH, BLOCK_IW), dtype=tl.float32)
        for jh in range(NUM_H_STEPS):
            oh = base_h - jh
            kh = rem_h + jh * SH
            h_ok = (oh >= 0) & (oh < OH) & (kh < KH) & row_ok
            for jw in range(NUM_W_STEPS):
                ow = base_w - jw
                kw = rem_w + jw * SW
                w_ok = (ow >= 0) & (ow < OW) & (kw < KW) & col_ok
                go_ptr = GO + (plane * ps_out + oh * OW)[:, None] + ow[None, :]
                grad_val = tl.load(
                    go_ptr, mask=h_ok[:, None] & w_ok[None, :], other=0.0
                ).to(tl.float32)
                if UNIFORM_DIV:
                    acc += grad_val
                else:
                    # count_include_pad=False: per-window analytic divisor,
                    # matching the ATen backward semantics.
                    h0 = oh * SH - PH
                    w0 = ow * SW - PW
                    vh = tl.maximum(0, tl.minimum(IH, h0 + KH) - tl.maximum(0, h0))
                    vw = tl.maximum(0, tl.minimum(IW, w0 + KW) - tl.maximum(0, w0))
                    divisor = (vh[:, None] * vw[None, :]).to(tl.float32)
                    divisor = tl.where(divisor == 0, 1.0, divisor)
                    acc += grad_val / divisor

        if UNIFORM_DIV:
            acc = acc * INV_DIV
        gi_ptr = GI + (plane * ps_in + ih * IW)[:, None] + iw[None, :]
        tl.store(
            gi_ptr,
            acc.to(GI.dtype.element_ty),
            mask=row_ok[:, None] & col_ok[None, :],
        )


def _pool_tiling(dim_h, dim_w, total_planes, tile_limit):
    """Pick (block_h, block_w, planes) for the plane-batched tiles.

    Blocks are capped so the tile stays inside the DSM budget with plane
    batching filling the rest; the shrink loop trades tile size for more units
    until one <=48-CTA wave is filled (bandwidth spreads across the SIPs for
    small problems), stopping at the tile floor.
    """
    block_h = min(triton.next_power_of_2(dim_h), POOL_BLOCK_CAP)
    block_w = min(triton.next_power_of_2(dim_w), POOL_BLOCK_CAP)
    planes = max(1, tile_limit // (block_h * block_w))
    h_blocks = triton.cdiv(dim_h, block_h)
    w_blocks = triton.cdiv(dim_w, block_w)
    units = triton.cdiv(total_planes, planes) * h_blocks * w_blocks
    while (
        units < POOL_GRID_LIMIT
        and block_h * block_w * planes > POOL_TILE_FLOOR
        and (planes > 1 or block_h > 1 or block_w > 1)
    ):
        if planes > 1:
            planes //= 2
        elif block_h >= block_w:
            block_h //= 2
        else:
            block_w //= 2
        h_blocks = triton.cdiv(dim_h, block_h)
        w_blocks = triton.cdiv(dim_w, block_w)
        units = triton.cdiv(total_planes, planes) * h_blocks * w_blocks
    return block_h, block_w, planes, h_blocks, w_blocks, units


def _pool_uniform_divisor(kernel_h, kernel_w, count_include_pad, divisor_override):
    """Return (uniform_flag, reciprocal) for the divisor branches.

    count_include_pad=True and divisor_override divide every window by the same
    value, so the kernel multiplies the fp32 accumulator by a host-computed
    reciprocal; count_include_pad=False needs per-window analytic counts.
    """
    if divisor_override is not None:
        return True, 1.0 / divisor_override
    if count_include_pad:
        return True, 1.0 / (kernel_h * kernel_w)
    return False, 1.0


def avg_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    logger.debug("GEMS_ENFLAME AVG_POOL2D")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    input = input.contiguous()

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )

    in_n, in_c, in_h, in_w = input.shape

    out_h = pool2d_output_size(in_h, kernel_h, stride_h, padding_h, 1, ceil_mode)
    out_w = pool2d_output_size(in_w, kernel_w, stride_w, padding_w, 1, ceil_mode)

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output

    if out_h * out_w >= POOL_FWD_BIG_PLANE_OUT:
        fwd_tile_limit = POOL_FWD_TILE_BIG
        fwd_warps = POOL_FWD_WARPS_BIG
    else:
        fwd_tile_limit = POOL_FWD_TILE_SMALL
        fwd_warps = POOL_FWD_WARPS_SMALL
    block_oh, block_ow, planes, oh_blocks, ow_blocks, units = _pool_tiling(
        out_h, out_w, in_n * in_c, fwd_tile_limit
    )
    grid = (min(units, POOL_GRID_LIMIT),)
    uniform_div, inv_div = _pool_uniform_divisor(
        kernel_h, kernel_w, count_include_pad, divisor_override
    )

    avg_pool2d_forward_kernel[grid](
        input,
        output,
        in_n * in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        units,
        oh_blocks,
        ow_blocks,
        inv_div,
        KH=kernel_h,
        KW=kernel_w,
        SH=stride_h,
        SW=stride_w,
        PH=padding_h,
        PW=padding_w,
        UNIFORM_DIV=uniform_div,
        P=planes,
        BLOCK_OH=block_oh,
        BLOCK_OW=block_ow,
        num_warps=fwd_warps,
    )

    return output


def avg_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    logger.debug("GEMS_ENFLAME AVG_POOL2D_BACKWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    grad_output = grad_output.contiguous()

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    if grad_output.numel() == 0:
        # The kernel-side zero fill relies on masked loads from grad_output,
        # which never run against an empty tensor: materialize zeros here.
        return torch.zeros(
            (in_n, in_c, in_h, in_w), device=grad_output.device, dtype=grad_output.dtype
        )

    # Every input element is stored by exactly one unit (zeros where no window
    # contributes), so an uninitialized buffer is safe and the zero-init pass
    # plus the fp32->dtype cast copy of the old implementation disappear.
    grad_input = torch.empty(
        (in_n, in_c, in_h, in_w), device=grad_output.device, dtype=grad_output.dtype
    )
    if grad_input.numel() == 0:
        return grad_input

    block_ih, block_iw, planes, ih_blocks, iw_blocks, units = _pool_tiling(
        in_h, in_w, in_n * in_c, POOL_BWD_TILE_LIMIT
    )
    grid = (min(units, POOL_GRID_LIMIT),)
    uniform_div, inv_div = _pool_uniform_divisor(
        kernel_h, kernel_w, count_include_pad, divisor_override
    )

    # Each input element receives from at most cdiv(K-1, S)+1 outputs per axis.
    num_h_steps = (kernel_h + stride_h - 2) // stride_h + 1
    num_w_steps = (kernel_w + stride_w - 2) // stride_w + 1

    avg_pool2d_backward_kernel[grid](
        grad_output,
        grad_input,
        in_n * in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        units,
        ih_blocks,
        iw_blocks,
        inv_div,
        KH=kernel_h,
        KW=kernel_w,
        SH=stride_h,
        SW=stride_w,
        PH=padding_h,
        PW=padding_w,
        UNIFORM_DIV=uniform_div,
        P=planes,
        BLOCK_IH=block_ih,
        BLOCK_IW=block_iw,
        NUM_H_STEPS=num_h_steps,
        NUM_W_STEPS=num_w_steps,
        num_warps=POOL_BWD_WARPS,
    )

    return grad_input
