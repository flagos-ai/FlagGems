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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Pixel Shuffle: (N, C*r^2, H, W) -> (N, C, H*r, W*r)
# Direct index mapping kernel - each output element reads from the correct
# input position without intermediate tensors.
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],
)
@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    C,
    H,
    W,
    R,
    C_out,
    H_out,
    W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Output layout: (N, C_out, H_out, W_out)
    ow = offsets % W_out
    tmp = offsets // W_out
    oh = tmp % H_out
    tmp2 = tmp // H_out
    c_out = tmp2 % C_out
    n = tmp2 // C_out

    # Map to input: h_in = oh // R, w_in = ow // R
    h_in = oh // R
    dh = oh % R
    w_in = ow // R
    dw = ow % R

    # Input channel: c_in = c_out * R * R + dh * R + dw
    c_in = c_out * R * R + dh * R + dw

    # Input linear index
    in_idx = n * (C * H * W) + c_in * (H * W) + h_in * W + w_in

    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


@libentry()
@triton.jit
def pixel_shuffle_pow2_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    LOG2_WOUT: tl.constexpr,
    LOG2_HOUT: tl.constexpr,
    LOG2_COUT: tl.constexpr,
    LOG2_R: tl.constexpr,
    LOG2_W: tl.constexpr,
    LOG2_HW: tl.constexpr,
    LOG2_CHW: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    PERSIST: tl.constexpr,
):
    # Power-of-two dims (r, W, H, C_out): the div/mod decode lowers to
    # shift/mask arithmetic. PERSIST selects a 48-program grid-stride
    # loop; the flat arm keeps one-block programs for small sizes.
    LOG2_R2: tl.constexpr = 2 * LOG2_R
    WOUT_MASK: tl.constexpr = (1 << LOG2_WOUT) - 1
    HOUT_MASK: tl.constexpr = (1 << LOG2_HOUT) - 1
    COUT_MASK: tl.constexpr = (1 << LOG2_COUT) - 1
    R_MASK: tl.constexpr = (1 << LOG2_R) - 1
    if PERSIST:
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        for start in tl.range(pid * BLOCK, n_elements, nprog * BLOCK):
            o = start + tl.arange(0, BLOCK)
            ow = o & WOUT_MASK
            t = o >> LOG2_WOUT
            oh = t & HOUT_MASK
            t2 = t >> LOG2_HOUT
            c = t2 & COUT_MASK
            n = t2 >> LOG2_COUT
            h = oh >> LOG2_R
            dh = oh & R_MASK
            w = ow >> LOG2_R
            dw = ow & R_MASK
            cin = (c << LOG2_R2) + (dh << LOG2_R) + dw
            in_idx = (n << LOG2_CHW) + (cin << LOG2_HW) + (h << LOG2_W) + w
            if EVEN:
                val = tl.load(in_ptr + in_idx)
                tl.store(out_ptr + o, val)
            else:
                mk = o < n_elements
                val = tl.load(in_ptr + in_idx, mask=mk)
                tl.store(out_ptr + o, val, mask=mk)
    else:
        pid = tl.program_id(0)
        o = pid * BLOCK + tl.arange(0, BLOCK)
        ow = o & WOUT_MASK
        t = o >> LOG2_WOUT
        oh = t & HOUT_MASK
        t2 = t >> LOG2_HOUT
        c = t2 & COUT_MASK
        n = t2 >> LOG2_COUT
        h = oh >> LOG2_R
        dh = oh & R_MASK
        w = ow >> LOG2_R
        dw = ow & R_MASK
        cin = (c << LOG2_R2) + (dh << LOG2_R) + dw
        in_idx = (n << LOG2_CHW) + (cin << LOG2_HW) + (h << LOG2_W) + w
        if EVEN:
            val = tl.load(in_ptr + in_idx)
            tl.store(out_ptr + o, val)
        else:
            mk = o < n_elements
            val = tl.load(in_ptr + in_idx, mask=mk)
            tl.store(out_ptr + o, val, mask=mk)


@libentry()
@triton.jit
def pixel_shuffle_pack16_kernel(
    in_ptr32,
    out_ptr64,
    n_words,
    LOG2_WD: tl.constexpr,
    LOG2_H: tl.constexpr,
    LOG2_COUT: tl.constexpr,
    LOG2_HW2: tl.constexpr,
    LOG2_CHW2: tl.constexpr,
    LOG2_W2: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # r == 2 with 2-byte dtypes: the output row interleave of two input
    # rows is expressed as register bit ops over int32 words, so both the
    # load side (two contiguous int32 streams) and the store side (flat
    # int64) are fully contiguous. One int64 word covers 4 elements.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in tl.range(pid * BLOCK, n_words, nprog * BLOCK):
        m = start + tl.arange(0, BLOCK)
        row = m >> LOG2_WD
        tp = m & ((1 << LOG2_WD) - 1)
        dh = row & 1
        row2 = row >> 1
        h = row2 & ((1 << LOG2_H) - 1)
        row3 = row2 >> LOG2_H
        c = row3 & ((1 << LOG2_COUT) - 1)
        n = row3 >> LOG2_COUT
        base = (
            (n << LOG2_CHW2) + ((c << 2) + (dh << 1)) * (1 << LOG2_HW2) + (h << LOG2_W2)
        )
        x_w = tl.load(in_ptr32 + base + tp)
        y_w = tl.load(in_ptr32 + base + tp + (1 << LOG2_HW2))
        w0 = (x_w & 0xFFFF) | (y_w << 16)
        w1 = ((x_w >> 16) & 0xFFFF) | ((y_w >> 16) << 16)
        # zero-extend each int32 half before packing into int64
        val = (w0.to(tl.int64) & 0xFFFFFFFF) | ((w1.to(tl.int64) & 0xFFFFFFFF) << 32)
        if EVEN:
            tl.store(out_ptr64 + m, val)
        else:
            tl.store(out_ptr64 + m, val, mask=m < n_words)


@libentry()
@triton.jit
def pixel_shuffle_r3_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    CHW,
    LOG2_W: tl.constexpr,
    LOG2_H: tl.constexpr,
    LOG2_COUT: tl.constexpr,
    LOG2_HW: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # r == 3 with power-of-two W/H/C_out: division by 3 uses the magic
    # multiply (x * 0xAAAB) >> 17, valid for x < 2^15.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    W_MASK: tl.constexpr = (1 << LOG2_W) - 1
    H_MASK: tl.constexpr = (1 << LOG2_H) - 1
    COUT_MASK: tl.constexpr = (1 << LOG2_COUT) - 1
    for start in tl.range(pid * BLOCK, n_elements, nprog * BLOCK):
        o = start + tl.arange(0, BLOCK)
        x = o >> LOG2_W
        xo = (x * 0xAAAB) >> 17
        row = xo
        lane = (o & W_MASK) + ((x - 3 * xo) << LOG2_W)
        ql = (lane * 0xAAAB) >> 17
        dw = lane - 3 * ql
        w = ql
        qd = (row * 0xAAAB) >> 17
        dh = row - 3 * qd
        row2 = qd
        h = row2 & H_MASK
        row3 = row2 >> LOG2_H
        c = row3 & COUT_MASK
        n = row3 >> LOG2_COUT
        cin = c * 9 + dh * 3 + dw
        in_idx = n * CHW + (cin << LOG2_HW) + (h << LOG2_W) + w
        if EVEN:
            val = tl.load(in_ptr + in_idx)
            tl.store(out_ptr + o, val)
        else:
            mk = o < n_elements
            val = tl.load(in_ptr + in_idx, mask=mk)
            tl.store(out_ptr + o, val, mask=mk)


@libentry()
@triton.jit
def pixel_shuffle_r3_pack16_kernel(
    in_ptr16,
    out_ptr32,
    n_words,
    CHW,
    LOG2_W: tl.constexpr,
    LOG2_H: tl.constexpr,
    LOG2_COUT: tl.constexpr,
    LOG2_HW: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # r == 3 with 2-byte dtypes: pack two output elements per int32 word
    # (W_out = 3*W is even, so a word never straddles an output row) and
    # halve the store issue count. Loads stay 2-byte gathers.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in tl.range(pid * BLOCK, n_words, nprog * BLOCK):
        m = start + tl.arange(0, BLOCK)
        o = m << 1
        x = o >> LOG2_W
        xo = (x * 0xAAAB) >> 17
        row = xo
        lane0 = (o & ((1 << LOG2_W) - 1)) + ((x - 3 * xo) << LOG2_W)
        lane1 = lane0 + 1
        ql0 = (lane0 * 0xAAAB) >> 17
        dw0 = lane0 - 3 * ql0
        w0i = ql0
        ql1 = (lane1 * 0xAAAB) >> 17
        dw1 = lane1 - 3 * ql1
        w1i = ql1
        qd = (row * 0xAAAB) >> 17
        dh = row - 3 * qd
        row2 = qd
        h = row2 & ((1 << LOG2_H) - 1)
        row3 = row2 >> LOG2_H
        c = row3 & ((1 << LOG2_COUT) - 1)
        n = row3 >> LOG2_COUT
        base = n * CHW + ((c * 9 + dh * 3) << LOG2_HW) + (h << LOG2_W)
        a0 = base + (dw0 << LOG2_HW) + w0i
        a1 = base + (dw1 << LOG2_HW) + w1i
        v0 = tl.load(in_ptr16 + a0)
        v1 = tl.load(in_ptr16 + a1)
        lo = v0.to(tl.int32) & 0xFFFF
        hi = v1.to(tl.int32) & 0xFFFF
        word = lo | (hi << 16)
        if EVEN:
            tl.store(out_ptr32 + m, word)
        else:
            tl.store(out_ptr32 + m, word, mask=m < n_words)


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _log2(v):
    return v.bit_length() - 1


def _next_pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


# Grid-stride persistent kernels cap the grid at the SPC count; below this
# size the flat arm (one block per program) avoids the loop scaffold.
_FLAT_MAX_ELEMENTS = 48 * 1024
# Magic-multiply //3 domain guards for the upscale-factor-3 kernels.
_R3_LANE_MAX = 32768


def _launch_pow2(x, out, r, Cout, C, H, W, n):
    Hout, Wout = H * r, W * r
    if n <= _FLAT_MAX_ELEMENTS:
        block = 1024
        grid = (triton.cdiv(n, block),)
        persist = False
        warps = 1
    elif n <= 524288:
        block = 2048
        grid = (min(48, triton.cdiv(n, block)),)
        persist = True
        warps = 4
    else:
        block = 4096
        grid = (min(48, triton.cdiv(n, block)),)
        persist = True
        warps = 4
    pixel_shuffle_pow2_kernel[grid](
        x,
        out,
        n,
        LOG2_WOUT=_log2(Wout),
        LOG2_HOUT=_log2(Hout),
        LOG2_COUT=_log2(Cout),
        LOG2_R=_log2(r),
        LOG2_W=_log2(W),
        LOG2_HW=_log2(H * W),
        LOG2_CHW=_log2(C * H * W),
        BLOCK=block,
        EVEN=(n % block == 0),
        PERSIST=persist,
        num_warps=warps,
    )


def _launch_pack16(x, out, r, Cout, C, H, W, n):
    x32 = x.view(-1).view(torch.int32)
    out64 = out.view(-1).view(torch.int64)
    n_words = out64.numel()
    block = 4096 if n_words >= 4096 else _next_pow2(max(n_words, 64))
    grid = (min(48, triton.cdiv(n_words, block)),)
    pixel_shuffle_pack16_kernel[grid](
        x32,
        out64,
        n_words,
        LOG2_WD=_log2(W // 2),
        LOG2_H=_log2(H),
        LOG2_COUT=_log2(Cout),
        LOG2_HW2=_log2(H * W // 2),
        LOG2_CHW2=_log2(C * H * W // 2),
        LOG2_W2=_log2(W // 2),
        BLOCK=block,
        EVEN=(n_words % block == 0),
        num_warps=4,
    )


def _launch_r3(x, out, r, Cout, C, H, W, n):
    if n <= _FLAT_MAX_ELEMENTS:
        block = 1024
        warps = 1
    else:
        block = 2048
        warps = 4
    grid = (min(48, triton.cdiv(n, block)),)
    pixel_shuffle_r3_kernel[grid](
        x,
        out,
        n,
        C * H * W,
        LOG2_W=_log2(W),
        LOG2_H=_log2(H),
        LOG2_COUT=_log2(Cout),
        LOG2_HW=_log2(H * W),
        BLOCK=block,
        EVEN=(n % block == 0),
        num_warps=warps,
    )


def _launch_r3_pack16(x, out, r, Cout, C, H, W, n):
    x16 = x.view(-1).view(torch.int16)
    out32 = out.view(-1).view(torch.int32)
    n_words = out32.numel()
    if n_words <= _FLAT_MAX_ELEMENTS:
        block = 1024
        warps = 1
    else:
        block = 4096
        warps = 4
    grid = (min(48, triton.cdiv(n_words, block)),)
    pixel_shuffle_r3_pack16_kernel[grid](
        x16,
        out32,
        n_words,
        C * H * W,
        LOG2_W=_log2(W),
        LOG2_H=_log2(H),
        LOG2_COUT=_log2(Cout),
        LOG2_HW=_log2(H * W),
        BLOCK=block,
        EVEN=(n_words % block == 0),
        num_warps=warps,
    )


def pixel_shuffle(input, upscale_factor):
    logger.debug("GEMS_ENFLAME PIXEL_SHUFFLE")
    r = int(upscale_factor)
    assert input.ndim == 4
    N, C, H, W = input.shape
    assert C % (r * r) == 0

    scale_square = r * r
    C_out = C // scale_square
    H_out = H * r
    W_out = W * r

    input = input.contiguous()
    output = torch.empty(
        (N, C_out, H_out, W_out), device=input.device, dtype=input.dtype
    )

    n_elements = output.numel()
    if n_elements == 0:
        return output

    elem_size = input.element_size()
    pow2_dims = _is_pow2(r) and _is_pow2(W) and _is_pow2(H) and _is_pow2(C_out)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        if r == 2 and elem_size == 2 and pow2_dims and W >= 2 and n_elements < 2**31:
            _launch_pack16(input, output, r, C_out, C, H, W, n_elements)
        elif (
            r == 3
            and _is_pow2(W)
            and _is_pow2(H)
            and _is_pow2(C_out)
            and 3 * W <= _R3_LANE_MAX
            and n_elements <= _R3_LANE_MAX * W
        ):
            if elem_size == 2 and W >= 2:
                _launch_r3_pack16(input, output, r, C_out, C, H, W, n_elements)
            else:
                _launch_r3(input, output, r, C_out, C, H, W, n_elements)
        elif pow2_dims and n_elements < 2**31:
            _launch_pow2(input, output, r, C_out, C, H, W, n_elements)
        else:
            pixel_shuffle_kernel[grid](
                input,
                output,
                n_elements,
                C,
                H,
                W,
                r,
                C_out,
                H_out,
                W_out,
            )
    return output
