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

from flag_gems.runtime import torch_device_fn

_EMAX = {"e4m3": 8, "e5m2": 15, "fp4": 2}
_XMAX = {"e4m3": 448.0, "e5m2": 57344.0, "fp4": 6.0}
# fp32 bit patterns of the destination magnitudes (448.0f, 57344.0f)
_XMAX_BITS = {"e4m3": 0x43E00000, "e5m2": 0x47600000}


def _resolve_format(dst_type):
    """Map the reference's dst_type argument (torch dtype object or ScalarType int)."""
    if isinstance(dst_type, torch.dtype):
        if dst_type == torch.float8_e4m3fn:
            return "e4m3"
        if dst_type == torch.float8_e5m2:
            return "e5m2"
        if dst_type in (torch.float4_e2m1fn_x2, torch.uint8, torch.int8):
            return "fp4"
        raise ValueError("unsupported dst_type %r" % (dst_type,))
    if dst_type in (24, 292):
        return "e4m3"
    if dst_type == 296:
        return "fp4"
    raise ValueError("unsupported dst_type %r" % (dst_type,))


@triton.jit
def _e8m0_scale(amax, EMAX: tl.constexpr, ALG: tl.constexpr):
    """Exact per-block E8M0 byte and power-of-two scale from the block amax."""
    bits = amax.to(tl.int32, bitcast=True)
    expf = (bits >> 23) & 0xFF
    e = expf - 127 - EMAX
    if ALG == 1:
        mant = bits & 0x7FFFFF
        e = e + tl.where(mant > 6291456, 1, 0)
    # no `amax > 0` select: for amax == 0 the exponent field is 0, so
    # e + 127 == -EMAX <= 0 and the clamp already yields byte 0, and the
    # reconstructed 2**127 scale multiplied by an all-zero tile is exactly 0.
    sbyte = tl.minimum(tl.maximum(e + 127, 0), 254)
    ebits = tl.minimum(tl.maximum(127 - e, 1), 254)
    scale = (ebits << 23).to(tl.float32, bitcast=True)
    return sbyte, scale


@triton.jit
def _sat_bits(y, XMAXB: tl.constexpr):
    """Magnitude saturation to +-XMAX in the integer bit domain (exact)."""
    b = y.to(tl.int32, bitcast=True)
    mag = b & 0x7FFFFFFF
    sign = b - mag
    return (tl.minimum(mag, XMAXB) + sign).to(tl.float32, bitcast=True)


@triton.jit
def _mxq_fp8(
    x_ptr,
    o_ptr,
    s_ptr,
    n_elem,
    IS_E4M3: tl.constexpr,
    ALG: tl.constexpr,
    BLOCKS: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    TILE: tl.constexpr = BLOCKS * 32
    base = pid * TILE
    j = tl.arange(0, BLOCKS)
    # flat contiguous tile: the lane axis of both the DMA and the compute tile is
    # memory-consecutive, so there is no strided load and no output gather.
    offs = base + tl.arange(0, TILE)
    if MASKED:
        mask = offs < n_elem
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
    x2 = tl.reshape(x.to(tl.float32), (BLOCKS, 32))
    # integer-domain block max: for non-negative IEEE-754 values the bit pattern
    # orders exactly like the value, so masking the sign bit and doing an integer
    # max is the same reduction without the float |.| pass and its NaN->+inf
    # remap, and it drops straight into the unchanged exponent slice below.
    amax = tl.max(x2.to(tl.int32, bitcast=True) & 0x7FFFFFFF, axis=1)
    amax = amax.to(tl.float32, bitcast=True)
    if IS_E4M3:
        sbyte, scale = _e8m0_scale(amax, 8, ALG)
        y = _sat_bits(x2 * scale[:, None], 0x43E00000)
        q = tl.reshape(y, (TILE,)).to(tl.float8e4nv)
    else:
        sbyte, scale = _e8m0_scale(amax, 15, ALG)
        y = _sat_bits(x2 * scale[:, None], 0x47600000)
        q = tl.reshape(y, (TILE,)).to(tl.float8e5)
    s = sbyte.to(tl.uint8)
    blk = base // 32 + j
    if MASKED:
        tl.store(o_ptr + offs, q, mask=mask)
        tl.store(s_ptr + blk, s, mask=blk < (n_elem + 31) // 32)
    else:
        tl.store(o_ptr + offs, q)
        tl.store(s_ptr + blk, s)


@triton.jit
def _mxq_fp4(
    x_ptr,
    o_ptr,
    s_ptr,
    n_elem,
    ALG: tl.constexpr,
    BLOCKS: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    TILE: tl.constexpr = BLOCKS * 32
    base = pid * TILE
    offs = base + tl.arange(0, TILE)
    if MASKED:
        mask = offs < n_elem
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
    x2 = tl.reshape(x.to(tl.float32), (BLOCKS, 32))
    # |x| serves both the block max and the quantized magnitude; the scale is a
    # strictly positive power of two, so sign(x * scale) == sign(x) and the sign
    # bit can be taken from x directly instead of from a second signed product.
    ax = tl.abs(x2)
    amax = tl.max(ax.to(tl.int32, bitcast=True), axis=1)
    amax = amax.to(tl.float32, bitcast=True)
    sbyte, scale = _e8m0_scale(amax, 2, ALG)
    m = tl.minimum(ax * scale[:, None], 6.0)
    # e2m1 magnitude codes are 0,.5,1,1.5,2,3,4,6: one magic round of a
    # piecewise-linear map, c = round_even(m * s) + b with (s,b) chosen per
    # segment, instead of three separately materialized segment formulas.
    s = tl.where(m < 2.0, 2.0, tl.where(m < 4.0, 1.0, 0.5))
    mo = tl.where(m < 2.0, 12582912.0, tl.where(m < 4.0, 12582914.0, 12582916.0))
    c = (m * s + mo) - 12582912.0
    code = c.to(tl.int32)
    sign = (x2.to(tl.int32, bitcast=True) >> 31) & 1
    code = code | (sign << 3)
    # nibble pack: a pair-axis split plus one shift and one or, instead of a
    # materialized (BLOCKS*16, 2) product followed by a width-2 reduction.
    lo, hi = tl.split(tl.reshape(code, (BLOCKS * 16, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    poffs = base // 2 + tl.arange(0, BLOCKS * 16)
    blk = base // 32 + tl.arange(0, BLOCKS)
    s = sbyte.to(tl.uint8)
    if MASKED:
        tl.store(o_ptr + poffs, packed, mask=poffs < (n_elem + 1) // 2)
        tl.store(s_ptr + blk, s, mask=blk < (n_elem + 31) // 32)
    else:
        tl.store(o_ptr + poffs, packed)
        tl.store(s_ptr + blk, s)


def _pick_blocks(total_blocks):
    if total_blocks <= 2048:
        return 8
    if total_blocks <= 16384:
        return 32
    return 128


def _launch(
    input,
    *,
    axis=-1,
    round_mode="rint",
    dst_type=296,
    block_size=32,
    scale_alg=0,
    dst_type_max=0.0,
    max_low_bound=0.0,
):
    fmt = _resolve_format(dst_type)
    x = input
    nd = x.dim()
    ax = axis + nd if axis < 0 else axis
    flipped = False
    if ax != nd - 1:
        x = x.transpose(ax, nd - 1).contiguous()
        flipped = True
    k = x.shape[nd - 1]
    bs = int(block_size)
    nblk = (k + bs - 1) // bs
    scale_rows = (nblk + 1) // 2
    out_shape1 = x.shape[:-1] + (scale_rows, 2)
    n_rows = x.numel() // k if k else 0
    total_blocks = n_rows * nblk
    alg = 1 if scale_alg == 1 else 0

    if fmt == "fp4":
        out0 = torch.empty(x.shape[:-1] + (k // 2,), dtype=torch.uint8, device=x.device)
    else:
        dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
        out0 = torch.empty(x.shape, dtype=dtype, device=x.device)

    if k % bs == 0 and nblk % 2 == 0:
        out1 = torch.empty(total_blocks, dtype=torch.uint8, device=x.device).view(
            out_shape1
        )
    else:
        out1 = torch.zeros(out_shape1, dtype=torch.uint8, device=x.device)

    if total_blocks == 0:
        return out0, out1

    blocks = _pick_blocks(total_blocks)
    grid = (triton.cdiv(total_blocks, blocks),)
    masked = (total_blocks % blocks) != 0
    if fmt == "fp4":
        _mxq_fp4[grid](x, out0, out1, x.numel(), alg, blocks, masked)
    else:
        _mxq_fp8[grid](x, out0, out1, x.numel(), fmt == "e4m3", alg, blocks, masked)

    if flipped:
        out0 = out0.transpose(ax, nd - 1).contiguous()
        out1 = out1.permute(list(range(nd - 2)) + [nd - 1, nd - 2]).contiguous()
    return out0, out1


def npu_dynamic_mx_quant(
    input,
    *,
    axis=-1,
    round_mode="rint",
    dst_type=296,
    block_size=32,
    scale_alg=0,
    dst_type_max=0.0,
    max_low_bound=0.0,
):
    """Per-block dynamic MX quantization to E4M3 or packed E2M1 with E8M0 scales."""
    if input.dtype != torch.bfloat16 or input.ndim < 1 or not input.is_contiguous():
        raise ValueError("expected a contiguous BF16 activation")
    if (
        axis not in (-1, input.ndim - 1)
        or block_size != 32
        or input.shape[-1] % 64
        or not input.numel()
    ):
        raise ValueError(
            "expected complete pairs of 32-element MX blocks along the last axis"
        )
    if (
        dst_type not in (torch.float8_e4m3fn, 24, 296)
        or scale_alg not in (0, 1)
        or round_mode != "rint"
        or dst_type_max != 0
        or max_low_bound != 0
    ):
        raise ValueError(
            "supported modes are E4M3 and packed E2M1, with rint and default saturation bounds"
        )

    if dst_type == 296 and scale_alg != 0:
        raise ValueError("packed E2M1 requires scale_alg=0")
    if input.device.type != "npu":
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(input.device):
        return _launch(
            input,
            axis=axis,
            round_mode=round_mode,
            dst_type=dst_type,
            block_size=block_size,
            scale_alg=scale_alg,
            dst_type_max=dst_type_max,
            max_low_bound=max_low_bound,
        )
