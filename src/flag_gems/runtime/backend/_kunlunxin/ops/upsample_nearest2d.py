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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
device = device.name

# Upper bound on the per-program BLOCK_SIZE (number of output lanes handled by
# one program). The unbounded `next_pow2(cdiv(total, 12))` produced tiles of up
# to 33.5M lanes for pathological test shapes, which both bloats the compiled IR
# and triggers the known XPU big-tile slowdown. A hard cap keeps every tile small
# and the grid bounded.
MAX_BLOCK_SIZE = 262144


# @triton.autotune(
#     configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
# )
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N: tl.constexpr,
    C: tl.constexpr,
    # OH/OW/IH/IW are the divisors of the index decomposition below.  On XPU a
    # vector integer divide or modulo by a *runtime* value is emulated per lane
    # and costs ~1.9ns per element, while the identical arithmetic with
    # constexpr divisors folds to shifts and multiplies and costs ~0.011ns --
    # the same as a kernel that does no index math at all.  Measured on card 5,
    # fp16, equal memory traffic: harness/scratch_up2d/probe_divmod.py.  That
    # single factor accounted for the whole gap to torch here, which is why
    # these are constexpr despite the per-shape recompile it implies.
    OH: tl.constexpr,
    OW: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    reciprocal_scale_h,
    reciprocal_scale_w,
    total,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)
    offset_o = ((n * C + c) * OH + oh) * OW + ow
    offset_i = ((n * C + c) * IH + ih) * IW + iw
    data = tl.load(ptr_i + offset_i)
    if NEED_MASK:
        # Tail block may extend past total output elements; guard the store.
        # (offset_i is always in-bounds: oh/ow/n/c are derived via modulo.)
        tl.store(ptr_o + offset_o, data, mask=idx < total)
    else:
        tl.store(ptr_o + offset_o, data)


@triton.jit
def upsample_nearest2d_int_scale_kernel(
    ptr_o,
    ptr_i,
    NC: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    total,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Exact-integer-scale variant: derive the source index by integer divide.

    The general kernel above gets `ih`/`iw` from a runtime float reciprocal
    scale followed by an int cast.  That makes the resulting address vector
    opaque to the compiler, which then lowers the load as a general gather:
    measured 0.25-0.60 ns per output element.  When the scale is an exact
    integer the same indices are `oh // SH` and `ow // SW` with constexpr
    divisors, which the compiler folds and recognises, giving 0.065-0.080 ns per
    output element -- a 5-7x drop for identical results.  Evidence:
    harness/scratch_up2d/probe_intscale.py.
    """
    idx = ext.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    nc = idx // OW // OH % NC
    # The output is freshly allocated and contiguous, so its flat offset is idx.
    offset_i = (nc * IH + oh // SH) * IW + ow // SW
    data = tl.load(ptr_i + offset_i)
    if NEED_MASK:
        tl.store(ptr_o + idx, data, mask=idx < total)
    else:
        tl.store(ptr_o + idx, data)


@triton.jit
def upsample_nearest2d_packed_kernel(
    ptr_o,
    ptr_i,
    NELEM,
    IWC: tl.constexpr,
    SH: tl.constexpr,
    NPACK: tl.constexpr,
    EBITS: tl.constexpr,
    WBITS: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Duplicate lanes in the *value* domain, and tile the input flat.

    Both of the kernels above address the input with `ow // SW`, a staircase
    vector (stride 0 inside each run of SW lanes).  Every attempt to express
    that duplication through addressing was measured and is strictly worse than
    this: a stride-SW store costs 4890us where the staircase load costs 190us,
    a 2-D tile store costs 147373us (2-D tile pointer arithmetic is lowered per
    lane even when the addresses it produces are exactly contiguous), and
    `tl.interleave` does not compile here (uni_sram OOM).  Evidence:
    harness/scratch_up2d/probe_bcast{,2,3,4,5}.py.

    So no irregular address is formed at all.  SW copies of one element are
    packed into one integer of width E*SW and stored as a single wide lane, so
    the store is plain unit-stride.  A packed container word covers SW output
    elements, hence exactly IW words per input row *for any SW*, and with
    OH = IH*SH the container index of input element f is

        g = (f // IW) * SH * IW + f % IW = f + (SH - 1) * IW * (f // IW)

    one constexpr divide; the SH vertical copies are `g + k * IW`.  The load
    side is then free to tile the input flat rather than by row, which matters
    more than anything else here: with row tiling one load moves IW elements
    (512B-4KB) and the kernel runs at 0.24-0.27 of torch, while a flat BLOCK
    chunk moves 128KB and reaches 0.90-1.30.  The store runs of IW lanes are
    block-DMA'd despite the jump between them.

    When SW == 1 there is nothing to duplicate; the host then packs NPACK=1 and
    widens the copy granularity instead by viewing both sides as a wider
    integer, which is the same instruction-size lever.
    """
    f = ext.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    m = f < NELEM
    if NEED_MASK:
        u = tl.load(ptr_i + f, mask=m, other=0)
    else:
        u = tl.load(ptr_i + f)
    if NPACK == 1:
        w = u
    else:
        # The narrow load sign-extends, so mask back to the element bits
        # before replicating them into the upper halves of the container.
        if WBITS == 64:
            u = u.to(tl.int64)
        else:
            u = u.to(tl.int32)
        if EBITS == 8:
            u = u & 0xFF
        elif EBITS == 16:
            u = u & 0xFFFF
        else:
            u = u & 0xFFFFFFFF
        w = u
        for j in tl.static_range(1, NPACK):
            w = w | (u << (j * EBITS))
    g = f + (SH - 1) * IWC * (f // IWC)
    for k in tl.static_range(SH):
        if NEED_MASK:
            tl.store(ptr_o + g + k * IWC, w, mask=m)
        else:
            tl.store(ptr_o + g + k * IWC, w)


_INT_OF_BYTES = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _packed_plan(elem_bytes, SH, SW, IW):
    """Pick the integer container for the packed kernel, or None.

    Returns (src_dtype, dst_dtype, IWC, NPACK, EBITS, WBITS, group) where
    `group` is how many *distinct* input elements one source word holds (always
    1 unless SW == 1).  The container must be 4 or 8 bytes: a 2-byte container
    would need int16 shifts, which are not worth the risk for the one shape
    they would add, and anything wider than 8 bytes has no integer type.
    """
    if SH < 1 or SW < 1:
        return None
    if SW > 1:
        wbytes = elem_bytes * SW
        if wbytes not in (4, 8):
            return None
        return (
            _INT_OF_BYTES[elem_bytes],
            _INT_OF_BYTES[wbytes],
            IW,
            SW,
            elem_bytes * 8,
            wbytes * 8,
            1,
        )
    # SW == 1: no horizontal duplication to do.  Still worth routing here --
    # the flat tiling and the wide container are what the speed comes from --
    # but the "packing" degenerates to reading several distinct neighbours per
    # word, which is valid because whole rows are replicated verbatim.
    for wbytes in (8, 4):
        if wbytes < elem_bytes or wbytes % elem_bytes:
            continue
        group = wbytes // elem_bytes
        if IW % group:
            continue
        wide = _INT_OF_BYTES[wbytes]
        return (wide, wide, IW // group, 1, 0, wbytes * 8, group)
    return None


def _try_packed(input, output, SH, SW, IW):
    """Run the packed kernel; return True on success, False to fall back."""
    if not input.is_contiguous():
        return False
    if input.numel() == 0:
        return False
    if SH > 32:
        # The vertical copies are a static unroll; past a few dozen the IR
        # bloat stops paying for itself.
        return False
    plan = _packed_plan(input.element_size(), SH, SW, IW)
    if plan is None:
        return False
    src_dtype, dst_dtype, IWC, NPACK, EBITS, WBITS, group = plan
    try:
        src = input.view(src_dtype)
        dst = output.view(dst_dtype)
    except (RuntimeError, ValueError):
        # A non-zero storage offset that is not a whole number of container
        # words makes the reinterpretation illegal.
        return False
    nelem = input.numel() // group
    # BLOCK is flat, not per-shape: swept over 2048..262144 x {fp16, fp32} x
    # five shapes (IW from 64 to 2048), 16384 container words is the best tile
    # in 9 of the 10 rows and within 6% of the best in the tenth, while 131072
    # and above lose 30-70%.  harness/scratch_up2d/probe_pkblock.py.
    block = min(16384, max(1024, triton.next_power_of_2(nelem)))
    grid = (triton.cdiv(nelem, block),)
    upsample_nearest2d_packed_kernel[grid](
        dst,
        src,
        nelem,
        IWC,
        SH,
        NPACK,
        EBITS,
        WBITS,
        BLOCK=block,
        NEED_MASK=(nelem % block != 0),
    )
    return True


def _integer_scales(OH, OW, IH, IW, scales_h, scales_w):
    """Return (SH, SW) when nearest indexing is an exact integer divide, else None.

    `oh // SH` equals torch's `floor(oh * IH / OH)` only when OH is a whole
    multiple of IH, and an explicitly supplied scale must agree with that
    multiple -- otherwise torch indexes with the given scale rather than with
    the shape ratio, and the two disagree (IH=4, OH=8, scales_h=4.0 maps oh=7 to
    1, not to 3).
    """
    if IH <= 0 or IW <= 0 or OH % IH or OW % IW:
        return None
    SH, SW = OH // IH, OW // IW
    if scales_h is not None and float(scales_h) != float(SH):
        return None
    if scales_w is not None and float(scales_w) != float(SW):
        return None
    return SH, SW


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape
    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW
    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = N * C * OH * OW
    # Bounded block size: keep the usual ~12-program cluster layout for small
    # outputs, but never build a huge single-program tile.
    block_size = min(
        triton.next_power_of_2(triton.cdiv(total_threads, 12)), MAX_BLOCK_SIZE
    )
    need_mask = total_threads % block_size != 0
    grid = (triton.cdiv(total_threads, block_size),)
    scales = _integer_scales(OH, OW, IH, IW, scales_h, scales_w)
    with torch_device_fn.device(input.device):
        if scales is not None:
            SH, SW = scales
            if _try_packed(input, output, SH, SW, IW):
                return output
            upsample_nearest2d_int_scale_kernel[grid](
                output,
                input,
                N * C,
                OH,
                OW,
                IH,
                IW,
                SH,
                SW,
                total_threads,
                BLOCK_SIZE=block_size,
                NEED_MASK=need_mask,
            )
            return output
        upsample_nearest2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            total_threads,
            SAME_H=(OH == IH),
            SAME_W=(OW == IW),
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
        )
    return output
