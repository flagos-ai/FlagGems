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

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy, tle_dma_available

logger = logging.getLogger(__name__)

# Above this many output elements, use the split path (SDNN-DMA interior /
# border copies) instead of the single-launch flat gather kernel. Measured
# crossover on fp16 with DMA-able borders: 0.59M -> flat 98 us vs split 152 us;
# 4.39M -> flat 632 us vs split 265 us, i.e. the split path's ~150 us of extra
# launches amortise from ~2.5M elements on.
_SPLIT_THRESHOLD = 2000000

# Lane tile of the flat gather kernel (one program per (image, tile)).
_FLAT_BLOCK = 1024


def _split_borders_ok(pad_left, pad_right):
    if pad_left == 1 and pad_right == 0:
        return False
    if pad_right == 1 and pad_left == 0:
        return False
    return True


# Flat 1D kernel over the ENTIRE output (all batches at once).
#
# ROOT CAUSE of the old slowness: the previous kernel wrapped every store index
# with `% HW_out` ("modulo wrap") to avoid masked stores. On KunlunXin XPU that
# runtime modulo defeats OffsetAnalysis, so EVERY load/store degrades to the
# discrete per-element path (~1.2 GB/s). Even a pure contiguous copy written with
# `%total` measured 228ms / 1.2 GB/s vs 0.49ms / 578 GB/s for the mask-based
# copy — a ~470x penalty (see reflection_pad2d_perf_fix.md).
#
# Fix: flatten (b, h_out, w_out) into one linear output index `o` and store to
# `o` directly (provably stride-1 -> block DMA). A single boolean mask
# `o < total_out` handles the tail. Because the layout is one flat contiguous
# buffer, the only masked-out threads sit at the very end (o >= total_out) and
# would write PAST the whole buffer — so even if a masked store were not
# suppressed it could not corrupt a valid element; and it is in fact suppressed
# here (verified maxdiff=0 on all shapes, including a tail-masked shape). This
# removes the "adjacent batch corruption" hazard that motivated the modulo wrap.
#
# The reflected input index is still a data-dependent gather (structural XPU
# wall), and the flat-index decode needs integer div/mod (slow on XPU), so the
# big shape stays ~40ms; but that is ~4.5x faster than the 183ms modulo version.
@triton.jit
def reflection_pad2d_kernel(
    in_ptr,
    out_ptr,
    H_in,
    W_in,
    pad_left,
    pad_top,
    W_out,
    HW_out,
    HW_in,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat output index -> (batch, h_out, w_out)
    b = o // HW_out
    rem = o % HW_out
    h_idx = rem // W_out
    w_idx = rem - h_idx * W_out

    # Reflected height index. pad_top < H_in is validated on the host, so a single
    # period (abs + where) is exact — no `% (2*(H_in-1))` needed.
    y = h_idx - pad_top
    pH = 2 * (H_in - 1)
    t_h = tl.abs(y)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    # Reflected width index (same reasoning; pad_left < W_in validated).
    x = w_idx - pad_left
    pW = 2 * (W_in - 1)
    t_w = tl.abs(x)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    in_offs = b * HW_in + ih * W_in + iw
    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    # Flat contiguous copy (no padding path). Mask-based, contiguous offsets ->
    # block DMA, same as the padded kernel's store side.
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def _row_copy_kernel(
    dst,
    src,
    ROWS_PER_IMG,
    L,
    SRC_IMG_STRIDE,
    DST_IMG_STRIDE,
    SRC_RS,
    DST_RS,
    BLOCK: tl.constexpr,
):
    # Copy ROWS_PER_IMG rows of L contiguous elements per leading-dims image:
    #   dst[b*DST_IMG + r*DST_RS + c] = src[b*SRC_IMG + r*SRC_RS + c]
    # for r in [0, ROWS_PER_IMG), c in [0, L). Both sides keep a provably
    # stride-1 inner axis (block DMA); only the image/row strides differ. The
    # per-image decomposition is required: in the padded-layout output the rows
    # are contiguous only *within one image* -- the next image starts after the
    # full padded plane -- so a single flat row stride would be wrong beyond
    # the first image. This is the fast primitive for the interior/border
    # copies of the split path (no gather, no native copy).
    pid = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    b = pid // ROWS_PER_IMG
    r = pid % ROWS_PER_IMG
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    vals = tl.load(src + b * SRC_IMG_STRIDE + r * SRC_RS + offs, mask=mask)
    tl.store(dst + b * DST_IMG_STRIDE + r * DST_RS + offs, vals, mask=mask)


def _row_copy(
    dst,
    src,
    rows_per_img,
    cols,
    src_img_stride,
    dst_img_stride,
    src_row_stride,
    dst_row_stride,
    n_img,
):
    BLOCK = 1024
    grid = (rows_per_img * n_img, triton.cdiv(cols, BLOCK))
    with torch_device_fn.device(dst.device):
        _row_copy_kernel[grid](
            dst,
            src,
            rows_per_img,
            cols,
            src_img_stride,
            dst_img_stride,
            src_row_stride,
            dst_row_stride,
            BLOCK=BLOCK,
        )


# Top/bottom strip kernel for the split path.
#
# The output of reflection_pad2d decomposes into:
#   * interior  [pad_top : pad_top+H_in, pad_left : pad_left+W_in]  -- a pure
#     strided copy of the input (handled by the `_row_copy` kernel),
#   * top/bottom strips -- (pad_top+pad_bottom) full output rows per (n, c),
#   * left/right edge columns of the interior rows (edge kernel + row copies).
#
# This kernel covers the top/bottom strips. A program is pinned to ONE (n, c)
# group and a BLOCK-sized window of one strip, so the lane-varying part of the
# store offset is exactly `blk * BLOCK + arange(BLOCK)` (provably stride-1 ->
# block DMA, same as the flat kernel's store). Only the reflected LOAD is a
# data-dependent gather, which is unavoidable for the border. Measured on XPU:
# any per-lane-computed store offset (scatter) costs ~2.2 ns/elem vs
# ~0.13 ns/elem for a provably contiguous store -- hence the scalar-nc grid.
#
# One launch covers both strips: programs [0, B*blocks_top) do the top strip
# (rows [0, pad_top)), the rest do the bottom strip (rows [pad_top+H_in, H_out)).
# Reflection here is single-period (validated pad < dim), so:
#   top row h_out=hh      -> input row pad_top - hh
#   bottom row h_out=..hh -> input row H_in - 2 - hh
@triton.jit
def reflection_pad2d_tb_kernel(
    in_ptr,
    out_ptr,
    H_in,
    W_in,
    pad_left,
    pad_top,
    W_out,
    HW_in,
    HW_out,
    B,
    per_nc,
    blocks_top,
    len_top,
    len_bot,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nc = pid // per_nc
    k = pid % per_nc
    is_top = k < blocks_top
    blk = tl.where(is_top, k, k - blocks_top)
    o = blk * BLOCK + tl.arange(0, BLOCK)
    strip_len = tl.where(is_top, len_top, len_bot)
    mask = o < strip_len

    hh = o // W_out
    w_out = o % W_out
    ih = tl.where(is_top, pad_top - hh, H_in - 2 - hh)
    x = w_out - pad_left
    t_w = tl.abs(x)
    pW = 2 * (W_in - 1)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    # Clamp load indices into range so the unmasked load address is always
    # in-bounds even for masked-off tail lanes (max_pool2d recipe).
    nc = tl.minimum(nc, B - 1)
    ih = tl.minimum(tl.maximum(ih, 0), H_in - 1)
    iw = tl.minimum(tl.maximum(iw, 0), W_in - 1)

    in_offs = nc * HW_in + ih * W_in + iw
    vals = tl.load(in_ptr + in_offs)
    h0 = tl.where(is_top, 0, pad_top + H_in)
    tl.store(out_ptr + nc * HW_out + h0 * W_out + o, vals, mask=mask)


# Left/right edge-column kernel for the split path: builds the reflected edge
# columns of the interior rows into a contiguous temp of shape (B, H_in, l+r).
# Column sc of the temp holds, for input row hrow:
#   sc < pad_left  -> input column pad_left - sc            (left edge, mirrored)
#   sc >= pad_left -> input column W_in - 2 - (sc - pad_left) (right edge)
# The temp is then copied into the output's two edge views by the `_row_copy`
# kernel (a direct triton store into those strided column runs cannot be
# proven contiguous and degrades to ~2 ns/elem scatter).
@triton.jit
def reflection_pad2d_edge_kernel(
    in_ptr,
    tmp_ptr,
    H_in,
    W_in,
    pad_left,
    HW_in,
    LR,
    B,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    nc = o // (H_in * LR)
    rem = o % (H_in * LR)
    hrow = rem // LR
    sc = rem % LR
    iw = tl.where(sc < pad_left, pad_left - sc, W_in - 2 - (sc - pad_left))

    # Clamp so the unmasked load address is always in-bounds (tail lanes are
    # dropped by the masked store).
    nc = tl.minimum(nc, B - 1)
    hrow = tl.minimum(hrow, H_in - 1)
    iw = tl.minimum(tl.maximum(iw, 0), W_in - 1)

    in_offs = nc * HW_in + hrow * W_in + iw
    vals = tl.load(in_ptr + in_offs)
    tl.store(tmp_ptr + o, vals, mask=mask)


def launch_reflection_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    # Validate reflection padding constraints
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        expected_shape = (*leading_shape, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    # No padding: just copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        BLOCK = 1024
        total = B * H_in * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    # BLOCK=1024 is the best all-round tile on XPU: small shapes avoid the
    # per-program waste of a huge block, while medium/large shapes still get
    # enough work per program to stay off the launch floor (measured sweep).
    BLOCK = _FLAT_BLOCK
    HW_out = H_out * W_out
    HW_in = H_in * W_in
    total_out = B * HW_out

    if (
        total_out >= _SPLIT_THRESHOLD
        and tle_dma_available()
        and _split_borders_ok(pad_left, pad_right)
    ):
        x3 = x.reshape(B, H_in, W_in)
        o3 = out.reshape(B, H_out, W_out)
        dma = True
        interior = o3[:, pad_top : pad_top + H_in, pad_left : pad_left + W_in]

        # 1) left/right edge columns of the interior rows: build the reflected
        #    columns into a contiguous temp, then move each side into its
        #    strided destination. Done BEFORE the interior copy because a
        #    one-column side is widened to two columns (a 2-D DMA transfer with
        #    an extent of 1 faults the engine) and the extra column lands on the
        #    first/last interior column, which the interior copy then rewrites.
        if pad_left + pad_right > 0:
            lr = pad_left + pad_right
            total_edge = B * H_in * lr
            tmp = torch.empty((B, H_in, lr), device=x.device, dtype=x.dtype)
            grid_edge = (triton.cdiv(total_edge, BLOCK),)
            with torch_device_fn.device(x.device):
                reflection_pad2d_edge_kernel[grid_edge](
                    x,
                    tmp,
                    H_in,
                    W_in,
                    pad_left,
                    HW_in,
                    lr,
                    B,
                    total_edge,
                    BLOCK=BLOCK,
                )
            sides = []
            if pad_left > 0:
                width = pad_left
                if width == 1 and lr >= 2:
                    # col 1 of the temp is a reflected value too, so the widened
                    # write stays in bounds; it is overwritten in step 2.
                    width = 2
                sides.append(
                    (
                        width,
                        tmp[:, :, 0:width],
                        o3[:, pad_top : pad_top + H_in, 0:width],
                    )
                )
            if pad_right > 0:
                width = pad_right
                src = tmp[:, :, lr - width : lr]
                if width == 1 and pad_left >= 1:
                    # the column before the right part is the last reflected
                    # left column: junk, overwritten by the interior copy.
                    width = 2
                    src = tmp[:, :, pad_left - 1 : pad_left + 1]
                sides.append(
                    (
                        width,
                        src,
                        o3[:, pad_top : pad_top + H_in, W_out - width : W_out],
                    )
                )
            for width, src_side, dst_side in sides:
                done = False
                if dma:
                    try:
                        done = tle_copy(src_side, dst_side)
                    except Exception:  # tle build without a usable dsa path
                        done = False
                if not done:
                    _row_copy(
                        dst_side,
                        src_side,
                        H_in,
                        width,
                        H_in * lr,
                        HW_out,
                        lr,
                        W_out,
                        B,
                    )

        # 2) interior: bulk strided copy of the input.
        interior_done = False
        if dma:
            try:
                interior_done = tle_copy(x3, interior)
            except Exception:  # tle build without a usable dsa path
                interior_done = False
        if not interior_done:
            _row_copy(interior, x, H_in, W_in, HW_in, HW_out, W_in, W_out, B)

        # 3) top/bottom strips: full output rows, contiguous store per program.
        if pad_top + pad_bottom > 0:
            len_top = pad_top * W_out
            len_bot = pad_bottom * W_out
            block_tb = min(
                8192, max(128, triton.next_power_of_2(max(len_top, len_bot)))
            )
            blocks_top = triton.cdiv(len_top, block_tb)
            blocks_bot = triton.cdiv(len_bot, block_tb)
            per_nc = blocks_top + blocks_bot
            grid_tb = (B * per_nc,)
            with torch_device_fn.device(x.device):
                reflection_pad2d_tb_kernel[grid_tb](
                    x,
                    out,
                    H_in,
                    W_in,
                    pad_left,
                    pad_top,
                    W_out,
                    HW_in,
                    HW_out,
                    B,
                    per_nc,
                    blocks_top,
                    len_top,
                    len_bot,
                    BLOCK=block_tb,
                )
        return out

    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad2d_kernel[grid](
            x,
            out,
            H_in,
            W_in,
            pad_left,
            pad_top,
            W_out,
            HW_out,
            HW_in,
            total_out,
            BLOCK=BLOCK,
        )
    return out


def reflection_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
