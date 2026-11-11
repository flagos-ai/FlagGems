# Copyright 2026, The FlagOS Contributors.
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
"""Kunlunxin (XPU) override for ``aten._pad_circular``.

The generic ``flag_gems.ops._pad_circular`` kernels (1D/2D/3D) fail on XPU:

* The 2D kernel (2D ``(BLOCK_H, BLOCK_W)`` tile, ``rows[:, None] * Wo`` index
  arithmetic and a 2D load/store mask) cannot be converted to the XPU
  local-memory model: ``TritonXPULegalize`` aborts with
  ``'arith.muli' op requires the same type for all operands and results``
  (``loc("src_base")``), i.e. the same 2D-tile / uni_sram wall as
  ``linalg_eig`` -- even ``n=2`` fails.
* The 1D/3D kernels wrap with a two-step ``tl.where`` select
  (``sw = w - plw`` then ``+W`` / ``-W``). On XPU the data-dependent select is
  miscompiled: within each 8-lane group only the first two elements are
  correct and the rest read garbage, so every padded position is wrong (the
  test compares with ``assert_close(atol=0, rtol=0)``).

Fix: compute the circular source index *arithmetically* instead of by select,

    sw = (w + W - plw) % W

``w + W - plw`` is provably non-negative (``w >= 0`` and the host validation
``pl <= in_shape[d]`` gives ``W - plw >= 0``; crop pads, ``plw < 0``, only make
it larger), so the modulo is exact, never touches the negative-index or
data-dependent-select paths, and always yields ``sw`` in ``[0, W)`` -- the
load is in-bounds even for masked-off lanes. Verified on XPU: exact match with
``torch.ops.aten._pad_circular`` for all 10 shape/pad cases in
``tests/test_pad_circular.py`` (1D/2D/3D, positive and negative pads).

The 2D/3D cases use the row-per-program layout (scalar row index derived with
``//`` / ``%``, 1D ``w`` vector with ``w < Wo`` mask) -- the scalar-hoist
pattern of the generic 3D kernel, which is the only tile shape that
legalizes on XPU; the generic 2D kernel's 2D tile is replaced.

ATen dispatch reaches this vendor function through the ``SpecOpRegistrar``/
``_FULL_CONFIG`` mapping; ``flag_gems._pad_circular`` is therefore this
function on XPU and tests must call it via ``flag_gems._pad_circular``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _pad_circular_1d_kernel(
    x_ptr,
    out_ptr,
    W,
    Wo,
    plw,
    BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    wblk = tl.program_id(1)
    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < Wo
    # circular wrap of the last dim: arithmetic modulo, non-negative dividend.
    sw = (w + W - plw) % W
    src = b * W + sw
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + b * Wo + w, vals, mask=mask)


@triton.jit
def _pad_circular_2d_kernel(
    x_ptr,
    out_ptr,
    H,
    W,
    Ho,
    Wo,
    plh,
    plw,
    BLOCK_W: tl.constexpr,
):
    # rows enumerate the collapsed (batch, Ho) planes; one program per output
    # row, width vectorized (scalar-hoist pattern that legalizes on XPU).
    row = tl.program_id(0)
    wblk = tl.program_id(1)

    b = row // Ho
    h = row % Ho

    sh = (h + H - plh) % H
    src_base = (b * H + sh) * W

    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < Wo
    sw = (w + W - plw) % W
    src = src_base + sw
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + row * Wo + w, vals, mask=mask)


@triton.jit
def _pad_circular_3d_kernel(
    x_ptr,
    out_ptr,
    D,
    H,
    W,
    Do,
    Ho,
    Wo,
    pld,
    plh,
    plw,
    BLOCK_W: tl.constexpr,
):
    # rows enumerate the collapsed (batch, Do, Ho) planes; one program per
    # output row, width vectorized (scalar-hoist pattern that legalizes on
    # XPU). Depth/height wrap is scalar per program, width is vectorized.
    row = tl.program_id(0)
    wblk = tl.program_id(1)

    b = row // (Do * Ho)
    rem = row % (Do * Ho)
    d = rem // Ho
    h = rem % Ho

    sd = (d + D - pld) % D
    sh = (h + H - plh) % H
    src_base = ((b * D + sd) * H + sh) * W

    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < Wo
    sw = (w + W - plw) % W
    src = src_base + sw
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + row * Wo + w, vals, mask=mask)


def _pad_circular(x: torch.Tensor, pad):
    logger.debug("GEMS_KUNLUNXIN _PAD_CIRCULAR")

    ndim = x.dim()
    if len(pad) % 2 != 0:
        raise ValueError("padding length must be even")
    npad = len(pad) // 2
    if npad < 1 or npad > 3:
        raise ValueError(
            "flag_gems _pad_circular supports padding the last 1, 2 or 3 dimensions"
        )
    if ndim <= npad:
        raise ValueError("input must have at least one non-padded (batch) dimension")

    x = x.contiguous()
    in_shape = list(x.shape)
    out_shape = list(in_shape)
    pad_left = [0] * ndim
    for i in range(npad):
        d = ndim - 1 - i
        pl = int(pad[2 * i])
        pr = int(pad[2 * i + 1])
        new_size = in_shape[d] + pl + pr
        if new_size <= 0:
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{d}: input {in_shape[d]} pad ({pl}, {pr})"
            )
        # circular padding wraps at most once; guard the invariant the kernel relies on
        if pl > in_shape[d] or pr > in_shape[d]:
            raise ValueError("Padding value causes wrapping around more than once.")
        out_shape[d] = new_size
        pad_left[d] = pl

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    total = 1
    for s in out_shape:
        total *= s
    if total == 0:
        return out

    # BLOCK_W=1024 for the 1D-pad path (bandwidth-bound copy); the 2D/3D
    # row-per-program paths use at most 1024 lanes and never waste more than a
    # factor of two of them on a row.
    if npad == 1:
        W = in_shape[-1]
        Wo = out_shape[-1]
        plw = pad_left[-1]
        B = 1
        for s in in_shape[:-1]:
            B *= s
        BLOCK_W = 1024
        grid = (B, triton.cdiv(Wo, BLOCK_W))
        with torch_device_fn.device(x.device):
            _pad_circular_1d_kernel[grid](
                x,
                out,
                W,
                Wo,
                plw,
                BLOCK_W=BLOCK_W,
            )
    elif npad == 2:
        H = in_shape[-2]
        W = in_shape[-1]
        Ho = out_shape[-2]
        Wo = out_shape[-1]
        plh = pad_left[-2]
        plw = pad_left[-1]
        B = 1
        for s in in_shape[:-2]:
            B *= s
        R = B * Ho
        BLOCK_W = min(1024, triton.next_power_of_2(max(Wo, 1)))
        grid = (R, triton.cdiv(Wo, BLOCK_W))
        with torch_device_fn.device(x.device):
            _pad_circular_2d_kernel[grid](
                x,
                out,
                H,
                W,
                Ho,
                Wo,
                plh,
                plw,
                BLOCK_W=BLOCK_W,
            )
    else:  # npad == 3
        D = in_shape[-3]
        H = in_shape[-2]
        W = in_shape[-1]
        Do = out_shape[-3]
        Ho = out_shape[-2]
        Wo = out_shape[-1]
        pld = pad_left[-3]
        plh = pad_left[-2]
        plw = pad_left[-1]
        B = 1
        for s in in_shape[:-3]:
            B *= s
        R = B * Do * Ho
        BLOCK_W = min(1024, triton.next_power_of_2(max(Wo, 1)))
        grid = (R, triton.cdiv(Wo, BLOCK_W))
        with torch_device_fn.device(x.device):
            _pad_circular_3d_kernel[grid](
                x,
                out,
                D,
                H,
                W,
                Do,
                Ho,
                Wo,
                pld,
                plh,
                plw,
                BLOCK_W=BLOCK_W,
            )
    return out
