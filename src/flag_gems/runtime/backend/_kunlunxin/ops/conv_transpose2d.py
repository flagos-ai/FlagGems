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
"""Triton conv_transpose2d for Kunlunxin (XPU3).

Two XPU3 codegen properties drive this implementation (see the KernelWiki
entries `pattern-xpu3-gapped-access-3gbs-cliff` and
`pattern-xpu3-strided-store-miscompile`):

* the DMA only forms a fast transfer for a single contiguous run per program;
  a "gap" (row stride != row length, strided store, clamped index) collapses to
  ~3 GB/s per-lane accesses;
* a store whose lane stride is > 1 is *miscompiled* (the offset analysis assumes
  contiguity and spills whole sub-runs into the neighbouring row), so output
  stores must stay unit-stride.

Hence: lanes always enumerate a contiguous run of output columns (unit-stride
store, mask-free because the row is over-allocated to a whole number of tiles),
and the input is materialised into a zero-padded staging buffer so window taps
that fall outside the input read real zeros instead of needing a mask/clamp.

To keep the *loads* contiguous for `stride == 1`, the output rows are
decomposed by their `stride_h` phase (`oh = rh + stride_h*q`) and the phase is
passed as a `tl.constexpr`, which resolves the `kh` divisibility test at compile
time.  (A runtime `if` wrapped around the accumulation inside a
`tl.static_range` miscompiles here: the last iteration's predicate clobbers the
loop-carried accumulator.)  For `stride > 1` the column tap addresses are
strided by construction; they are applied with `tl.where` (no branch, no mask on
the load) and are the known-slow path.

`U` output channels share one input load (register reuse).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _ceil_div(a, b):
    return -(-a // b)


@triton.jit
def _pad_copy_kernel(src, dst, iw, iwp, boff_w, boff_h, ih, ihr, BLOCK: tl.constexpr):
    """Copy one source row into the interior of the zero-padded staging buffer.

    Grid: (n*c*ih, cdiv(iw, BLOCK)).  Both the load and the store are a single
    contiguous run per program.
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    mask = col < iw
    nc = row // ih
    r = row - nc * ih
    v = tl.load(src + row * iw + col, mask=mask, other=0.0)
    dst_row = nc * ihr + boff_h + r
    tl.store(dst + dst_row * iwp + boff_w + col, v, mask=mask)


@triton.jit
def _deconv_rows(
    xpad,
    w,
    bias,
    out,
    IC: tl.constexpr,
    COG: tl.constexpr,
    IHP: tl.constexpr,
    IWP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    BOFF_H: tl.constexpr,
    BOFF_W: tl.constexpr,
    OH: tl.constexpr,
    OWP: tl.constexpr,
    G: tl.constexpr,
    NCB: tl.constexpr,
    U: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RH: tl.constexpr,
):
    pid0 = tl.program_id(0)
    cb = pid0 % NCB
    rest = pid0 // NCB
    g = rest % G
    nidx = rest // G

    q = tl.program_id(1)
    oh = RH + q * SH
    if oh >= OH:
        return

    ow = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)

    acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK,), dtype=tl.float32)

    w_plane: tl.constexpr = COG * KH * KW
    x_plane: tl.constexpr = IHP * IWP
    taps: tl.constexpr = KH * KW
    x_base = xpad + (nidx * IC * G + g * IC) * x_plane
    y_stride = OH * OWP
    y_base = out + ((nidx * (COG * G) + g * COG + cb * U) * OH + oh) * OWP
    w_base = w + (g * IC) * w_plane + (cb * U) * taps

    for kh in tl.static_range(0, KH):
        NUM_H = RH + PH - kh * DH
        if NUM_H % SH == 0:
            ih = q + (NUM_H // SH) + BOFF_H
            for kw in tl.static_range(0, KW):
                NUM_W = PW - kw * DW
                if SW == 1:
                    iw = ow + NUM_W + BOFF_W
                    ok = ow >= 0
                else:
                    num_w = ow + NUM_W
                    iw = num_w // SW + BOFF_W
                    ok = num_w % SW == 0
                xp = x_base + (ih * IWP) + iw
                wc0 = kh * KW + kw
                for ci in tl.range(0, IC):
                    v = tl.load(xp + ci * x_plane).to(tl.float32)
                    if SW > 1:
                        v = tl.where(ok, v, 0.0)
                    wc = w_base + wc0 + ci * w_plane
                    acc0 += v * tl.load(wc).to(tl.float32)
                    if U > 1:
                        acc1 += v * tl.load(wc + taps).to(tl.float32)
                    if U > 2:
                        acc2 += v * tl.load(wc + 2 * taps).to(tl.float32)
                        acc3 += v * tl.load(wc + 3 * taps).to(tl.float32)

    if HAS_BIAS:
        ob = bias + g * COG + cb * U
        acc0 += tl.load(ob).to(tl.float32)
        if U > 1:
            acc1 += tl.load(ob + 1).to(tl.float32)
        if U > 2:
            acc2 += tl.load(ob + 2).to(tl.float32)
            acc3 += tl.load(ob + 3).to(tl.float32)

    op = y_base + ow
    tl.store(op, acc0.to(out.dtype.element_ty))
    if U > 1:
        tl.store(op + y_stride, acc1.to(out.dtype.element_ty))
    if U > 2:
        tl.store(op + 2 * y_stride, acc2.to(out.dtype.element_ty))
        tl.store(op + 3 * y_stride, acc3.to(out.dtype.element_ty))


_XPAD_CACHE = {}


def _padded_input(shape, dtype, device):
    key = (tuple(shape), dtype, str(device))
    t = _XPAD_CACHE.get(key)
    if t is None:
        # Allocation + zero initialisation only.  The interior is rewritten by
        # _pad_copy_kernel on every call, so the halo stays zero for good.
        t = torch.zeros(shape, dtype=dtype, device=device)
        _XPAD_CACHE[key] = t
    return t


def _pick_block(width):
    # NB: a program whose lane count is below the XPU3 core width is padded up
    # to it by the launcher, and the padded lanes spill into the *next* row of
    # the output (measured: BLOCK=32 duplicates every program's run).  Keep the
    # tile at least 64 lanes wide.
    block = 64
    while block < width and block < 512:
        block *= 2
    return block


_KCACHE = {}


def _ladder(ow, cog):
    """Candidate ``(U, BLOCK)`` pairs, most-preferred first.

    The XPU3 ``TritonXPUUnrollControl`` pass aborts compilation when the
    unrolled body (``taps * U`` copies of the inner loop) gets too large, so the
    exact tile that compiles depends on the geometry.  Walk a small ladder once
    per shape and memoise the winner; failing candidates are compile-time
    errors, so no partial output is produced.
    """
    native = _pick_block(ow)
    blocks = []
    for b in (native, max(64, native // 2), min(512, native * 2), 64, 128, 256):
        if b >= 64 and b not in blocks:
            blocks.append(b)
    cands = []
    for b in blocks:
        for u in (4, 2, 1):
            if cog % u == 0 and (u, b) not in cands:
                cands.append((u, b))
    return cands


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE2D")

    from flag_gems.ops.conv_transpose2d import (
        _pair,
        _unsupported_conv_transpose2d,
        _validate_conv_transpose2d_args,
    )

    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)

    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oph, opw = _pair(output_padding)
    dh, dw = _pair(dilation)

    # The validator rejects non-contiguous tensors, so normalise first: the
    # gather-free kernel only ever reads strided planes.
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    if not _validate_conv_transpose2d_args(
        input, weight, bias, sh, sw, ph, pw, oph, opw, groups, dh, dw
    ):
        _unsupported_conv_transpose2d(
            input, weight, bias, sh, sw, ph, pw, oph, opw, groups, dh, dw
        )

    n, c, ih, iw = input.shape
    ic = c // groups
    cog = weight.shape[1]
    kh, kw = weight.shape[2], weight.shape[3]
    co = cog * groups
    oh = (ih - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    ow = (iw - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1

    if n == 0 or co == 0 or oh <= 0 or ow <= 0:
        out = torch.empty(
            (n, co, max(oh, 0), max(ow, 0)), device=input.device, dtype=input.dtype
        )
        return out.squeeze(0) if input_was_unbatched else out

    # Halo geometry.  Every window offset a program can reach must land inside
    # the staging buffer, and the halo must be at least one real zero wide.
    boff_h = _ceil_div(max(0, (kh - 1) * dh - ph), sh) + 1
    boff_w = _ceil_div(max(0, (kw - 1) * dw - pw), sw) + 1
    qn = _ceil_div(oh, sh)
    # the widest offset the lane range can reach bounds the staging buffer, and
    # it only depends on OWP >= OW, so size it for the largest candidate block
    iwr = iw + boff_w + max(0, (512 - 1 + pw) // sw + boff_w + 2 - iw)
    ihr = ih + boff_h + max(0, (qn - 1) + (sh - 1 + ph) // sh + boff_h + 2 - ih)

    xpad = _padded_input((n, c, ihr, iwr), input.dtype, input.device)
    _pad_copy_kernel[(n * c * ih, _ceil_div(iw, 1024))](
        input, xpad, iw, iwr, boff_w, boff_h, ih, ihr, BLOCK=1024
    )

    key = (
        n,
        c,
        ih,
        iw,
        ic,
        cog,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        oh,
        ow,
        input.dtype,
    )
    cands = [_KCACHE[key]] if key in _KCACHE else _ladder(ow, cog)
    last_error = None
    for u, block in cands:
        tiles = _ceil_div(ow, block)
        owp = tiles * block
        out = torch.empty((n, co, oh, owp), device=input.device, dtype=input.dtype)
        bias_arg = bias if bias is not None else out
        grid0 = n * groups * (cog // u)
        try:
            for rh in range(min(sh, oh)):
                rows = _ceil_div(oh - rh, sh)
                _deconv_rows[(grid0, rows, tiles)](
                    xpad,
                    weight,
                    bias_arg,
                    out,
                    ic,
                    cog,
                    ihr,
                    iwr,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    boff_h,
                    boff_w,
                    oh,
                    owp,
                    groups,
                    cog // u,
                    u,
                    BLOCK=block,
                    HAS_BIAS=bias is not None,
                    RH=rh,
                )
        except triton.runtime.errors.OutOfResources as err:
            last_error = err
            continue
        _KCACHE[key] = (u, block)
        break
    else:
        raise last_error

    if owp != ow:
        out = out[..., :ow].contiguous()
    if input_was_unbatched:
        out = out.squeeze(0)
    return out
