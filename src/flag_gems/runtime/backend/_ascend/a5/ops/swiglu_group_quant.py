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

"""SwiGLU fused with group quantisation to FP8, for Ascend.

Replaces ``aclnnSwigluGroupQuant`` for the FP8 output modes, which is what
vllm-ascend actually calls on this chip:

  * ``torch.ops._C_ascend.npu_swiglu_group_quant`` (vllm-ascend ``csrc/moe/
    swiglu_group_quant``) from ``ops/fused_moe/shared_experts.py`` and
    ``quantization/methods/w4a8/w4a8_mxfp4.py``.  Note the mode numbering there
    is shifted by one: its ``quant_mode=2`` is MX, i.e. ``quant_mode=1`` here
    and in the CANN interface.
  * ``cann_ops_nn.swiglu_group_quant`` -> ``aclnnSwigluGroupQuant``, the vendor
    this module was measured against.

Computation, per token row, with ``H = D / 2``::

    x0, x1   = x[..., :H], x[..., H:]
    x0, x1   = min(x0, c), clamp(x1, -c, c)           # if clamp_limit > 0
    y_origin = x0 / (1 + exp(-x0)) * x1               # SwiGLU
    y        = y_origin * weight[row]                 # if weight given
    amax     = max(|y|) over each block, >= 1e-4      # block = 128 or 32
    y_fp8    = round_to_fp8(y / scale),  scale ~ amax / fp8_max

with three shapes of the scale, selected by ``(quant_mode, round_scale)``:

    mode 0, round_scale=False   fp32 scale = amax/fp8_max, block 128
    mode 0, round_scale=True    fp32 scale, rounded up to a power of 2
    mode 1, round_scale=True    uint8 E8M0 exponent, block 32   <- the MX mode
                                vllm-ascend uses in production

Modes 2 and 3 (HiFloat8 static/dynamic) are not implemented: Triton has no
hifloat8 type, and emitting its bit layout by hand would cost more than the
operator.  Use the vendor there.

Accuracy: every output is **byte-exact** against the vendor over the sweep this
kernel was developed under -- the fp8 payload, the E8M0 scale bytes, the fp32
scale (both ``round_scale`` settings), and ``y_origin``, for both input dtypes
and both fp8 output types, with and without weight / clamp / group_index, and on
inf / nan / zero / saturating edge blocks.

That includes ``mode 0, round_scale=False``, whose fp32 scale carries the fp32
rounding of the SwiGLU straight through with nothing to absorb it, and it still
matches at 0 ULP: ``tl.exp`` lowers to the same vector instruction the vendor's
``VFSwiGlu`` uses.  Computing the same fp32 SwiGLU with *torch* on NPU instead
differs from the vendor by up to 3 ULP (both within 2 ULP of an fp64
reference) -- so a torch reference will report a few-ULP scale difference on
that one path, and the thing that disagrees is ``aclnnExp``, not this kernel.
That is why ``tests/test_swiglu_group_quant.py`` keeps a ULP band on the fp32
scale and a tie rate on the fp8 payload while comparing byte-exactly whenever
the vendor is importable.

``H = 1280`` (D = 2560) at large T is the one shape the vendor keeps: 1280 is
not a power of two, Triton tile widths must be, so the widest legal chunk
(1024) covers 2048 columns in two steps and part of the work is redundant.
``should_use_triton`` returns False there.

This file sits under ``a5/ops`` rather than ``_ascend/ops`` because the tile
thresholds in ``tile_cols`` and the cost models behind ``should_use_triton``
were fitted on Ascend950 (NpuArch DAV_3510).  Atlas A2/A3 keep calling the
vendor.  The core count itself is read from the driver, so a part with a
different one still gets the right grid.

Design notes:

  * the per-block amax is a plain ``tl.max`` over a ``(rows*nblk, B)`` reshape;
    the reshape is free because the block axis is the contiguous one.
  * ``group_index`` is summed **inside** the kernel (G is 1..256 int64 values,
    so a few instructions) rather than on the host.  A host-side sum would need
    a device sync per call, which breaks ACL graph capture.  Rows past
    ``real_bs`` are skipped by the loop bound, so a short ``group_index`` makes
    the kernel proportionally cheaper, matching the vendor.
  * a column chunk must either cover the whole row or stay at least 1024 wide,
    and must never exceed H.  ``tile_cols`` encodes both; it is a DMA
    descriptor effect, not arithmetic, and it is the constraint that shapes the
    whole tiling.
  * the sign-of-zero fixup adds the kernel's ``ZERO`` argument, not a literal
    ``0.0``, because the literal is folded away as an identity.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._ascend.utils import CORE_NUM

try:
    # Direct C binding, skipping the backend_register/python wrapper chain.
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream
except ImportError:  # pragma: no cover - older torch_npu
    try:
        from torch_npu._C import _npu_getCurrentRawStream as _raw_stream
    except ImportError:
        _raw_stream = None

logger = logging.getLogger(__name__)

__all__ = [
    "SwigluGroupQuantPlan",
    "swiglu_group_quant",
]


# The kernel multiplies the amax by a *rounded* fp32 1/fp8max instead of
# dividing by fp8max (INV_FP8_E4M3_MAX_VALUE / INV_FP8_E5M2_MAX_VALUE in the
# vendor's swiglu_group_quant_base.h).  Reproducing the exact constant matters:
# it decides the scale exponent at the boundary, and a division would round
# differently.
_INV_FP8_MAX = {
    torch.float8_e4m3fn: 0x3B124925,
    torch.float8_e5m2: 0x37924925,
}
_FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
_BLOCK = {0: 128, 1: 32}
_IN_DTYPES = (torch.bfloat16, torch.float16)

# Largest tile in output elements (rows * cols).  A tile holds x0 and x1 as
# fp32, the fp32 SwiGLU, and the fp8 output, so ~16 bytes of UB per output
# element; 16384 is the largest that compiles at every (rows, cols) shape.
# 32768 does not fit.
_UB_TILE_ELEMS = 16384
# Rows per tile.  Large row counts only matter at small H, where a whole row is
# cheap.
_ROWS = (1, 2, 4, 8, 16, 32)
# A column chunk narrower than this collapses unless it covers the whole row.
# It is a DMA-descriptor cliff, not a tuning knob; see tile_cols.
_MIN_STRIDED_COLS = 1024
_DISPATCH_US = 16.0  # aclnn host dispatch the vendor pays in eager mode
# Compiled plans, keyed by specialization; see _plan_key.  Bounded because every
# entry holds a CompiledKernel.
_PLAN_CACHE = {}
_PLAN_CACHE_MAX = 64


@triton.jit
def _sgq_kernel(
    x_ptr,
    w_ptr,
    gi_ptr,
    y_ptr,
    s_ptr,
    yo_ptr,
    M,
    ZERO,
    D: tl.constexpr,
    H: tl.constexpr,
    NBLK: tl.constexpr,
    NSP: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    B: tl.constexpr,
    CB: tl.constexpr,
    INVB: tl.constexpr,
    FMAX: tl.constexpr,
    MX: tl.constexpr,
    RS: tl.constexpr,
    HASW: tl.constexpr,
    HASGI: tl.constexpr,
    CLAMP: tl.constexpr,
    CL: tl.constexpr,
    E5: tl.constexpr,
    ORIGIN: tl.constexpr,
    RMASK: tl.constexpr,
    STEPS: tl.constexpr,
    OVERLAP: tl.constexpr,
    PROGS: tl.constexpr,
):
    """One program walks ROWS rows at a time, striding by PROGS * ROWS.

    ``ZERO`` is always 0.0 and is an argument rather than a constant on purpose;
    see the sign-of-zero fixup in the inner loop.

    ``COLS`` columns of the output are handled per inner step, ``CB = COLS / B``
    blocks, ``STEPS`` steps to cover a row.  ``RMASK`` is set when the row count
    is not a multiple of ROWS or when ``group_index`` makes it dynamic.
    ``OVERLAP`` is set when COLS does not divide H, which happens whenever H is
    not a power of two (D = 2560 -> H = 1280); see the inner loop.
    """
    pid = tl.program_id(0)
    N: tl.constexpr = ROWS * CB
    # 1/fp8max as a bit pattern, so the fp32 constant is the vendor's exactly.
    inv_fp8 = tl.full((), INVB, tl.int32).to(tl.float32, bitcast=True)

    real = M
    if HASGI:
        # group_index is a count per expert; the operator processes only the
        # first sum(group_index) rows and leaves the rest of the output buffer
        # alone.  Summed here rather than on the host to avoid a device sync.
        g = tl.arange(0, GP)
        gv = tl.load(gi_ptr + g, mask=g < G, other=0).to(tl.int32)
        real = tl.minimum(tl.sum(gv), M)
        real = tl.maximum(real, 0)

    c = tl.arange(0, COLS)
    bl = tl.arange(0, CB)
    for r0 in range(pid * ROWS, real, PROGS * ROWS):
        r = r0 + tl.arange(0, ROWS)
        rm = (r < real)[:, None]
        xb = x_ptr + r[:, None] * D
        if HASW:
            # Hoisted: the weight is per row, so loading it per column chunk
            # would repeat it STEPS times for no reason.
            wv = tl.load(w_ptr + r, mask=r < real, other=0.0)[:, None]
        for ci in range(0, STEPS):
            # When COLS does not divide H, the last chunk *overlaps* the one
            # before instead of running past the row and masking the overhang.
            # A column mask is ruinous here: it stops the DMA using its
            # contiguous path.  Overlapping recomputes some blocks, which is
            # safe because a block's amax, scale and payload depend only on
            # that block, so the second write is identical to the first.
            c0 = ci * COLS
            if OVERLAP:
                c0 = min(c0, H - COLS)
            off = (c0 + c)[None, :]
            if RMASK:
                x0 = tl.load(xb + off, mask=rm, other=0.0).to(tl.float32)
                x1 = tl.load(xb + H + off, mask=rm, other=0.0).to(tl.float32)
            else:
                x0 = tl.load(xb + off).to(tl.float32)
                x1 = tl.load(xb + H + off).to(tl.float32)
            if CLAMP:
                x0 = tl.minimum(x0, CL)
                x1 = tl.minimum(tl.maximum(x1, -CL), CL)
            # Spelled exactly as the vendor's VFSwiGlu (neg, exp, add, div, mul);
            # x0 * sigmoid(x0) differs by up to 2 ULP and would lose byte parity.
            y = (x0 / (tl.exp(-x0) + 1.0)) * x1

            if ORIGIN:
                # y_origin is the SwiGLU *before* the weight, in x's dtype.
                yov = y.to(yo_ptr.dtype.element_ty)
                if RMASK:
                    tl.store(yo_ptr + r[:, None] * H + off, yov, mask=rm)
                else:
                    tl.store(yo_ptr + r[:, None] * H + off, yov)
            if HASW:
                y = y * wv
            # Normalise the sign of zero: -0.0 + 0.0 == +0.0.  The vendor's fp8
            # payload for an exact -0.0 is byte 0x00, while a tiny negative that
            # merely underflows keeps its sign (0x80 on both sides) -- so this is
            # a zero-only fixup that neither the cast nor a patch of the output
            # bytes can do.  ZERO is a *runtime* argument, always 0.0, because a
            # literal `y + 0.0` is folded away as an identity, leaving 0x80
            # where the vendor gives 0x00.  The bit-test spelling
            # `where(bitcast(y) == INT32_MIN, 0, ...)` is correct too but
            # slower.
            # y_origin is stored above, before this, because the vendor's
            # y_origin does keep the -0.0 (bf16 bits 0x8000).
            y = y + ZERO

            yt = tl.reshape(y, (N, B))
            if MX:
                amax = tl.max(tl.abs(yt), 1)
            else:
                # mode 0 masks inf/nan out of the amax (the kernel's
                # `y * 0 != y * 0` test) instead of letting them poison it.
                fin = (yt * 0.0) == (yt * 0.0)
                amax = tl.max(tl.where(fin, tl.abs(yt), 0.0), 1)
            # The vendor floors the amax at 1e-4 before deriving the scale.
            amax = tl.maximum(amax, 1e-4)
            sf = amax * inv_fp8
            if RS:
                sb = sf.to(tl.int32, bitcast=True)
                # ceil(log2(sf)) straight off the fp32 fields, as the vendor
                # does: exponent - 127, plus one if any mantissa bit is set.
                e = ((sb >> 23) & 0xFF) - 127 + ((sb & 0x7FFFFF) != 0).to(tl.int32)
                inv = (((127 - e) & 0xFF) << 23).to(tl.float32, bitcast=True)
            else:
                inv = FMAX / amax

            q = yt * inv[:, None]
            if not MX:
                # A scaled value that overflowed to inf falls back to unscaled.
                q = tl.where((q * 0.0) == (q * 0.0), q, yt)
            q = tl.reshape(q, (ROWS, COLS))
            if E5:
                q8 = q.to(tl.float8e5)
            else:
                q8 = q.to(tl.float8e4nv)

            yo = y_ptr + r[:, None] * H + off
            sp = (c0 // B + bl)[None, :]
            if MX:
                sv = tl.reshape((e + 127).to(tl.uint8), (ROWS, CB))
                so = s_ptr + r[:, None] * NSP + sp
            elif RS:
                sv = tl.reshape(
                    (((e + 127) & 0xFF) << 23).to(tl.float32, bitcast=True),
                    (ROWS, CB),
                )
                so = s_ptr + r[:, None] * NBLK + sp
            else:
                sv = tl.reshape(sf, (ROWS, CB))
                so = s_ptr + r[:, None] * NBLK + sp
            if RMASK:
                tl.store(yo, q8, mask=rm)
                tl.store(so, sv, mask=rm)
            else:
                tl.store(yo, q8)
                tl.store(so, sv)


def scale_shape(shape, quant_mode, round_scale=None):
    """Vendor scale layout for ``x`` of shape ``shape``.

    mode 1 (MX): uint8 viewed as E8M0, ``[..., ceil(nblk/2), 2]`` -- i.e. a
    contiguous ``[..., nblk]`` padded to an even byte count, pad byte 0.
    mode 0: fp32 ``[..., nblk]``.
    """
    H = shape[-1] // 2
    nblk = -(-H // _BLOCK[quant_mode])
    if quant_mode == 1:
        return tuple(shape[:-1]) + ((nblk + 1) // 2, 2)
    return tuple(shape[:-1]) + (nblk,)


def tile_cols(H, quant_mode=1):
    """Legal column counts per step, widest first.

    Triton needs a power-of-two tile width, so a chunk is a power of two and at
    most ``H`` -- going over would need a column mask, which is ruinous (see the
    kernel's inner loop).

    The floor is **1024 columns**, and that is not a tuning choice: a chunk
    narrower than that falls off a cliff whenever it does not cover the whole
    row.  ``cols == H`` is always fine -- each row's chunk is then contiguous
    and the load is one descriptor per row -- but a *strided* chunk of 512 or
    fewer columns collapses by an order of magnitude.  It is not a row-count
    effect; more rows per tile does not recover it.

    ``top`` is therefore always offered even when it is below the floor, which
    is what makes H = 128/256/512 work at all.
    """
    B = _BLOCK[quant_mode]
    top = 1 << (H.bit_length() - 1)  # largest power of two <= H
    out = [top]
    c = top // 2
    while c >= _MIN_STRIDED_COLS and c >= B:
        out.append(c)
        c //= 2
    return out


def est_kernel_us(M, H, rows, cols):
    """Cost model behind pick_tile: a pipeline fill, then steady-state steps.

    Fitted to the tables in ``pick_tile``::

        fill   = 1.20 us + 0.024 us * cores_busy
        step   = 0.14 us + 0.242 us per 1000 tile elements
        total  = fill + steps_per_core * step

    The shape matters more than the constants.  Cost is *not* proportional to
    waves of launches: the steps after the first pipeline against each other, so
    a core walking many steps pays the fill once.  What does scale with
    occupancy is the fill, because concurrently starting cores contend.  That is
    why ``rows=1`` loses above T=64 despite filling every core: it buys
    occupancy that was already saturated and pays the contention for it.

    The constants were fitted on a 64-vector-core part; ``CORE_NUM`` is read
    from the device, so a part with a different core count still gets the right
    occupancy term even though the fit is not re-derived for it.

    The model only describes tiles ``tile_cols`` offers.  It does **not** know
    about the narrow-strided-chunk cliff, which would swamp everything here;
    that is excluded by construction instead of fitted.
    """

    groups = -(-M // rows)
    cores = min(groups, CORE_NUM)
    waves = -(-groups // CORE_NUM)
    # ceil, so a cols that does not divide H prices its overlapping last chunk
    # as a full step -- which is what it costs.
    steps = waves * -(-H // cols)
    return 1.20 + 0.024 * cores + steps * (0.14 + 0.242e-3 * rows * cols)


def pick_tile(M, H, quant_mode=1):
    """(rows, cols) per tile: minimise the modelled cost over legal tiles.

    The free parameters are the rows per tile and the output columns per inner
    step, under ``rows * cols <= 16384`` elements.  Two regimes:

      * below one tile per core, only the launch cost matters, so the widest
        single-row tile wins -- extra rows just idle cores.
      * once every core has work, cost is proportional to elements per core and
        the largest tile that fits wins, because the fill is paid once.  Which
        row count that implies depends on H: 8 rows at H=2048, 32 at H=512.

    Two traps, both handled by ``tile_cols`` rather than by the cost model:

      * a strided chunk of <= 512 columns collapses;
      * forcing ``cols`` above the largest power of two <= H wastes the overhang
        on masked lanes.
    """
    best, cost = None, None
    for rows in _ROWS:
        for cols in tile_cols(H, quant_mode):
            if rows * cols <= _UB_TILE_ELEMS:
                c = est_kernel_us(M, H, rows, cols)
                if cost is None or c < cost:
                    best, cost = (rows, cols), c
    if best is None:
        raise ValueError(f"no tile fits for H={H} (block {_BLOCK[quant_mode]})")
    return best


def vendor_est_us(M, H):
    """Vendor kernel time, fitted to the same measurements as est_kernel_us.

    Same shape as this kernel's model -- which is the point: both sides are
    transcendental-bound, so they scale the same way and differ only in the
    constants::

        vendor = 2.50 us + 0.029 us * cores_busy + 0.270 ns per element per core
        this   = 1.20 us + 0.023 us * cores_busy + per-step, ~0.24 ns per element
    """
    cores = min(M, CORE_NUM)
    return 2.50 + 0.029 * cores + 0.270e-3 * M * H / CORE_NUM


def should_use_triton(M, H, quant_mode=1, graph_captured=False):
    """True where this kernel is at least as fast as the vendor.

    The one family of exceptions is ``H = 1280`` (D = 2560) at large T, where
    this returns False from T = 768 up.  That is the overlapping-chunk tax:
    1280 is not a power of two, so the widest legal chunk (1024) covers 2048
    columns in two steps and part of the work is redundant.  At small T the
    tile is not the bottleneck, which is why the cut is on T as well as H.

    Both cost models are compared with a 10% band, because at ``H = 1792``
    they land within 1% of each other while this kernel is in fact ahead.

    ``graph_captured`` picks what the vendor is charged for.  False (the
    default) adds the ~16 us of aclnn host dispatch it pays in eager mode; True
    compares kernel time only, which is what survives ACL graph capture -- so
    compare kernel-time ratios against ``graph_captured=True``.
    """
    rows, cols = pick_tile(M, H, quant_mode)
    est = est_kernel_us(M, H, rows, cols)
    vend = vendor_est_us(M, H)
    return est < 1.10 * (vend if graph_captured else vend + _DISPATCH_US)


def _validate(
    x, weight, group_index, dst_type, quant_mode, block_size, round_scale, clamp_limit
):
    if not getattr(x, "is_npu", False):
        raise ValueError("x must be an NPU tensor")
    if x.dtype not in _IN_DTYPES:
        raise ValueError(
            f"x dtype {x.dtype} not supported for quant_mode {quant_mode}; "
            "expected float16 or bfloat16"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}")
    if quant_mode not in _BLOCK:
        raise ValueError(
            f"quant_mode {quant_mode} not supported (2 and 3 are HiFloat8, "
            "which Triton cannot emit; use the vendor)"
        )
    if dst_type not in _FP8_MAX:
        raise ValueError(
            f"dst_type {dst_type} not supported; expected float8_e4m3fn or "
            "float8_e5m2"
        )
    if quant_mode == 1 and not round_scale:
        raise ValueError("quant_mode 1 (MX) requires round_scale=True")
    D = x.shape[-1]
    if D % 256:
        # Same constraint the vendor enforces (it fails at runtime otherwise).
        raise ValueError(f"last dim {D} must be a multiple of 256")
    if block_size not in (0, _BLOCK[quant_mode]):
        raise ValueError(
            f"block_size {block_size} invalid for quant_mode {quant_mode}; "
            f"expected 0 or {_BLOCK[quant_mode]}"
        )

    if (
        clamp_limit is not None
        and float(clamp_limit) != -1.0
        and float(clamp_limit) <= 0.0
    ):
        raise ValueError(f"clamp_limit must be -1.0 (off) or > 0, got {clamp_limit}")
    T = x.numel() // D
    if weight is not None:
        if weight.dtype != torch.float32:
            raise ValueError(f"weight must be float32, got {weight.dtype}")
        if weight.numel() != T:
            raise ValueError(f"weight has {weight.numel()} elements, expected {T}")
    if group_index is not None:
        if group_index.dtype != torch.int64:
            raise ValueError(f"group_index must be int64, got {group_index.dtype}")
        if group_index.dim() != 1:
            raise ValueError("group_index must be 1-D")


class SwigluGroupQuantPlan:
    """Prepare once, launch many: resolve the tile, compile, bind a raw launch.

    Holding the plan removes the Triton dispatch from the steady-state path,
    which matters because this operator is a few microseconds long.
    ``swiglu_group_quant`` keeps a bounded cache of plans so the one-shot entry
    point also pays this only once per specialization.
    """

    __slots__ = (
        "T",
        "D",
        "H",
        "nblk",
        "nsp",
        "rows",
        "cols",
        "grid",
        "dtype",
        "dst_type",
        "quant_mode",
        "round_scale",
        "clamp_limit",
        "output_origin",
        "has_w",
        "has_gi",
        "g",
        "masked",
        "launch",
        "_shape",
    )

    def __init__(
        self,
        x,
        weight=None,
        group_index=None,
        dst_type=torch.float8_e4m3fn,
        quant_mode=1,
        block_size=0,
        round_scale=None,
        clamp_limit=-1.0,
        output_origin=False,
        rows=None,
        cols=None,
    ):
        if round_scale is None:
            round_scale = quant_mode == 1
        _validate(
            x,
            weight,
            group_index,
            dst_type,
            quant_mode,
            block_size,
            round_scale,
            clamp_limit,
        )
        self._shape = tuple(x.shape)
        self.D = x.shape[-1]
        self.H = self.D // 2
        self.T = x.numel() // self.D
        self.dtype = x.dtype
        self.dst_type = dst_type
        self.quant_mode = quant_mode
        self.round_scale = bool(round_scale)
        self.clamp_limit = float(clamp_limit if clamp_limit is not None else -1.0)
        self.output_origin = bool(output_origin)
        self.has_w = weight is not None
        self.has_gi = group_index is not None
        self.g = group_index.numel() if self.has_gi else 0
        B = _BLOCK[quant_mode]
        self.nblk = -(-self.H // B)
        self.nsp = (self.nblk + 1) // 2 * 2

        clamp_on = self.clamp_limit != -1.0
        cands = []
        if rows and cols:
            cands.append((rows, cols))
        tile = pick_tile(self.T, self.H, quant_mode)
        legal = [
            (r, c)
            for r in _ROWS
            for c in tile_cols(self.H, quant_mode)
            if r * c <= _UB_TILE_ELEMS
        ]
        legal.sort(key=lambda rc: est_kernel_us(self.T, self.H, *rc))
        for rc in [tile] + legal:
            if rc not in cands:
                cands.append(rc)

        err = None
        for r, c in cands:
            # A dynamic row count (group_index) or a ragged last row group needs
            # the row mask; a ragged last column chunk needs the column overlap.
            rmask = 1 if (self.T % r or self.has_gi) else 0
            overlap = 1 if self.H % c else 0
            steps = -(-self.H // c)
            grid = min(triton.cdiv(self.T, r), CORE_NUM)
            gp = max(8, triton.next_power_of_2(max(self.g, 1)))
            # the 0.0 is ZERO: a runtime zero, so the sign-of-zero fixup in the
            # kernel survives the compiler.  Do not make it a constexpr.
            args = (
                self.T,
                0.0,
                self.D,
                self.H,
                self.nblk,
                self.nsp,
                self.g,
                gp,
                r,
                c,
                B,
                c // B,
                _INV_FP8_MAX[dst_type],
                _FP8_MAX[dst_type],
                1 if quant_mode == 1 else 0,
                1 if round_scale else 0,
                1 if self.has_w else 0,
                1 if self.has_gi else 0,
                1 if clamp_on else 0,
                self.clamp_limit,
                1 if dst_type == torch.float8_e5m2 else 0,
                1 if self.output_origin else 0,
                rmask,
                steps,
                overlap,
                grid,
            )
            y = torch.empty((self.T, self.H), dtype=dst_type, device=x.device)
            s = self._new_scale(x.device)
            yo = (
                torch.empty((self.T, self.H), dtype=x.dtype, device=x.device)
                if self.output_origin
                else y
            )

            try:
                # JITFunction.run returns the CompiledKernel for *this*
                # specialization.  Do not go fishing in device_caches for the
                # newest entry instead: on a cache hit nothing is added, so that
                # binds whatever was compiled last -- a mode-0 kernel to a
                # mode-1 plan, if one process builds both.
                ck = _sgq_kernel[(grid,)](
                    x.view(self.T, self.D), weight, group_index, y, s, yo, *args
                )
                torch.npu.synchronize()
            except Exception as e:  # compiler crash / UB overflow
                logger.debug(
                    "swiglu_group_quant: tile rows=%d cols=%d rejected (%s)",
                    r,
                    c,
                    e,
                )
                err = e
                continue
            self.rows, self.cols, self.grid = r, c, grid
            self.masked = (rmask, overlap)
            ck = ck.result() if hasattr(ck, "result") else ck
            if _raw_stream is None:  # pragma: no cover - torch_npu ships it
                # No direct stream binding: go through the JIT launcher.
                def launch(x, w, gi, y, s, yo, _grid=grid, _args=args):
                    _sgq_kernel[(_grid,)](x, w, gi, y, s, yo, *_args)

            else:
                # The stream has to be the one the tensors live on; the plan is
                # cached per device index, so binding x's is what keeps a plan
                # reusable on a multi-device process.
                dev = x.device.index
                if dev is None:  # pragma: no cover - allocated tensors carry one
                    dev = triton.runtime.driver.active.get_current_device()
                fn, meta = ck.function, ck.packed_metadata

                def launch(
                    x,
                    w,
                    gi,
                    y,
                    s,
                    yo,
                    _ck=ck,
                    _fn=fn,
                    _meta=meta,
                    _grid=grid,
                    _dev=dev,
                    _args=args,
                ):
                    _ck.run(
                        _grid,
                        1,
                        1,
                        _raw_stream(_dev),
                        _fn,
                        _meta,
                        None,
                        None,
                        None,
                        x,
                        w,
                        gi,
                        y,
                        s,
                        yo,
                        *_args,
                    )

            self.launch = launch
            return
        raise RuntimeError(
            f"no tile compiles for T={self.T} H={self.H} mode={quant_mode}: "
            f"{type(err).__name__}: {str(err)[:200] if err else 'none fit UB'}"
        )

    def _new_scale(self, device):
        if self.quant_mode == 1:
            # zeros so the pad byte of an odd block count stays 0
            return torch.zeros((self.T, self.nsp), dtype=torch.uint8, device=device)
        return torch.empty((self.T, self.nblk), dtype=torch.float32, device=device)

    def __call__(
        self, x, weight=None, group_index=None, out=None, scale=None, y_origin=None
    ):
        if tuple(x.shape) != self._shape or x.dtype != self.dtype:
            raise ValueError(
                f"plan is for {self._shape} {self.dtype}, got "
                f"{tuple(x.shape)} {x.dtype}"
            )
        if out is None:
            out = torch.empty((self.T, self.H), dtype=self.dst_type, device=x.device)
        if scale is None:
            scale = self._new_scale(x.device)
        if y_origin is None and self.output_origin:
            y_origin = torch.empty((self.T, self.H), dtype=self.dtype, device=x.device)
        sv = scale.view(self.T, self.nsp if self.quant_mode == 1 else self.nblk)
        self.launch(
            x.view(self.T, self.D),
            weight,
            group_index,
            out.view(self.T, self.H),
            sv,
            (y_origin if self.output_origin else out).view(self.T, self.H),
        )
        shp = self._shape[:-1]
        return (
            out.view(*shp, self.H),
            scale.view(scale_shape(self._shape, self.quant_mode)),
            y_origin.view(*shp, self.H) if self.output_origin else None,
        )

    def __repr__(self):
        return (
            f"SwigluGroupQuantPlan(T={self.T}, H={self.H}, rows={self.rows}, "
            f"cols={self.cols}, grid={self.grid}, "
            f"rowmask={bool(self.masked[0])}, "
            f"overlap={bool(self.masked[1])}, mode={self.quant_mode}, "
            f"round_scale={self.round_scale}, clamp={self.clamp_limit}, "
            f"{self.dtype} -> {self.dst_type})"
        )


def _plan_key(
    x,
    weight,
    group_index,
    dst_type,
    quant_mode,
    round_scale,
    clamp_limit,
    output_origin,
):
    """Everything the compiled specialization and the bound launch depend on."""
    return (
        tuple(x.shape),
        x.dtype,
        x.device.index,
        dst_type,
        quant_mode,
        bool(round_scale),
        float(clamp_limit if clamp_limit is not None else -1.0),
        weight is not None,
        group_index.numel() if group_index is not None else -1,
        bool(output_origin),
    )


def _get_plan(
    x,
    weight,
    group_index,
    dst_type,
    quant_mode,
    block_size,
    round_scale,
    clamp_limit,
    output_origin,
):
    key = _plan_key(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        round_scale,
        clamp_limit,
        output_origin,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = SwigluGroupQuantPlan(
            x,
            weight,
            group_index,
            dst_type,
            quant_mode,
            block_size,
            round_scale,
            clamp_limit,
            output_origin,
        )
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
            # FIFO: every entry pins a CompiledKernel, and the shapes a model
            # cycles through are few, so insertion order is a good enough proxy.
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        _PLAN_CACHE[key] = plan
        logger.debug("swiglu_group_quant: compiled %r", plan)
    return plan


def swiglu_group_quant(
    x,
    weight=None,
    group_index=None,
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    block_size=0,
    round_scale=None,
    clamp_limit=-1.0,
    output_origin=False,
    out=None,
    scale=None,
    y_origin=None,
):
    """SwiGLU plus group quantisation, as ``(y, scale, y_origin)``.

    Same result as ``aclnnSwigluGroupQuant`` for ``quant_mode`` 0 and 1.

    Args:
        x: ``[..., D]`` fp16/bf16, contiguous, ``D`` a multiple of 256.  The
            last dim splits into the gate half ``x[..., :D/2]`` and the linear
            half ``x[..., D/2:]``.
        weight: optional fp32 per-token multiplier, one element per row of x.
        group_index: optional int64 counts per expert.  Only the first
            ``sum(group_index)`` rows are written; the rest of ``out`` / ``scale``
            keeps whatever it held, which is what the vendor does -- pass your
            own buffers if that tail matters.
        dst_type: ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
        quant_mode: 0 = block 128 with an fp32 scale, 1 = MX block 32 with a
            uint8 E8M0 scale.  Note vllm-ascend's ``_C_ascend`` entry point
            numbers these one higher (its 2 is this 1).
        block_size: 0 (take the mode's block) or the mode's block size.
        round_scale: round the scale up to a power of two.  Defaults to True for
            mode 1, which requires it, and False for mode 0.
        clamp_limit: -1.0 (off) or a positive bound; the gate half is clamped
            above only, the linear half symmetrically.
        output_origin: also return the pre-weight SwiGLU in x's dtype.
        out, scale, y_origin: optional output buffers.

    Returns:
        ``(y, scale, y_origin)``; ``y_origin`` is None unless
        ``output_origin=True`` (the vendor returns an empty tensor there).

        The mode 1 scale is ``uint8`` holding E8M0 exponents, laid out as the
        vendor does (``[..., ceil(nblk/2), 2]``).  CANN's own python binding
        hands back the same bytes typed ``float8_e8m0fnu``; ``view`` across the
        two if a consumer needs that dtype.
    """
    logger.debug("GEMS_ASCEND SWIGLU_GROUP_QUANT")
    if round_scale is None:
        round_scale = quant_mode == 1
    plan = _get_plan(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        block_size,
        round_scale,
        clamp_limit,
        output_origin,
    )
    return plan(x, weight, group_index, out=out, scale=scale, y_origin=y_origin)
