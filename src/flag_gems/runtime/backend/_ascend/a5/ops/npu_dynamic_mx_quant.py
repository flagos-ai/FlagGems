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

"""Bit-exact MX dynamic quantization (DynamicMxQuant) in Triton for Ascend 950DT.

Overrides ``torch_npu.npu_dynamic_mx_quant`` (op-plugin
``DynamicMxQuantKernelNpuOpApi.cpp`` -> ``aclnnDynamicMxQuantV3``) for the
activation-quantization paths in vllm-ascend (``w8a8_mxfp8.py``,
``fp8_block.py::_mx_quantize``).

Output is **byte-identical** to the vendor for every finite input, for both the
fp8 payload and the uint8 E8M0 scale bytes, across every (input dtype, output
dtype, scale_alg, block_size) combination this module accepts -- including
saturating blocks, all-zero blocks, -0.0, denormals and ``finfo.max``.

The one exception is a block containing inf or nan: the vendor emits +0 for
every element of such a block, while this kernel emits -0 for the negative
ones (numerically equal, and both dequantize to zero).  Clearing that sign bit
needs a third per-block broadcast operand, which costs 4.2x on this backend
(4.2 -> 17.7 us at M=16 K=7168), so it is opt-in via ``exact_inf_nan=True``.
The scale byte (255) is correct either way.

Semantics, as measured against the vendor on crafted blocks (not from the doc):
  * scale byte = (exponent of the block amax) - emax + 127, where emax is 8 for
    float8_e4m3fn and 15 for float8_e5m2.  The exponent is taken from an fp32
    view of the amax, so that a denormal amax (which has no implicit leading 1,
    and so an exponent field that is not its exponent) still comes out right.
  * ``scale_alg=1`` (CEIL; vllm-ascend uses it for MiniMax) adds 1 iff the amax
    mantissa exceeds 0.75, i.e. amax > 1.75 * 2^e.  Since fmax = 1.75 * 2^emax
    for both e4m3 and e5m2, that is exactly "the smallest scale for which
    nothing overflows", so the CEIL path never saturates.  Verified: alg1 - alg0
    is 1 on 25.9% of random blocks, matching P(mantissa > 0x60).
  * ``scale_alg=0`` (FLOOR) must saturate: the top element(s) clamp to +-fmax.
  * the two algs differ on a block whose ideal scale byte would be <= 0, and
    both behaviours are implemented: alg 0 emits scale byte 0 and y = +-0 with
    the sign preserved (so all-zero and denormal-amax blocks vanish), while
    alg 1 pins the byte at 0 and still scales by 2^127, keeping the data.
  * a block containing inf or nan takes scale byte 255 and y = 0.
  * the scale tensor is uint8 ``[..., ceil(nblk / 2), 2]``, i.e. a contiguous
    ``[..., nblk]`` padded to an even byte count; the pad byte is 0.

Design notes, all measured on this box (Ascend950DT_9582, 64 vector cores,
FlagTree triton 0.7.0+ascend.gitde535dd8, CANN 9.2.1):
  * the per-block amax is taken on the **integer magnitude bits**, not with
    fp32 abs: clearing the sign bit makes the int pattern monotonic in |x|, and
    this avoids widening the reduce input to fp32.
  * the reciprocal and the saturation cap are built from bit patterns rather
    than computed: rc = 2^(127-eb) is ``(254-eb) << 23`` bitcast to fp32, and
    cap = 1.75 * 2^e_amax is ``(amax_bits & ~mantissa) | fmax_mantissa``.
    Deriving the cap from the amax bits instead of from exponent arithmetic
    (``((eb + emax - shift) << mbits) | fmax_mantissa``) matters twice over: the
    old form produced a negative exponent field for fp16 whenever eb was clamped
    to 0, and it was also what crashed the bisheng backend ("Do not know how to
    expand the result of this operator!") for tiles of 5120..8192 elements.
  * saturation runs in the integer magnitude domain (``min(mag, cap) | sign``),
    which is much cheaper here than clamping the fp32 product: an fp32
    ``min``/``max`` pair costs ~65 us at M=4096 K=7168 against ~26 us for the
    int16 pair, because the fp32 temporaries spill UB.
  * the inf/nan and tiny-block fixups ride on ``cap``, which saturation already
    broadcasts, so they are free.  The backend tolerates two per-block
    broadcast operands (the int16 ``cap`` and the fp32 ``rc``) and falls off a
    cliff at three; that is the whole reason the inf/nan sign fix is optional.
  * tiles must be **whole rows**, but the reason is tile efficiency rather than
    stride.  A multi-row partial-K tile is genuinely strided and costs 14x
    (1300 us against 95 at M=4096 K=7168), and reassembling chunks with
    insert_slice costs more than the spill it avoids (150 us) -- but a 1-row x
    512 tile is perfectly contiguous and still loses, because small tiles run at
    2.8 ns/element against 0.27 at 14336 elements.  That ~10x swamps the
    occupancy a column split would buy, which is why the vendor's own 2-D
    rows x 512 split measures 595 us here at M=4096.
  * the one free parameter is therefore the tile size in elements (ROWS * K),
    and its cost curve is **not monotonic** for 16-bit input: 512 -> 1.44 us,
    2048 -> 3.50, 4096 -> 6.00, 8192 -> 11.00, then it collapses to 3.94 at
    14336 and 4.30 at 16384 before rising again to 7.80 at 32768.  ``pick_rows``
    steers for that cliff (tiles per core x the measured per-tile cost) and
    lands on the best measured ROWS in 31 of 37 calibrated cells, worst case
    1.17x.  fp32 has no cliff and half the element budget.

Performance against the vendor, bf16 -> e4m3, block 32, verified bit-exact at
every point.  "kernel" is the profiler's kernel time, "wall" is eager
wall-clock, which includes ~15-18 us of aclnn host dispatch that ACL graph
capture removes in production; both are reported because they disagree.

    K=7168     vendor kernel   this kernel   kernel    wall
    M=1              3.12 us       3.95 us     0.79x     2.88x
    M=4              3.25          4.05        0.80x     2.90x
    M=16             3.65          4.12        0.89x     3.60x
    M=64             3.85          4.37        0.88x     2.97x
    M=256            5.17          8.00        0.65x     2.59x
    M=1024           8.08         25.69        0.31x     0.86x
    M=4096          19.38         96.62        0.20x     0.22x

    K=512      vendor kernel   this kernel   kernel    wall
    M=16             3.28 us       2.38 us     1.38x     2.91x
    M=256            3.77          4.38        0.86x     3.17x
    M=4096           5.43          8.80        0.62x     2.32x

This kernel wins on eager wall time up to M~256 (vendor pays ~16-18 us aclnn
dispatch), and on kernel time only where the vendor is latency-bound and one
tile is cheap (K<=512, M<=64, 1.38x peak).  It loses in the prefill regime.
The gap at large M is vector-issue cost: memory traffic is identical on both
sides; the vendor keeps the whole chain in registers (~6.5 passes/element),
while Triton materialises every intermediate to UB (~10 passes).

Not supported, with reasons:
  * ``float4_e2m1fn_x2`` (GLM W4A4C8): stride-2 packing is ~50x slower or
    hangs ``bishengir-compile`` (>30 min on a 7168-wide tile).  Use the vendor.
  * ``scale_alg=2``: rejected by this CANN build, nothing to match.
  * ``axis`` other than the last: kernel assumes the quantized axis is contiguous.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["npu_dynamic_mx_quant"]

# input dtype -> (mantissa bits, integer width)
_IN_DTYPE = {
    torch.bfloat16: (7, 16),
    torch.float16: (10, 16),
    torch.float32: (23, 32),
}
# output dtype -> emax (fmax = 1.75 * 2^emax for both)
_OUT_DTYPE = {
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 15,
}

# The bisheng crash for tiles of 5120..8192 elements is gone: it came from
# building the saturation cap out of exponent arithmetic.  The cap is now
# derived from the amax bits ((mm & ~MANT) | FM), and all hidden sizes x ROWS
# in (1,2,4) x both algs compile and are byte-exact.  DynamicMxQuantPlan still
# falls back to another ROWS if a compile fails.
_UB_TILE_ELEMS = 32768  # largest whole-row tile that fits UB (16-bit in)
_CORES = 64

# Measured single-tile kernel time (us) vs tile elements (= ROWS * K).
# The 16-bit curve is NOT monotonic: cost climbs to 11.0 us at 8192 elements
# then collapses to ~4 us at 14336-16384 (backend switches to a better lowering).
# fp32 has no cliff and 32768 elements overflows UB.
_TILE_US = {
    2: (
        (512, 1.44),
        (2048, 3.50),
        (4096, 6.00),
        (7168, 9.80),
        (8192, 11.00),
        (14336, 3.94),
        (16384, 4.30),
        (28672, 6.90),
        (32768, 7.80),
    ),
    4: ((2048, 3.25), (4096, 5.62), (8192, 10.42), (16384, 20.09)),
}
_DISPATCH_US = 16.0  # aclnn host dispatch the vendor pays in eager
_GRAPH_FLOOR_US = 3.0  # vendor kernel floor once dispatch is captured


@triton.jit
def _dynamic_mx_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    K: tl.constexpr,
    NSP: tl.constexpr,
    ROWS: tl.constexpr,
    TB: tl.constexpr,
    B: tl.constexpr,
    EMAX: tl.constexpr,
    MBITS: tl.constexpr,
    CEIL: tl.constexpr,
    IBITS: tl.constexpr,
    E5: tl.constexpr,
    MASKED: tl.constexpr,
    NANSGN: tl.constexpr,
    PROGS: tl.constexpr,
):
    """One program handles ROWS whole rows per step, striding by PROGS * ROWS."""
    pid = tl.program_id(0)
    W: tl.constexpr = TB * B
    N: tl.constexpr = ROWS * TB
    SGN: tl.constexpr = -(2 ** (IBITS - 1))
    MAG: tl.constexpr = 2 ** (IBITS - 1) - 1
    MANT: tl.constexpr = 2**MBITS - 1
    FM: tl.constexpr = MANT - (MANT >> 2)  # mantissa of fmax (1.75)
    w = tl.arange(0, W)
    bl = tl.arange(0, TB)
    for r0 in range(pid * ROWS, M, PROGS * ROWS):
        r = r0 + tl.arange(0, ROWS)
        xb = x_ptr + r[:, None] * K
        yb = y_ptr + r[:, None] * K
        sb = s_ptr + r[:, None] * NSP
        rm = (r < M)[:, None]
        for k0 in range(0, K, W):
            if MASKED:
                flat = tl.load(xb + (k0 + w)[None, :], mask=rm, other=0)
            else:
                flat = tl.load(xb + (k0 + w)[None, :])
            xt = tl.reshape(flat, (N, B))

            if IBITS == 16:
                xi = xt.to(tl.int16, bitcast=True)
                mg = (xi & MAG).to(tl.int16)
                mm = tl.max(mg, 1)
                af = mm.to(tl.int16).to(xt.dtype, bitcast=True).to(tl.float32)
            else:
                xi = xt.to(tl.int32, bitcast=True)
                mg = xi & MAG
                mm = tl.max(mg, 1)
                af = mm.to(tl.float32, bitcast=True)
            ab = af.to(tl.int32, bitcast=True)
            ef = (ab >> 23) & 0xFF
            eb = ef - EMAX
            if CEIL:
                eb = eb + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
            nan = ef == 0xFF
            if CEIL:
                flush = nan
            else:
                flush = (eb <= 0) | nan
            ebs = tl.where(nan, 255, tl.maximum(eb, 0))
            rc = ((254 - tl.minimum(ebs, 254)) << 23).to(tl.float32, bitcast=True)
            if CEIL:
                cap = tl.where(nan, 0, MAG)
            else:
                cap = tl.where(flush, 0, (mm.to(tl.int32) & ~MANT) | FM)
            if IBITS == 16:
                sgn = (
                    (xi & tl.where(nan, 0, SGN)[:, None].to(tl.int16))
                    if NANSGN
                    else (xi & SGN)
                ).to(tl.int16)
                src = (tl.minimum(mg, cap[:, None].to(tl.int16)) | sgn).to(tl.int16)
            else:
                sgn = (xi & tl.where(nan, 0, SGN)[:, None]) if NANSGN else (xi & SGN)
                src = tl.minimum(mg, cap[:, None]) | sgn
            q = src.to(xt.dtype, bitcast=True).to(tl.float32) * rc[:, None]
            if E5:
                q8 = q.to(tl.float8e5)
            else:
                q8 = q.to(tl.float8e4nv)
            if MASKED:
                tl.store(yb + (k0 + w)[None, :], tl.reshape(q8, (ROWS, W)), mask=rm)
                tl.store(
                    sb + (k0 // B + bl)[None, :],
                    tl.reshape(ebs.to(tl.uint8), (ROWS, TB)),
                    mask=rm,
                )
            else:
                tl.store(yb + (k0 + w)[None, :], tl.reshape(q8, (ROWS, W)))
                tl.store(
                    sb + (k0 // B + bl)[None, :],
                    tl.reshape(ebs.to(tl.uint8), (ROWS, TB)),
                )


def scale_shape(shape, block_size, axis=-1):
    """Vendor scale layout: the quantized axis becomes ceil(nblk/2), trailing 2.

    Flattening the last two dims gives a contiguous [..., even(nblk)] byte
    tensor whose pad byte is 0, which is what ``_mx_quantize`` in vllm-ascend
    does with ``scale.flatten(-2)``.
    """
    if axis not in (-1, len(shape) - 1):
        raise ValueError(f"axis must be the last axis, got {axis}")
    nblk = shape[-1] // block_size
    return tuple(shape[:-1]) + ((nblk + 1) // 2, 2)


def tile_us(elems, itemsize):
    """Interpolate the measured single-tile cost curve (see _TILE_US)."""
    pts = _TILE_US[4 if itemsize > 2 else 2]
    if elems <= pts[0][0]:
        return pts[0][1] * elems / pts[0][0]
    for (t0, c0), (t1, c1) in zip(pts, pts[1:]):
        if elems <= t1:
            return c0 + (c1 - c0) * (elems - t0) / (t1 - t0)
    t, c = pts[-1]
    return c * elems / t


def est_kernel_us(M, K, itemsize=2, rows=None):
    """Predicted kernel time: tiles per core x the measured per-tile cost."""
    r = pick_rows(M, K, itemsize) if rows is None else rows
    groups = -(-M // r)
    iters = -(-groups // _CORES)
    return iters * tile_us(r * K, itemsize)


def pick_rows(M, K, itemsize):
    """Whole rows per tile: minimize tiles-per-core x measured per-tile cost."""
    budget = _UB_TILE_ELEMS // max(1, itemsize // 2)
    best, best_cost, r = 1, None, 1
    while r <= 32 and r * K <= budget:
        cost = est_kernel_us(M, K, itemsize, rows=r)
        if best_cost is None or cost < best_cost:
            best, best_cost = r, cost
        r *= 2
    return best


def should_use_triton(M, K, dtype=torch.bfloat16, block_size=32, graph_captured=False):
    """True where this kernel is at least as fast as the vendor."""
    if dtype not in _IN_DTYPE or block_size != 32:
        return False
    itemsize = 4 if dtype == torch.float32 else 2
    est = est_kernel_us(M, K, itemsize)
    return est < (_GRAPH_FLOOR_US if graph_captured else _DISPATCH_US)


def _validate(x, dst_type, block_size, axis, round_mode, scale_alg):
    if not x.is_npu:
        raise ValueError("x must be an NPU tensor")
    if x.dtype not in _IN_DTYPE:
        raise ValueError(
            f"x dtype {x.dtype} not supported; "
            f"expected one of {sorted(map(str, _IN_DTYPE))}"
        )
    if dst_type not in _OUT_DTYPE:
        raise ValueError(
            f"dst_type {dst_type} not supported (float4_e2m1fn_x2 needs a "
            "stride-2 pack that this compiler cannot lower; use the vendor)"
        )
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}")
    if axis not in (-1, x.dim() - 1):
        raise ValueError(f"axis must be the last axis, got {axis}")
    if round_mode != "rint":
        raise ValueError(f"round_mode {round_mode!r} not supported, only 'rint'")
    if scale_alg not in (0, 1):
        raise ValueError(
            f"scale_alg {scale_alg} not supported " "(2 is rejected by this CANN build)"
        )
    if block_size < 8 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of 2 >= 8, got {block_size}")
    if x.shape[-1] % block_size:
        raise ValueError(
            f"last dim {x.shape[-1]} must be a multiple of " f"block_size {block_size}"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")


class DynamicMxQuantPlan:
    """Prepare once, launch many: resolves the tile, compiles, and binds the
    raw launch so the steady-state path has no Python dispatch overhead."""

    __slots__ = (
        "M",
        "K",
        "nblk",
        "nsp",
        "rows",
        "grid",
        "dtype",
        "dst_type",
        "block_size",
        "scale_alg",
        "masked",
        "launch",
        "version",
        "exact_inf_nan",
        "_shape",
    )

    def __init__(
        self,
        x,
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        axis=-1,
        round_mode="rint",
        scale_alg=0,
        rows=None,
        exact_inf_nan=False,
    ):
        _validate(x, dst_type, block_size, axis, round_mode, scale_alg)
        self._shape = tuple(x.shape)
        self.K = x.shape[-1]
        self.M = x.numel() // self.K
        self.dtype = x.dtype
        self.dst_type = dst_type
        self.block_size = block_size
        self.scale_alg = scale_alg
        self.exact_inf_nan = bool(exact_inf_nan)
        self.version = "v0"
        self.nblk = self.K // block_size
        self.nsp = (self.nblk + 1) // 2 * 2

        mbits, ibits = _IN_DTYPE[x.dtype]
        emax = _OUT_DTYPE[dst_type]
        e5 = 1 if dst_type == torch.float8_e5m2 else 0
        ceil = 1 if scale_alg == 1 else 0
        nansgn = 1 if exact_inf_nan else 0

        cands = [rows] if rows else []
        budget = _UB_TILE_ELEMS // max(1, x.element_size() // 2)
        legal = [
            r for r in (1, 2, 4, 8, 16, 32) if r not in cands and r * self.K <= budget
        ]
        cands += sorted(
            legal, key=lambda r: est_kernel_us(self.M, self.K, x.element_size(), rows=r)
        )
        err = None
        for cand in cands:
            if cand * self.K > _UB_TILE_ELEMS // max(1, x.element_size() // 2):
                continue
            masked = 1 if self.M % cand else 0
            grid = min(triton.cdiv(self.M, cand), _CORES)
            args = (
                self.M,
                self.K,
                self.nsp,
                cand,
                self.nblk,
                block_size,
                emax,
                mbits,
                ceil,
                ibits,
                e5,
                masked,
                nansgn,
                grid,
            )
            y = torch.empty((self.M, self.K), dtype=dst_type, device=x.device)
            s = torch.zeros((self.M, self.nsp), dtype=torch.uint8, device=x.device)
            try:
                _dynamic_mx_quant_kernel[(grid,)](x.view(self.M, self.K), y, s, *args)
                torch.npu.synchronize()
            except Exception as e:
                err = e
                continue
            self.rows, self.masked, self.grid = cand, masked, grid

            # Dispatch through the normal Triton call path rather than
            # caching a raw CompiledKernel.  ``device_caches[dev][0]`` is
            # keyed by the full constexpr specialization, but "grab the
            # last-inserted entry" is only correct the moment a *new* kernel
            # variant is compiled: once a second plan for a different
            # constexpr combination (e.g. a different dst_type or scale_alg)
            # on the same shape warms the cache, the "last" entry points at
            # *that* variant's binary, and a plan built before it would
            # silently launch the wrong compiled kernel on every call. The
            # standard ``kernel[grid](...)`` call re-keys on the constexpr
            # values every time, so it always finds the binary that matches
            # this plan's own (dst_type, scale_alg, ...) regardless of what
            # has been compiled since.
            def launch(
                x, y, s, _kernel=_dynamic_mx_quant_kernel, _grid=(grid,), _args=args
            ):
                _kernel[_grid](x, y, s, *_args)

            self.launch = launch
            return
        raise RuntimeError(
            f"no tile compiles for M={self.M} K={self.K} dtype={x.dtype} "
            f"scale_alg={scale_alg}: {type(err).__name__}: "
            f"{str(err)[:200] if err else 'no candidate fit UB'}"
        )

    def __call__(self, x, out=None, scale=None):
        if tuple(x.shape) != self._shape or x.dtype != self.dtype:
            raise ValueError(
                f"plan is for {self._shape} {self.dtype}, got "
                f"{tuple(x.shape)} {x.dtype}"
            )
        if out is None:
            out = torch.empty(self._shape, dtype=self.dst_type, device=x.device)
        if scale is None:
            scale = torch.zeros(
                scale_shape(self._shape, self.block_size),
                dtype=torch.uint8,
                device=x.device,
            )
        self.launch(
            x.view(self.M, self.K),
            out.view(self.M, self.K),
            scale.view(self.M, self.nsp),
        )
        return out, scale

    def __repr__(self):
        return (
            f"DynamicMxQuantPlan(M={self.M}, K={self.K}, rows={self.rows}, "
            f"grid={self.grid}, masked={bool(self.masked)}, "
            f"block={self.block_size}, alg={self.scale_alg}, "
            f"exact_inf_nan={self.exact_inf_nan}, "
            f"{self.dtype} -> {self.dst_type}, {self.version})"
        )


def npu_dynamic_mx_quant(
    x,
    axis=-1,
    round_mode="rint",
    dst_type=torch.float8_e4m3fn,
    block_size=32,
    scale_alg=0,
):
    """MX dynamic quantization; byte-identical to ``torch_npu.npu_dynamic_mx_quant``.

    Returns ``(y, scale)`` matching the vendor layout exactly.  For repeated
    launches on a fixed shape, construct a ``DynamicMxQuantPlan`` instead to
    avoid the per-call compilation check.
    """
    logger.debug("GEMS_ASCEND NPU_DYNAMIC_MX_QUANT")
    return DynamicMxQuantPlan(x, dst_type, block_size, axis, round_mode, scale_alg)(x)
