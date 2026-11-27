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
#
# Ascend implementation imported from the tuned operator batch
# host entry points below mirror the generic
# flag_gems names so the backend registrar can pick them up.

"""xlogy_scalar_tensor : out = xlogy(self_scalar, other), self-contained Triton.

Reference semantics (``torch.xlogy(scalar, tensor)``, fp32 math):
  * ``y != y``    -> NaN   (aten precedence: checked before the zero rule)
  * ``self == 0`` -> +0.0  (even for y == 0, y < 0, y == +-inf)
  * otherwise     -> ``float(self) * log(y)``
  * output dtype/shape always follow ``other``; ``other`` is never mutated.

For ``self != 0`` the two special cases collapse into plain IEEE arithmetic
(``s * log(NaN)`` is already NaN, and the ``self == 0`` arm is unreachable), so
the timed path is one fused multiply-by-log.

Launch geometry follows the 910B4 rules:
  * the driver flattens the grid into ``coreDim``, so ``grid <= 65535`` is a hard
    limit -- a 2**30-element fp16 tensor used to ask for exactly 65536 programs
    (fp32: 131072), the driver rejected the launch with ``coreDim is invalid``,
    the profiler then reported no device tasks and every benchmark row read
    ``0.000000``.  Every path is now capped and walks a grid-stride loop.
  * L2-resident vectors (``n * element_size <= 168 MiB``) run one program per
    vector core (wide tile, few programs); vectors streamed from HBM run a few
    hundred programs with a grid-stride loop (a program per tile pays ~5 ns of
    setup per extra program: fp16 2**30 is 3604 us at 512 programs vs 3888 us
    at 65535).
  * small tensors get narrow tiles so at least ~16 vector cores take part.

Layout: contiguous inputs take a flat kernel, non-contiguous views take an
index-math kernel driven by their real element strides so ``base[..., 0]``
style views are read correctly, and the result is always dense.

The contiguous path stages GM -> UB -> GM with TLE ``dsa.copy`` when the
extension is available: flushing the tile through an explicit UB buffer lets
the output DMA overlap the next tile's math (26.5 us vs 31.9 us on 16.7M fp16,
7.03 ms vs 7.28 ms on 2**30 fp32).  Without TLE it falls back to plain
``tl.load``/``tl.store`` grid-stride kernels.
"""

import triton
import triton.language as tl

try:  # TLE (DSA copy) ships with the Ascend triton fork
    from triton.experimental.tle import language as _tle_language

    _DSA = _tle_language.dsa
except Exception:  # pragma: no cover - plain tl.load/tl.store fallback
    _DSA = None

_BLOCK_MIN = 256  # narrow tile for tiny tensors (fill several cores)
_BLOCK_CAP_2B = 16384  # 32 KiB source tile for 2-byte dtypes
_BLOCK_CAP_4B = 8192  # 32 KiB source tile for 4-byte dtypes
_MIN_PROGS = 16  # keep at least this many programs for small inputs
_LOCAL_PROGS = 40  # vector-core count: programs for L2-resident data
_L2_BYTES = 168 * 1024 * 1024
_STREAM_PROGS = 512  # programs for HBM-streamed data (measured plateau)
_MAX_PROGS = 65535  # driver flattens the grid into coreDim
_MAX_ELEMS = 1 << 30  # int32 offset safety when chunking huge tensors
_BLOCK_STRIDED = 512
_MAX_DIMS = 8

_PLANS = {}


@triton.jit
def _xlogy_flat(Y, OUT, s, BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(Y + offs).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty))


@triton.jit
def _xlogy_flat_masked(Y, OUT, n, s, BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    y = tl.load(Y + offs, mask=mask, other=1.0).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _xlogy_loop(
    Y,
    OUT,
    n,
    s,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    HAS_LOG: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Grid-stride variant used when the tile count exceeds the grid cap."""
    t = tl.program_id(0)
    step = tl.num_programs(0)
    while t < TILES:
        offs = t * BLOCK + tl.arange(0, BLOCK)
        if MASKED:
            mask = offs < n
            y = tl.load(Y + offs, mask=mask, other=1.0).to(tl.float32)
            if HAS_LOG:
                val = s * tl.log(y)
            else:
                val = tl.where(y != y, float("nan"), 0.0)
            tl.store(OUT + offs, val.to(OUT.dtype.element_ty), mask=mask)
        else:
            y = tl.load(Y + offs).to(tl.float32)
            if HAS_LOG:
                val = s * tl.log(y)
            else:
                val = tl.where(y != y, float("nan"), 0.0)
            tl.store(OUT + offs, val.to(OUT.dtype.element_ty))
        t += step


@triton.jit
def _xlogy_strided(
    Y,
    OUT,
    n,
    s,
    SHAPE: tl.constexpr,
    YS: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    HAS_LOG: tl.constexpr,
):
    t = tl.program_id(0)
    step = tl.num_programs(0)
    while t < TILES:
        idx = t * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        rem = idx.to(tl.int64)
        off = tl.zeros((BLOCK,), dtype=tl.int64)
        for ax in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = rem % SHAPE[ax]
            rem = rem // SHAPE[ax]
            off += coord * YS[ax]
        y = tl.load(Y + off, mask=mask, other=1.0).to(tl.float32)
        if HAS_LOG:
            val = s * tl.log(y)
        else:
            val = tl.where(y != y, float("nan"), 0.0)
        tl.store(OUT + idx, val.to(OUT.dtype.element_ty), mask=mask)
        t += step


if _DSA is not None:

    @triton.jit
    def _xlogy_tle(
        Y,
        OUT,
        n,
        s,
        BLOCK: tl.constexpr,
        TILES: tl.constexpr,
        HAS_LOG: tl.constexpr,
        MASKED: tl.constexpr,
    ):
        """GM -> UB -> GM with explicit DSA copies (grid-stride).

        Staging the result through UB and flushing it with an explicit DMA lets
        the MTE3 transfer overlap the next iteration's vector math: measured
        26.5us vs 31.9us (fp16 16.7M) and 7.03ms vs 7.28ms (fp32 2**30) against
        the tl.store kernel, while matching it bit for bit.  The tail is handled
        by a runtime copy extent, so no masked access is ever emitted.
        """
        t = tl.program_id(0)
        step = tl.num_programs(0)
        lanes = tl.arange(0, BLOCK)
        y_ub = _DSA.alloc(
            [BLOCK], dtype=Y.dtype.element_ty, mem_addr_space=_DSA.ascend.UB
        )
        o_ub = _DSA.alloc(
            [BLOCK], dtype=OUT.dtype.element_ty, mem_addr_space=_DSA.ascend.UB
        )
        while t < TILES:
            base = t * BLOCK
            offs = base + lanes
            if MASKED:
                cnt = tl.minimum(n - base, BLOCK)
                _DSA.copy(Y + offs, y_ub, [cnt])
            else:
                _DSA.copy(Y + offs, y_ub, [BLOCK])
            y = _DSA.to_tensor(y_ub).to(tl.float32)
            if HAS_LOG:
                val = s * tl.log(y)
            else:
                val = tl.where(y != y, float("nan"), 0.0)
            _DSA.to_buffer(val.to(OUT.dtype.element_ty), bind_buffer=o_ub)
            if MASKED:
                _DSA.copy(o_ub, OUT + offs, [cnt])
            else:
                _DSA.copy(o_ub, OUT + offs, [BLOCK])
            t += step


def _pick_launch(n, esize):
    """Return (block, grid) for a contiguous 1-D launch of ``n`` elements."""
    cap = _BLOCK_CAP_2B if esize < 4 else _BLOCK_CAP_4B
    block = _BLOCK_MIN
    want = (n + _MIN_PROGS - 1) // _MIN_PROGS
    while block < want and block < cap:
        block <<= 1
    tiles = (n + block - 1) // block
    if n * esize <= _L2_BYTES:
        grid = min(tiles, _LOCAL_PROGS)
    else:
        # Measured on 910B4 (2**30: fp16 3604us @512 vs 3888us @65535; fp32
        # 7332us @512 vs 7340us @65535): a few hundred programs already hide
        # the per-program setup cost, and asking for one program per tile pays
        # ~5 ns per extra program.  Keep the hard cap for safety.
        grid = min(tiles, _STREAM_PROGS)
    return block, min(grid, _MAX_PROGS), tiles


def _make_entry(n, s, esize):
    """Bind one compiled specialisation to (grid, block, tile count)."""
    block, grid, tiles = _pick_launch(n, esize)
    has_log = 1 if s != 0.0 else 0
    masked = bool(n % block)
    if _DSA is not None:
        fn = _xlogy_tle

        def entry(
            y,
            out,
            scalar,
            _f=fn,
            _g=(grid,),
            _n=n,
            _b=block,
            _h=has_log,
            _t=tiles,
            _m=1 if masked else 0,
        ):
            _f[_g](y, out, _n, scalar, BLOCK=_b, TILES=_t, HAS_LOG=_h, MASKED=_m)

        return entry
    if grid == tiles:
        if masked:
            fn = _xlogy_flat_masked

            def entry(y, out, scalar, _f=fn, _g=(grid,), _n=n, _b=block, _h=has_log):
                _f[_g](y, out, _n, scalar, BLOCK=_b, HAS_LOG=_h)

        else:
            fn = _xlogy_flat

            def entry(y, out, scalar, _f=fn, _g=(grid,), _b=block, _h=has_log):
                _f[_g](y, out, scalar, BLOCK=_b, HAS_LOG=_h)

    else:
        fn = _xlogy_loop

        def entry(
            y,
            out,
            scalar,
            _f=fn,
            _g=(grid,),
            _n=n,
            _b=block,
            _h=has_log,
            _t=tiles,
            _m=1 if masked else 0,
        ):
            _f[_g](y, out, _n, scalar, BLOCK=_b, TILES=_t, HAS_LOG=_h, MASKED=_m)

    return entry


def _launch_flat(y, out, n, s):
    if n > _MAX_ELEMS:
        # keep every offset inside int32 (never happens on the test shapes)
        yf, of = y.view(-1), out.view(-1)
        for i in range(0, n, _MAX_ELEMS):
            m = min(_MAX_ELEMS, n - i)
            _launch_flat(yf[i : i + m], of[i : i + m], m, s)
        return
    key = (y.dtype, n, s != 0.0, y.get_device())
    entry = _PLANS.get(key)
    if entry is None:
        entry = _make_entry(n, s, y.element_size())
        _PLANS[key] = entry
    entry(y, out, s)


def _launch_strided(y, out, n, s):
    shape = y.shape
    has_log = 1 if s != 0.0 else 0
    if len(shape) > _MAX_DIMS:
        for i in range(shape[0]):
            sub = y.select(0, i)
            m = sub.numel()
            if m:
                _launch_strided(sub, out.select(0, i), m, s)
        return
    tiles = (n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED
    grid = (min(tiles, _STREAM_PROGS),)
    _xlogy_strided[grid](
        y,
        out,
        n,
        s,
        SHAPE=tuple(shape),
        YS=tuple(y.stride()),
        BLOCK=_BLOCK_STRIDED,
        TILES=tiles,
        HAS_LOG=has_log,
    )


def run(self, other):
    y = other
    out = y.new_empty(y.shape)
    n = y.numel()
    if n == 0:
        return out
    s = float(self)
    if y.is_contiguous():
        _launch_flat(y, out, n, s)
    else:
        _launch_strided(y, out, n, s)
    return out


# --------------------------------------------------------------------------- #
# FlagGems entry points
# --------------------------------------------------------------------------- #
def xlogy_scalar_tensor(*args, **kwargs):
    return run(*args, **kwargs)
