"""TLE (Ascend, vec-explicit) implementation of the FlagGems ``isreal`` op.

``aten::isreal(Tensor self) -> Tensor`` returns a freshly allocated ``bool``
tensor that is ``True`` wherever the value has a zero imaginary component.
That definition splits into two disjoint data paths:

* real / integral / ``bool`` operands have no imaginary part at all, so the
  result is ``True`` everywhere and the input is never read.  The whole op is
  a 1-byte-per-element fill of the output storage.
* ``complex64`` / ``complex128`` operands are interleaved ``(re, im)`` pairs,
  so the result is ``im == 0.0``.  IEEE comparison gives exactly PyTorch's
  rule for free: ``-0.0 == 0.0`` is true and ``NaN == 0.0`` is false, and the
  real part (including a NaN real part) is irrelevant and never inspected.

``complex128`` needs one extra twist.  Its ``(re, im)`` pairs are ``float64``,
and the Ascend vector unit has no 64-bit float: a ``0.0`` broadcast against an
``f64`` tile is rejected outright by the compiler's ``hivm.hir.vbrc`` verifier
(allowed element types are 8/16/32-bit integer, 16/32-bit float, bfloat16,
64-bit integer and ``i1`` -- no ``f64``).  So the 128-bit path views the pair
buffer as ``int32`` words and tests bits, never forming a ``double``: a
``float64`` is ``+-0.0`` exactly when every bit but the sign is zero, i.e.
``((hi & 0x7FFFFFFF) | lo) == 0`` over its two 32-bit halves.  That is bit-exact
for ``+0.0``, ``-0.0``, ``NaN`` and ``inf``.  ``complex64`` keeps the plain
``f32`` comparison, which is native on this vector unit.

Kernel shape: ``vec-explicit``.  There is no ``tl.dot`` and no reuse, so the
whole story is explicit UB staging::

    GM --(MTE2, tle.dsa.copy)--> UB --> imag test --> UB --(MTE3, copy)--> GM

The output map is a plain bool tensor, i.e. one byte per element, so the
kernel takes an ``int8`` view of the freshly allocated storage and writes
``1`` / ``0`` bytes; ``torch.bool`` treats any non-zero byte as ``True``.

Grid: persistent.  Each program walks a strided sequence of tiles
(``pid``, ``pid + nprog``, ...), so the launch is capped by the runtime-queried
``num_vectorcore`` instead of by the tensor size.  The 2**30-element cases are
32768 to 262144 tiles, which is far beyond the 65535-program 1-D grid limit, so
a per-tile launch is not even legal there -- and the reference's per-tile launch
(BLOCK 1024/4096 over 2**30 elements, i.e. 1M tiny blocks) is exactly what makes
its complex path slow.  The core count is queried from the live device and never
hard-coded: the submission target is Ascend910B2 while the evaluation service
may report a different 910B SKU.

Tiles are sized by payload bytes (32 KB) so one ``tle.dsa.copy`` amortises the
per-copy fixed cost while the ``num_stages=2`` pipeline keeps the next tile's
transfer in flight.

The ragged tail (``n % BLOCK``) uses the masked ``tl.load`` / ``tl.store`` path
because ``tle.dsa.copy`` accepts an explicit shape but no mask.
"""

import logging

import torch
import torch_npu  # noqa: F401  -- must precede `import triton` in this env
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

# Dtypes whose values have no imaginary part -> result is all-True.
_REAL_DTYPES = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    )
)
# Dtypes with an interleaved (re, im) layout -> result is `im == 0`.
_COMPLEX_DTYPES = frozenset((torch.complex64, torch.complex128))

# Bytes of bool output produced per staged tile on the fill path.
_FILL_BLOCK = 32768
# Bytes of *input* read per staged tile on the complex path.
_C_TILE_BYTES = 32768
# Masked boundary-path tile widths.
_TAIL_FILL_BLOCK = 4096
_TAIL_C_BLOCK = 512
# Launched programs per vector core.  Exactly one: programs queue serially on a
# core, so extra programs per core cannot overlap anything, and they *hurt*
# whenever the tile count does not divide evenly -- a core that draws the
# ceil() programs idles while its neighbours drain the floor() ones.  The
# mid-size fills make that exact: 4096x4096 is 512 tiles, and 512/160 = 3.2
# tiles per program against a 4-tile maximum, i.e. 0.80 theoretical occupancy,
# which is the 0.8036 `aiv_mte3_ratio` the profiler measured for that kernel.
# One program per core bounds the imbalance at a single tile (512/40 = 12.8
# against 13 -> 0.985) and shortens the per-program pipeline prologue share.
_PROGRAMS_PER_CORE = 1
# Ascend maximum 1-D grid extent (only used if the core count cannot be read).
_MAX_GRID_1D = 65535

_NVC = [0]


def _num_vectorcore():
    """Vector-core count of the live target device (queried once, never hard-coded)."""
    nvc = _NVC[0]
    if nvc == 0:
        try:
            dev = torch.npu.current_device()
            props = triton.runtime.driver.active.utils.get_device_properties(dev)
            nvc = int(props.get("num_vectorcore", 0) or 0)
        except Exception:
            nvc = 0
        if nvc <= 0:
            nvc = 1
        _NVC[0] = nvc
    return nvc


def _grid(n_tiles):
    """Program count for ``n_tiles`` whole tiles: one per core, split evenly.

    A program owns a strided slice of the tile space, so a program count that
    does not divide the tile count hands the busiest program ``ceil`` tiles
    while the mean is lower -- and because programs queue *serially* on a core,
    the kernel costs the slowest core's total, not the mean.  Choosing
    ``per = ceil(n_tiles / cap)`` and then ``ceil(n_tiles / per)`` programs caps
    every program at ``per`` tiles while still filling every core it can reach.
    """
    cap = _num_vectorcore() * _PROGRAMS_PER_CORE
    if cap > _MAX_GRID_1D:
        cap = _MAX_GRID_1D
    if n_tiles <= cap:
        return n_tiles
    per = (n_tiles + cap - 1) // cap
    nprog = (n_tiles + per - 1) // per
    return nprog if nprog < cap else cap


@triton.jit
def _isreal_fill_kernel(o_ptr, n_tiles, BLOCK: tl.constexpr):
    """Write ``1`` to every element of the whole in-bounds tiles, strided per program."""
    npro = tl.num_programs(0)
    dto = o_ptr.dtype.element_ty
    for t in tle.dsa.pipeline(tl.program_id(0), n_tiles, npro, num_stages=2):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        o_ub = tle.dsa.to_buffer(tl.full([BLOCK], 1, dto), space=tle.dsa.ascend.UB)
        tle.dsa.copy(o_ub, o_ptr + offs, [BLOCK])


@triton.jit
def _isreal_fill_tail_kernel(o_ptr, base, n, BLOCK: tl.constexpr):
    """Masked boundary path: ``tle.dsa.copy`` cannot carry a mask."""
    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.full([BLOCK], 1, o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _isreal_complex_kernel(x_ptr, o_ptr, n_tiles, BLOCK: tl.constexpr):
    """``out = (imag == 0)`` over whole tiles of ``BLOCK`` complex elements.

    The interleaved pair tile is staged into UB by an explicit ``tle.dsa`` MTE2
    copy and read back from that buffer, so the input fill is a copy the
    pipeline stage can keep in flight instead of a load exposed against the
    predicate.  Two constraints pin the exact form: ``tle.dsa.copy`` needs the
    source and destination *shapes* to match, and its GM operand must be a
    pointer **tensor** (a bare ``ptr + offs`` scalar either fails the
    ``memref.copy`` shape check or cannot be re-materialised in
    ``TritonToLinalgIncubated``).  Hence the ``[BLOCK, 2]`` destination paired
    with a 2-D pointer tensor, which also lets ``tl.split`` de-interleave the
    ``re`` / ``im`` planes with no ``tl.reshape`` -- and a ``tl.reshape`` of a
    ``to_tensor`` result would drop the UB address space anyway.
    """
    npro = tl.num_programs(0)
    dtx = x_ptr.dtype.element_ty
    dto = o_ptr.dtype.element_ty
    for t in tle.dsa.pipeline(tl.program_id(0), n_tiles, npro, num_stages=2):
        x_ub = tle.dsa.alloc([BLOCK, 2], dtype=dtx, mem_addr_space=tle.dsa.ascend.UB)
        rows = t * BLOCK + tl.arange(0, BLOCK)
        tle.dsa.copy(
            x_ptr + rows[:, None] * 2 + tl.arange(0, 2)[None, :], x_ub, [BLOCK, 2]
        )
        _, imag = tl.split(tle.dsa.to_tensor(x_ub))
        o_ub = tle.dsa.to_buffer((imag == 0.0).to(dto), space=tle.dsa.ascend.UB)
        tle.dsa.copy(o_ub, o_ptr + t * BLOCK + tl.arange(0, BLOCK), [BLOCK])


@triton.jit
def _isreal_complex_tail_kernel(x_ptr, o_ptr, base, n, BLOCK: tl.constexpr):
    """Masked boundary path for the ragged complex tail."""
    start = base + tl.program_id(0) * BLOCK
    off = start + tl.arange(0, BLOCK)
    m = off < n
    flat = 2 * start + tl.arange(0, 2 * BLOCK)
    fm = flat < 2 * n
    pairs = tl.load(x_ptr + flat, mask=fm, other=0.0)
    _, imag = tl.split(tl.reshape(pairs, [BLOCK, 2]))
    tl.store(o_ptr + off, (imag == 0.0).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _isreal_c128_kernel(x_ptr, o_ptr, n_tiles, BLOCK: tl.constexpr):
    """``out = (imag == 0)`` for ``complex128``, tested on 32-bit words.

    ``x_ptr`` is the flattened ``(re, im)`` pair buffer reinterpreted as
    ``int32``, so complex element ``k`` owns the words ``4k .. 4k+3`` in the
    order ``re_lo, re_hi, im_lo, im_hi``.  Splitting the word tile twice
    de-interleaves those four planes with no gather, and the imaginary test
    stays entirely in 32-bit integer ALU: a ``float64`` is ``+-0.0`` exactly
    when every bit except the sign is zero, i.e.
    ``((im_hi & 0x7FFFFFFF) | im_lo) == 0``.  That is bit-exact for ``+0.0``,
    ``-0.0``, ``NaN`` and ``inf``, and it never asks the vector unit for a
    64-bit *float* value -- the compiler's ``vbrc`` broadcast has no ``f64``
    case, so ``imag == 0.0`` on ``float64`` cannot be lowered at all.
    """
    npro = tl.num_programs(0)
    dto = o_ptr.dtype.element_ty
    for t in tle.dsa.pipeline(tl.program_id(0), n_tiles, npro, num_stages=2):
        word = tl.load(x_ptr + t * (4 * BLOCK) + tl.arange(0, 4 * BLOCK))
        even, odd = tl.split(tl.reshape(word, [2 * BLOCK, 2]))
        _, im_lo = tl.split(tl.reshape(even, [BLOCK, 2]))
        _, im_hi = tl.split(tl.reshape(odd, [BLOCK, 2]))
        real = ((im_hi & 0x7FFFFFFF) | im_lo) == 0
        o_ub = tle.dsa.to_buffer(real.to(dto), space=tle.dsa.ascend.UB)
        tle.dsa.copy(o_ub, o_ptr + t * BLOCK + tl.arange(0, BLOCK), [BLOCK])


@triton.jit
def _isreal_c128_tail_kernel(x_ptr, o_ptr, base, n, BLOCK: tl.constexpr):
    """Masked boundary path for the ragged ``complex128`` tail."""
    start = base + tl.program_id(0) * BLOCK
    off = start + tl.arange(0, BLOCK)
    m = off < n
    a = 4 * start + tl.arange(0, 4 * BLOCK)
    fm = a < 4 * n
    word = tl.load(x_ptr + a, mask=fm, other=1)
    even, odd = tl.split(tl.reshape(word, [2 * BLOCK, 2]))
    _, im_lo = tl.split(tl.reshape(even, [BLOCK, 2]))
    _, im_hi = tl.split(tl.reshape(odd, [BLOCK, 2]))
    real = ((im_hi & 0x7FFFFFFF) | im_lo) == 0
    tl.store(o_ptr + off, real.to(o_ptr.dtype.element_ty), mask=m)


def _run_fill(x):
    """All-True result for a dtype without an imaginary part (input never read)."""
    n = x.numel()
    o = torch.empty(x.size(), dtype=torch.int8, device=x.device)
    if n:
        n_tiles = n // _FILL_BLOCK
        if n_tiles:
            _isreal_fill_kernel[(_grid(n_tiles),)](o, n_tiles, BLOCK=_FILL_BLOCK)
        rest = n - n_tiles * _FILL_BLOCK
        if rest:
            tg = ((rest + _TAIL_FILL_BLOCK - 1) // _TAIL_FILL_BLOCK,)
            _isreal_fill_tail_kernel[tg](
                o, n_tiles * _FILL_BLOCK, n, BLOCK=_TAIL_FILL_BLOCK
            )
    return o.view(torch.bool)


def _run_complex(x):
    """``imag == 0`` for an interleaved complex operand."""
    if x.is_conj():
        x = x.resolve_conj()
    if x.is_neg():
        x = x.resolve_neg()
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.numel()
    o = torch.empty(x.size(), dtype=torch.int8, device=x.device)
    if n:
        xf = torch.view_as_real(x).reshape(-1)
        block = _C_TILE_BYTES // (2 * xf.element_size())
        n_tiles = n // block
        if n_tiles:
            _isreal_complex_kernel[(_grid(n_tiles),)](xf, o, n_tiles, BLOCK=block)
        rest = n - n_tiles * block
        if rest:
            tg = ((rest + _TAIL_C_BLOCK - 1) // _TAIL_C_BLOCK,)
            _isreal_complex_tail_kernel[tg](
                xf, o, n_tiles * block, n, BLOCK=_TAIL_C_BLOCK
            )
    return o.view(torch.bool)


def _run_complex128(x):
    """``imag == 0`` for a ``complex128`` operand, tested on 32-bit words.

    The ``(re, im)`` pair buffer is reinterpreted as ``int32`` so the vector
    unit never has to form a ``float64`` comparison against ``0.0``.
    """
    if x.is_conj():
        x = x.resolve_conj()
    if x.is_neg():
        x = x.resolve_neg()
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.numel()
    o = torch.empty(x.size(), dtype=torch.int8, device=x.device)
    if n:
        # 2 float64 values per element -> 4 int32 words (an even count, so the
        # reinterpretation is legal for every shape including 0-dim).
        words = torch.view_as_real(x).reshape(-1).view(torch.int32)
        block = _C_TILE_BYTES // 16
        n_tiles = n // block
        if n_tiles:
            _isreal_c128_kernel[(_grid(n_tiles),)](words, o, n_tiles, BLOCK=block)
        rest = n - n_tiles * block
        if rest:
            tg = ((rest + _TAIL_C_BLOCK - 1) // _TAIL_C_BLOCK,)
            _isreal_c128_tail_kernel[tg](
                words, o, n_tiles * block, n, BLOCK=_TAIL_C_BLOCK
            )
    return o.view(torch.bool)


def isreal(self):
    """``isreal(self)`` -> freshly allocated ``bool`` tensor of ``self.shape``."""
    logger.debug("GEMS ISREAL")
    if not isinstance(self, torch.Tensor):
        raise TypeError("isreal(): argument 'self' must be a Tensor")
    dt = self.dtype
    if dt == torch.complex128:
        return _run_complex128(self)
    if dt in _COMPLEX_DTYPES:
        return _run_complex(self)
    if dt in _REAL_DTYPES:
        return _run_fill(self)
    raise TypeError(f"isreal(): unsupported dtype {dt}")
