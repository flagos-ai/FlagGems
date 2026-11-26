"""TLE (Ascend, vec-explicit) implementation of
``torch.ops.aten.value_selecting_reduction_backward``.

Semantics contract (FlagGems reference ``value_selecting_reduction_backward``)::

    assert grad.dtype not in (torch.bool,)
    result = grad.new_zeros(sizes)                 # dtype == grad.dtype
    if not keepdim:
        grad = grad.unsqueeze(dim); indices = indices.unsqueeze(dim)
    scatter_(result, dim, indices, grad)           # pure assignment, no reduce
    return result

``scatter_`` with ``reduce=None`` issues exactly one ``tl.store`` per source
element, so the operator is a *value-selecting* (one-hot per reduction group)
copy: every output element that is not selected by ``indices`` stays ``+0.0``,
and each selected element receives the upstream gradient *bit for bit*.  There
is no accumulation, so the result is a pure copy and the oracle
(``utils.gems_assert_equal``) is exact -- ``atol == rtol == 0`` plus a dtype
check.  Nothing here may round, reorder or widen the stored values: fp32 in
means fp32 out, fp16 in means fp16 out.

Shape decomposition (all indices are into the *contiguous* layout of ``out``,
which is freshly allocated with ``torch.empty``):

    outer = prod(sizes[:dim])       # leading groups
    L     = sizes[dim]              # the reduced (scattered-into) axis
    inner = prod(sizes[dim+1:])     # trailing elements per row of the axis
    out[o, l, j]  at  (o * L + l) * inner + j
    grad/indices are contiguous with ``numel == outer * inner`` and flat index
    ``r = o * inner + j``.

Two device paths, both TLE:

``inner == 1`` and ``L`` a power of two in ``[_MIN_L, _MAX_L]`` ("fast path"):
    Every row ``r`` of ``out`` is a contiguous run of ``L`` elements whose only
    non-zero element sits at ``indices[r]``.  Rows are processed in *tiles* of
    ``ROWS`` consecutive rows: the ``ROWS`` indices/gradients are fetched with
    one vector ``tl.load`` each, the ``(ROWS, L)`` selection is built with a
    single broadcast compare, and the whole tile (``ROWS * L`` contiguous
    elements) leaves the vector unit with ONE bulk ``tle.dsa.copy``.  Cost is
    therefore exactly one store per output element, and the tile is sized in
    *bytes* so the fixed per-iteration cost of the pipeline (measured at
    ~200 cycles/iteration on the eval part) is amortised the way it already is
    on the wide-row shapes.

    Tiling granularity is load-bearing: with one row per iteration the 512-wide
    and 256-wide shapes spend ~85% of each iteration on scalar bookkeeping
    (``aiv_scalar_ratio`` 0.996, 261 GB/s) whereas the same source at 32 KB per
    copy reaches 1481 GB/s.  ``ROWS`` is chosen so a tile is ~``_BYTES_PER_TILE``
    bytes, while never dropping below one tile per resident program (that keeps
    the small shapes parallel) and never exceeding the number of rows that exist.

``inner > 1`` / odd ``L`` ("generic path"):
    The selection offset of row ``r`` is ``(o * L + idx[r]) * inner + j``, which
    is not a stride-``L`` row, so the one-pass tile trick does not apply.  A
    vector zero-fill of ``out`` followed by a fully vectorised scatter of the
    ``outer * inner`` selected elements reproduces the reference exactly for any
    rank, any ``dim`` and any shape.

Grid policy: both paths are capped by the *runtime-queried* ``num_vectorcore``
of the live device (never a hard-coded 910B SKU core count -- the submission
target is Ascend910B2 while the evaluation service may report a different
910B part), and every loop strides by ``tl.num_programs(0)`` so a capped grid
still covers all the work.  Masked tails use ``tl.load``/``tl.store`` because
``tle.dsa.copy`` takes an explicit shape but no mask.
"""

import logging

import torch
import torch_npu  # noqa: F401  -- must precede `import triton` in this env
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

# Fast-path row width bounds (elements).  Below _MIN_L a whole-row bulk copy is
# dominated by the per-copy fixed cost and the generic path is cheaper; above
# _MAX_L the row no longer fits comfortably in UB.
_MIN_L = 8
_MAX_L = 8192
# Payload bytes one tiled store should move.  1 KB rows reach only ~260 GB/s
# while 32 KB reaches ~1.48 TB/s on the same source, so a tile this size sits
# well inside the amortised regime while keeping two pipeline stages of UB
# (~32 KB) far below the budget.
_BYTES_PER_TILE = 16384
# Tiling only pays where a row is *small*.  A one-row 1-D copy is already the
# faster form once the row itself carries ~8 KB: round 1 `core::float16::1` ran
# 8 KB 1-D rows at 0.023695 ms while round 2's two-row 16 KB 2-D tile ran the
# same geometry at 0.033037 ms, and the round-2 profile shows why -- the 2-D
# form is vector-bound (aiv_vec 0.88, aiv_mte3 0.335) because the 2-D broadcast
# alignment and the extra UB->UB staging copy cost vector work per element,
# whereas at a 1 KB row the tile is the opposite way round by a wide margin
# (0.26 TB/s for one row vs 1.05 TB/s for a 16-row tile).  So a byte budget that
# cannot reach this many rows is not worth a 2-D copy and keeps one row per
# copy.
_MIN_TILE_ROWS = 4
# A last dimension narrower than this many bytes is expanded by the backend
# (8x UB), so such rows are left at one row per copy.
_MIN_ROW_BYTES = 64
# Payload bytes moved by one staged copy on the generic zero-fill path.
_BYTES_PER_COPY = 16384
_BLOCK_MIN = 1024
_BLOCK_MAX = 8192
# Programs per vector core: several resident per core keep MTE3 busy while the
# next tile is built on the vector unit.
_PROGRAMS_PER_CORE = 4
# Rows per program on the vectorised generic scatter path.
_SCATTER_BLOCK = 1024
# Ascend 1-D grid cap, used only if the core count cannot be queried.
_MAX_GRID_1D = 65535
# First offset that needs 64-bit addressing.
_INT32_MAX = 2**31 - 1

_NVC_CACHE = {}


def _num_vectorcore():
    """Vector-core count of the *live* target device (never hard-coded)."""
    try:
        dev = torch.npu.current_device()
    except Exception:
        dev = 0
    nvc = _NVC_CACHE.get(dev)
    if nvc is None:
        nvc = 0
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(dev)
            nvc = int(props.get("num_vectorcore", 0) or 0)
        except Exception:
            nvc = 0
        if nvc <= 0:
            nvc = 1
        _NVC_CACHE[dev] = nvc
    return nvc


def _grid_cap():
    """Largest program count this device can keep busy (work-capped later)."""
    cap = _num_vectorcore() * _PROGRAMS_PER_CORE
    return cap if cap < _MAX_GRID_1D else _MAX_GRID_1D


def _grid(n_work):
    """Work-capped 1-D grid: never more programs than useful tiles."""
    cap = _grid_cap()
    return n_work if n_work < cap else cap


def _pick_block(itemsize):
    """Generic-path tile width in elements: ~16 KB of payload."""
    block = _BYTES_PER_COPY // itemsize
    if block < _BLOCK_MIN:
        block = _BLOCK_MIN
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    return block


def _pick_rows(ldim, itemsize, n_rows, cap):
    """Rows per tiled store: ~_BYTES_PER_TILE bytes, >=1 tile per program.

    ``tl.arange`` needs a power of two, so the byte budget is rounded down to
    one; the result never exceeds the number of rows that actually exist, so a
    shape smaller than one tile simply keeps the one-row-per-copy form.
    """
    row_bytes = ldim * itemsize
    if row_bytes < _MIN_ROW_BYTES:
        return 1
    budget = _BYTES_PER_TILE // row_bytes
    if budget < 1:
        budget = 1
    per_core = n_rows // cap if cap > 0 else 0
    if per_core < budget:
        budget = per_core
    if n_rows < budget:
        budget = n_rows
    # Too few rows to fill a tile means a wide row, and a wide row is better
    # served by the cheaper 1-D copy over one whole row (see _MIN_TILE_ROWS).
    if budget < _MIN_TILE_ROWS:
        return 1
    rows = 1
    while rows * 2 <= budget:
        rows *= 2
    return rows


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


@triton.jit
def _vsrb_rows_kernel(
    idx_ptr, g_ptr, o_ptr, r0, n_rows, BLOCK_L: tl.constexpr, INT64: tl.constexpr
):
    """One whole ``BLOCK_L``-wide ``out`` row per loop iteration.

    Row ``r`` is ``tl.where(cols == idx[r], grad[r], 0)`` evaluated entirely on
    the vector unit and written with a single contiguous bulk copy, so the zero
    fill and the selected value cost one store per element in total.  ``r0``
    lets the caller start at a row that is not tile-aligned (the tail group of a
    shape whose row count is not a multiple of ``ROWS``).
    """
    npro = tl.num_programs(0)
    dto = o_ptr.dtype.element_ty
    cols = tl.arange(0, BLOCK_L)
    # The compare stays on int32 lanes: round 5 proved that an int16 compare has
    # no vector lowering on this stack (it degenerates into a per-element scalar
    # `scf.for` over UB), so the *only* narrowing here is the select's data
    # width.  `g` is cast to the output dtype before it is broadcast and the
    # false operand is a typed zero tile, so `vsel` runs on `element_size`-wide
    # lanes and no separate f32->f16 fixup pass is needed afterwards.  For a
    # float32 output this is byte-for-byte the round-3 form.
    zero = tl.zeros([BLOCK_L], dtype=dto)
    for r in tle.dsa.pipeline(r0 + tl.program_id(0), n_rows, npro, num_stages=2):
        i = tl.load(idx_ptr + r).to(tl.int32)
        g = tl.load(g_ptr + r).to(dto)
        if INT64:
            offs = r.to(tl.int64) * BLOCK_L + cols
        else:
            offs = r * BLOCK_L + cols
        # Select, never accumulate: the selected element keeps `g` bit for bit,
        # every other element is +0.0 exactly as `new_zeros` leaves it.
        val = tl.where(cols == i, g, zero)
        ob = tle.dsa.to_buffer(val, space=tle.dsa.ascend.UB)
        tle.dsa.copy(ob, o_ptr + offs, [BLOCK_L])


@triton.jit
def _vsrb_tile_kernel(
    idx_ptr,
    g_ptr,
    o_ptr,
    r0,
    n_groups,
    BLOCK_L: tl.constexpr,
    ROWS: tl.constexpr,
    INT64: tl.constexpr,
):
    """Fast path: ``ROWS`` consecutive ``out`` rows per loop iteration.

    Every group is full (``n_groups = n_rows // ROWS``), so the tile is exactly
    ``ROWS * BLOCK_L`` contiguous elements and leaves the vector unit with one
    bulk ``tle.dsa.copy`` -- the fixed per-copy cost is paid once per tile
    instead of once per row.  Selection is still a pure ``tl.where``: one store
    per output element, no accumulation, no rounding.
    """
    npro = tl.num_programs(0)
    dto = o_ptr.dtype.element_ty
    cols = tl.arange(0, BLOCK_L)
    rows = tl.arange(0, ROWS)
    # int32 compare (see the rows kernel: an int16 compare loses vectorisation
    # entirely), but the select's payload is narrowed to the output dtype so
    # the f16/bf16 tiles never materialise a float32 temporary.
    zero = tl.zeros([BLOCK_L], dtype=dto)[None, :]
    for gp in tle.dsa.pipeline(tl.program_id(0), n_groups, npro, num_stages=2):
        base = r0 + gp * ROWS
        i = tl.load(idx_ptr + base + rows).to(tl.int32)
        g = tl.load(g_ptr + base + rows).to(dto)
        # (ROWS, BLOCK_L) one-hot selection; rows are contiguous in `out`, so
        # the flattened tile is one contiguous run of ROWS * BLOCK_L elements.
        sel = cols[None, :] == i[:, None]
        val = tl.where(sel, g[:, None], zero)
        ob = tle.dsa.to_buffer(val, space=tle.dsa.ascend.UB)
        if INT64:
            offs = (base + rows)[:, None].to(tl.int64) * BLOCK_L + cols[None, :]
        else:
            offs = (base + rows)[:, None] * BLOCK_L + cols[None, :]
        tle.dsa.copy(ob, o_ptr + offs, [ROWS, BLOCK_L])


@triton.jit
def _vsrb_fill_kernel(o_ptr, n, BLOCK: tl.constexpr, INT64: tl.constexpr):
    """Generic path step 1: vector zero-fill of the whole output."""
    npro = tl.num_programs(0)
    dto = o_ptr.dtype.element_ty
    nt = tl.cdiv(n, BLOCK)
    for t in tle.dsa.pipeline(tl.program_id(0), nt, npro, num_stages=2):
        if INT64:
            offs = t.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32).to(dto), mask=m)


@triton.jit
def _vsrb_scatter_kernel(
    idx_ptr, g_ptr, o_ptr, n_rows, ldim, inner, BLOCK: tl.constexpr, INT64: tl.constexpr
):
    """Generic path step 2: scatter the ``outer * inner`` selected elements.

    Row ``r`` has ``o = r // inner``, ``j = r % inner`` and lands at
    ``(o * ldim + indices[r]) * inner + j``; every write is a distinct element
    (value-selecting reductions have one index per group), so no atomics and no
    reduction are needed.
    """
    if INT64:
        off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n_rows
    i = tl.load(idx_ptr + off, mask=m, other=0)
    g = tl.load(g_ptr + off, mask=m, other=0)
    if INT64:
        ii = i.to(tl.int64)
    else:
        ii = i.to(tl.int32)
    o = off // inner
    j = off - o * inner
    sel = (o * ldim + ii) * inner + j
    tl.store(o_ptr + sel, g, mask=m)


def _launch_rows(ix, g, o, n_rows, ldim, itemsize, use64):
    """Fast-path launch: full tiles of ROWS rows, then a one-row tail."""
    cap = _grid_cap()
    rows = _pick_rows(ldim, itemsize, n_rows, cap)
    n_full = n_rows // rows
    rem = n_rows - n_full * rows
    if n_full > 0:
        if rows > 1:
            _vsrb_tile_kernel[(_grid(n_full),)](
                ix, g, o, 0, n_full, BLOCK_L=ldim, ROWS=rows, INT64=use64
            )
        else:
            _vsrb_rows_kernel[(_grid(n_full),)](
                ix, g, o, 0, n_full, BLOCK_L=ldim, INT64=use64
            )
    if rem > 0:
        # tle.dsa.copy has no mask, so the ragged tail keeps the masked store.
        start = n_full * rows
        _vsrb_rows_kernel[(_grid(rem),)](
            ix, g, o, start, n_rows, BLOCK_L=ldim, INT64=use64
        )


def _launch_generic(ix, g, o, n, n_rows, ldim, inner, itemsize, use64):
    """Generic-path launch: zero-fill ``o`` then scatter the selected rows."""
    block = _pick_block(itemsize)
    nt = (n + block - 1) // block
    _vsrb_fill_kernel[(_grid(nt),)](o, n, BLOCK=block, INT64=use64)
    if n_rows > 0:
        grid = ((n_rows + _SCATTER_BLOCK - 1) // _SCATTER_BLOCK,)
        _vsrb_scatter_kernel[grid](
            ix, g, o, n_rows, ldim, inner, BLOCK=_SCATTER_BLOCK, INT64=use64
        )


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim):
    """``value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim)``.

    Returns a fresh ``grad.dtype`` tensor of shape ``sizes``; ``torch.empty`` is
    used because the kernel writes *every* element of ``out`` (zeros plus the
    scattered rows), so no zero-initialised allocation is required.
    """
    logger.debug("GEMS VALUE_SELECTING_REDUCTION_BACKWARD")
    if grad.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.value_selecting_reduction_backward import (
            value_selecting_reduction_backward as _generic,
        )

        return _generic(grad, dim, indices, sizes, keepdim)

    ndim = len(sizes)
    d = int(dim)
    if d < 0:
        d += ndim

    out = torch.empty(list(sizes), dtype=grad.dtype, device=grad.device)
    n = out.numel()
    if n == 0:
        return out

    g = grad if grad.is_contiguous() else grad.contiguous()
    ix = indices if indices.is_contiguous() else indices.contiguous()

    outer = 1
    for s in sizes[:d]:
        outer *= int(s)
    ldim = int(sizes[d])
    inner = 1
    for s in sizes[d + 1 :]:
        inner *= int(s)

    gf = g.reshape(-1)
    xf = ix.reshape(-1)
    of = out.reshape(-1)
    n_rows = outer * inner
    if gf.numel() < n_rows:
        n_rows = gf.numel()
    use64 = n > _INT32_MAX or n_rows > _INT32_MAX
    itemsize = g.element_size()

    if inner == 1 and _is_pow2(ldim) and _MIN_L <= ldim <= _MAX_L:
        _launch_rows(xf, gf, of, n_rows, ldim, itemsize, use64)
    else:
        _launch_generic(xf, gf, of, n, n_rows, ldim, inner, itemsize, use64)
    return out
