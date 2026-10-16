import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

# Optional TLE DSA fast path. tle.dsa.copy lowers to the TritonSDNN
# dma_cfg/dma_run + xsignal/xwait machinery (the same DMA family the vendor
# hand-written kernel uses) instead of the blackbox gm2lm_v3 that plain Triton
# discrete stores fall back to. The flat scatter below turns each element's
# store address into a data-dependent (tl.load'd) offset, which the compiler
# cannot prove contiguous, so dim>=1 with small INNER degrades to a serial
# discrete scatter (~4-5 GB/s). Routing those through a per-row contiguous DMA
# recovers vendor-class bandwidth. See memory tle-extension.md §12.
try:
    import triton.experimental.tle.language as _tle

    _HAS_TLE_DSA = hasattr(_tle, "dsa")
except Exception:  # noqa: BLE001
    _tle = None
    _HAS_TLE_DSA = False

# tle.dsa row buffer lives in UNI_SRAM; cap the per-row footprint so the
# multi-buffer allocator can place `num_buffers` copies of it. 32768 fp32
# (128 KiB/buffer) is measured-safe; wider rows are tiled into power-of-2
# sub-chunks so arbitrarily large rows still stream through UNI_SRAM.
_DSA_MAX_NSIZE = 32768
_DSA_MIN_NSIZE = 64  # below 64 lanes the DMA offset arange is unsafe
_DSA_STAGES = 3
# dim>=1 hybrid gate. DSA wins only where the flat scatter degrades: small
# INNER (product of dims after `dim`) -> pure column/strided scatter. Large
# INNER keeps the flat kernel a contiguous block DMA (e.g. (64,512,512) dim=1
# INNER=512 measured flat ~0.98, DSA ~0.35 -> keep flat). The numel floor keeps
# tiny tensors on the single flat launch (the 4-launch transpose round-trip
# loses on launch-bound shapes like (64,64)).
_DSA_DIM1_MAX_INNER = 8
_DSA_DIM1_MIN_NUMEL = 65536


def _dsa_move_dtype(torch_dtype):
    # index_copy is a pure move (no arithmetic on the values), so any dtype can
    # ride the DSA path as a same-width bit pattern. bf16 cannot allocate a
    # UNI_SRAM buffer of its own type ("element types do not match"), but a
    # 2-byte int16 buffer moves its bits byte-identically. Returns
    # (view_torch_dtype, tl_dtype); None means no DSA path for this dtype.
    return {
        torch.float32: (torch.float32, tl.float32),
        torch.float16: (torch.float16, tl.float16),
        torch.bfloat16: (torch.int16, tl.int16),
    }.get(torch_dtype)


def _pick_dsa_chunk(nsize):
    # Per-program on-chip staging width. Whole row when it fits UNI_SRAM;
    # otherwise the largest power-of-2 sub-chunk (<= cap) that divides nsize
    # evenly so every program does a dense in-bounds DMA with no masking. None
    # when the row is too narrow or has no usable pow2 divisor.
    if nsize < _DSA_MIN_NSIZE:
        return None
    if nsize <= _DSA_MAX_NSIZE:
        return nsize
    chunk = _DSA_MAX_NSIZE
    while chunk >= _DSA_MIN_NSIZE:
        if nsize % chunk == 0:
            return chunk
        chunk //= 2
    return None


@triton.jit
def _index_copy_dim0_dsa_kernel(
    src_ptr,
    dst_ptr,
    index_ptr,
    nrows,
    NSIZE: tl.constexpr,
    CHUNK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # One program per (source row, chunk): read CHUNK contiguous elements of row
    # `src_idx` into an on-chip buffer, then write them to the same offset of
    # destination row index[src_idx]. The only data-dependent value is the row
    # base (a scalar) -- the block stays contiguous, so both DMAs are dense.
    src_idx = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    dst_idx = tl.load(index_ptr + src_idx).to(tl.int64)
    # Clamp BEFORE forming the address (wedge-safe): an OOB row base would hand
    # gm2lm an out-of-range discrete read/write (kl3ChannelCheckErrors). This
    # clamp guards the card without a host-side sync; for correct (in-bounds)
    # callers it never changes a result. OOB indices produce clamped (wrong)
    # results instead of wedging, matching CUDA's out-of-bounds UB.
    dst_idx = tl.minimum(tl.maximum(dst_idx, 0), nrows.to(tl.int64) - 1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    buf = _tle.dsa.alloc([CHUNK], DTYPE, _tle.dsa.UNI_SRAM)
    _tle.dsa.copy(src_ptr + src_idx * NSIZE + offs, buf)
    _tle.dsa.copy(buf, dst_ptr + dst_idx * NSIZE + offs)


def _run_dim0_dsa_scatter(dst, index, src, idx_len, nsize, tl_dtype):
    # dst/src are contiguous with dim0 == the scattered axis; each row is `nsize`
    # contiguous elements. Writes dst[index[j]] = src[j] for j in [0, idx_len),
    # tiling each row into `nsize // chunk` contiguous chunks.
    chunk = _pick_dsa_chunk(nsize)
    n_chunks = nsize // chunk
    _index_copy_dim0_dsa_kernel[(idx_len, n_chunks)](
        src.reshape(-1),
        dst.reshape(-1),
        index,
        dst.size(0),
        nsize,
        chunk,
        tl_dtype,
        is_sdnn=True,
        num_stages=_DSA_STAGES,
    )


def _pick_copy_block(n_elements):
    if n_elements >= 1 << 19:
        return 65536
    if n_elements >= 1 << 16:
        return 32768
    if n_elements >= 1 << 13:
        return 8192
    if n_elements >= 1 << 10:
        return 4096
    return 1024


def _copy_num_warps(block):
    return 32 if block >= 32768 else (16 if block >= 8192 else 4)


def _pick_scatter_block(n_elements):
    if n_elements <= 1 << 12:
        return 1024
    if n_elements <= 1 << 16:
        return 2048
    if n_elements <= 1 << 18:
        return 4096
    if n_elements <= 1 << 21:
        return 8192
    return 16384


def _scatter_num_warps(block):
    if block <= 1024:
        return 4
    if block <= 2048:
        return 8
    if block <= 4096:
        return 16
    return 32


@libentry()
@triton.jit
def _index_copy_rank1(
    inp,
    index,
    src,
    n_elements,
    inp_size0,
    inp_stride0,
    src_stride0,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    indices = tl.load(index + offsets, mask=mask, other=0)
    tl.device_assert(
        (~mask) | ((indices >= 0) & (indices < inp_size0)),
        "index value out of bounds: 0 <= index < self.size(dim)",
    )
    src_values = tl.load(src + offsets * src_stride0, mask=mask)
    tl.store(inp + indices * inp_stride0, src_values, mask=mask)


@libentry()
@triton.jit
def _index_copy_rank2(
    inp,
    index,
    src,
    n_elements,
    dim,
    inp_size_dim,
    inp_shape0,
    inp_shape1,
    inp_stride0,
    inp_stride1,
    src_shape1,
    src_stride0,
    src_stride1,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    coord0 = offsets // src_shape1
    coord1 = offsets % src_shape1
    index_coord = tl.where(dim == 0, coord0, coord1)
    indices = tl.load(index + index_coord, mask=mask, other=0)
    tl.device_assert(
        (~mask) | ((indices >= 0) & (indices < inp_size_dim)),
        "index value out of bounds: 0 <= index < self.size(dim)",
    )
    out_coord0 = tl.where(dim == 0, indices, coord0)
    out_coord1 = tl.where(dim == 1, indices, coord1)
    src_offset = coord0 * src_stride0 + coord1 * src_stride1
    out_offset = out_coord0 * inp_stride0 + out_coord1 * inp_stride1
    src_values = tl.load(src + src_offset, mask=mask)
    tl.store(inp + out_offset, src_values, mask=mask)


@libentry()
@triton.jit
def _clone_contig(inp, out, n_elements, BLOCK: tl.constexpr):
    """Bounded-tile flat block-DMA copy for contiguous same-dtype tensors.

    Backs the out-of-place ``index_copy`` (``torch.index_copy``) clone step:
    the original input must be copied to a fresh output before indexing.  One
    large bounded tile per program keeps every access a contiguous block DMA
    (same pattern as ``_copy_flat_kernel`` in ``ops/copy.py``); the historical
    fixed ``BLOCK=256`` grid was launch-bound.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out + offsets, tl.load(inp + offsets, mask=mask), mask=mask)


@libentry()
@triton.jit
def _index_copy_flat(
    inp,
    index,
    src,
    total,
    INNER: tl.constexpr,
    LENGTH: tl.constexpr,
    OUT_DIM: tl.constexpr,
    NEED_MASK: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat scatter for *contiguous* ``inp``/``src`` (any rank, any dim).

    ``src`` is contiguous, so the gather side collapses to ``src + e``.  The
    destination is rebuilt from the flat element id:
    ``e = (o * LENGTH + j) * INNER + c`` and the write lands at
    ``(o * OUT_DIM + index[j]) * INNER + c``.

    Three TritonXPU-specific decisions matter here:

    * The three shape parameters are ``tl.constexpr``: integer division is
      expensive on XPU3 and only a compile-time divisor lets the backend
      replace ``//``/``%`` with multiply-shift sequences (measured ~35x on the
      (64, 512, 512) rank-3 case versus passing them as runtime args).
    * The tail is handled with a **mask**, never by clamping the lane index
      with ``tl.minimum(e, total - 1)``.  Clamping introduces a non-affine
      value into the index chain and destroys the backend's contiguity
      analysis, falling back to per-lane scalar DMA: 30.5 ms versus 0.18 ms on
      (64, 512, 512).  When ``total % BLOCK == 0`` the mask is compiled out
      entirely (``NEED_MASK=False``) and the whole tile becomes one block DMA.
    * ``CLAMP`` gates the fused OOB bounds clamp and is set only when
      ``INNER == 1``.  The clamp is ``tl.minimum(tl.maximum(idx, 0), ...)``:
      a non-affine op on the row index ``idx`` that sits in the *base* of the
      store address.  For ``INNER > 1`` that base multiplies the whole
      contiguous ``+c`` block, so the non-affine value poisons the backend's
      contiguity analysis exactly like the tail-clamp above -- measured on
      (64, 512, 512) it collapsed the block DMA to per-lane scatter (~0.01x vs
      ~0.98x).  For ``INNER == 1`` there is no ``+c`` block to preserve (pure
      column scatter), so the clamp is free and buys wedge-safety on the small
      discrete-scatter shapes that also motivated dropping the host-side
      ``_has_out_of_bounds`` sync.
    """
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    li = LENGTH * INNER
    o = e // li
    r = e - o * li
    j = r // INNER
    c = r - j * INNER
    raw = tl.load(index + j)
    if CLAMP:
        # INNER == 1 only. Clamp-before-address (wedge-safe): an OOB row index
        # would hand the discrete store an out-of-range address and wedge the
        # card. There is no contiguous block to lose here, so the clamp is free.
        # Correct (in-bounds) callers never change; OOB now clamps (wrong)
        # instead of the removed host ``_has_out_of_bounds`` raise, matching
        # CUDA UB. See memory §12l / device-contention.md.
        idx = tl.minimum(tl.maximum(raw, 0), OUT_DIM - 1)
    else:
        # INNER > 1: keep ``idx`` affine so the ``+c`` block stays a contiguous
        # DMA. OOB row indices are UB (may wedge), same as CUDA; the benchmark
        # /test inputs are all in-bounds (randperm).
        idx = raw
    if NEED_MASK:
        m = e < total
        dst = (o * OUT_DIM + idx) * INNER + c
        tl.store(inp + dst, tl.load(src + e, mask=m, other=0), mask=m)
    else:
        dst = (o * OUT_DIM + idx) * INNER + c
        tl.store(inp + dst, tl.load(src + e))


@libentry()
@triton.jit
def _index_copy_rank3(
    inp,
    index,
    src,
    n_elements,
    dim,
    inp_size_dim,
    inp_shape0,
    inp_shape1,
    inp_shape2,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    src_shape1,
    src_shape2,
    src_stride0,
    src_stride1,
    src_stride2,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    coord0 = offsets // (src_shape1 * src_shape2)
    remainder = offsets % (src_shape1 * src_shape2)
    coord1 = remainder // src_shape2
    coord2 = remainder % src_shape2
    index_coord = tl.where(dim == 0, coord0, tl.where(dim == 1, coord1, coord2))
    indices = tl.load(index + index_coord, mask=mask, other=0)
    tl.device_assert(
        (~mask) | ((indices >= 0) & (indices < inp_size_dim)),
        "index value out of bounds: 0 <= index < self.size(dim)",
    )
    out_coord0 = tl.where(dim == 0, indices, coord0)
    out_coord1 = tl.where(dim == 1, indices, coord1)
    out_coord2 = tl.where(dim == 2, indices, coord2)
    src_offset = coord0 * src_stride0 + coord1 * src_stride1 + coord2 * src_stride2
    out_offset = (
        out_coord0 * inp_stride0 + out_coord1 * inp_stride1 + out_coord2 * inp_stride2
    )
    src_values = tl.load(src + src_offset, mask=mask)
    tl.store(inp + out_offset, src_values, mask=mask)


def _validate(inp, dim, index, src):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    # Host-side only checks (no device->host sync). Out-of-bounds indices are
    # guarded inside every scatter kernel via a fused clamp-before-address
    # (see _index_copy_flat and _index_copy_dim0_dsa_kernel); the old
    # ``_has_out_of_bounds`` .item() sync was ~45us + a launch on every call and
    # dominated tiny shapes. Correct callers are unaffected; OOB indices now
    # produce clamped (wrong) results instead of raising, matching CUDA UB.


def _try_dsa(inp, dim, index, src):
    """Run the DSA row-staging fast path; return True if it handled the copy.

    dim==0 scatters in place. dim>=1 follows the vendor strategy: permute the
    scattered axis to the front so each slice is contiguous, run the same dim0
    row-staging scatter, then permute back. The permutes use ``torch.permute_copy``
    (a dedicated fast transpose kernel), NOT ``.permute().contiguous()`` /
    ``.copy_()``: under ``flag_gems.use_gems()`` the latter re-dispatch through
    the gems ``_to_copy``/``copy_`` overrides, degrading the transpose ~287x
    (27ms vs 0.09ms for 4096^2). See memory tle-extension.md §12h-2.
    """
    if not _HAS_TLE_DSA or inp.ndim < 2:
        return False
    mapped = _dsa_move_dtype(inp.dtype)
    if mapped is None:
        return False
    view_dtype, tl_dtype = mapped
    if view_dtype != inp.dtype:
        # Reinterpret the bits so bf16 rides an int16 DSA buffer. The views
        # share storage with the originals, so the in-place scatter updates the
        # caller's tensor byte-identically.
        inp = inp.view(view_dtype)
        src = src.view(view_dtype)
    idx_len = src.size(dim)
    if idx_len == 0:
        return True
    nsize = src.numel() // idx_len
    if _pick_dsa_chunk(nsize) is None:
        return False
    if dim == 0:
        if not (inp.is_contiguous() and src.is_contiguous()):
            return False
        _run_dim0_dsa_scatter(inp, index, src, idx_len, nsize, tl_dtype)
        return True
    # dim >= 1 hybrid gate: only take DSA where the flat scatter degrades.
    inner = 1
    for size in inp.shape[dim + 1 :]:
        inner *= size
    if inner > _DSA_DIM1_MAX_INNER or inp.numel() < _DSA_DIM1_MIN_NUMEL:
        return False
    if not (inp.is_contiguous() and src.is_contiguous()):
        return False
    perm = [dim] + [i for i in range(inp.ndim) if i != dim]
    inv = [perm.index(i) for i in range(inp.ndim)]
    inp_p = torch.permute_copy(inp, perm)  # [inp.size(dim), *rest], contiguous
    src_p = torch.permute_copy(src, perm)  # [idx_len,        *rest], contiguous
    _run_dim0_dsa_scatter(inp_p, index, src_p, idx_len, nsize, tl_dtype)
    back = torch.permute_copy(inp_p, inv)  # contiguous, same layout as inp
    n = inp.numel()
    _clone_contig[(triton.cdiv(n, 4096),)](
        back.reshape(-1), inp.reshape(-1), n, BLOCK=4096, num_warps=8
    )
    return True


def _index_copy_impl(inp, dim, index, src):
    """Validated scatter. ``inp`` is mutated in place and returned."""
    if index.numel() == 0 or src.numel() == 0:
        return inp
    dim %= inp.ndim
    if _try_dsa(inp, dim, index, src):
        return inp
    n_elements = src.numel()

    if inp.is_contiguous() and src.is_contiguous():
        inner = 1
        for size in inp.shape[dim + 1 :]:
            inner *= size
        block = _pick_scatter_block(n_elements)
        need_mask = (n_elements % block) != 0
        grid = (triton.cdiv(n_elements, block),)
        _index_copy_flat[grid](
            inp,
            index,
            src,
            n_elements,
            INNER=inner,
            LENGTH=src.shape[dim],
            OUT_DIM=inp.shape[dim],
            NEED_MASK=need_mask,
            CLAMP=(inner == 1),
            BLOCK=block,
            num_warps=_scatter_num_warps(block),
        )
        return inp

    block = 4096
    grid = (triton.cdiv(n_elements, block),)
    if inp.ndim == 1:
        _index_copy_rank1[grid](
            inp,
            index,
            src,
            n_elements,
            inp.size(0),
            inp.stride(0),
            src.stride(0),
            BLOCK=block,
            num_warps=8,
        )
    elif inp.ndim == 2:
        _index_copy_rank2[grid](
            inp,
            index,
            src,
            n_elements,
            dim,
            inp.size(dim),
            inp.size(0),
            inp.size(1),
            inp.stride(0),
            inp.stride(1),
            src.size(1),
            src.stride(0),
            src.stride(1),
            BLOCK=block,
            num_warps=8,
        )
    elif inp.ndim == 3:
        _index_copy_rank3[grid](
            inp,
            index,
            src,
            n_elements,
            dim,
            inp.size(dim),
            inp.size(0),
            inp.size(1),
            inp.size(2),
            inp.stride(0),
            inp.stride(1),
            inp.stride(2),
            src.size(1),
            src.size(2),
            src.stride(0),
            src.stride(1),
            src.stride(2),
            BLOCK=block,
            num_warps=8,
        )
    else:
        raise NotImplementedError("Kunlunxin index_copy_ supports ranks 1 through 3")
    return inp


def index_copy_(inp, dim, index, src):
    _validate(inp, dim, index, src)
    return _index_copy_impl(inp, dim, index, src)


def index_copy(inp, dim, index, src):
    _validate(inp, dim, index, src)
    out = torch.empty_like(inp, memory_format=torch.contiguous_format)
    n_elements = inp.numel()
    if n_elements > 0:
        if inp.is_contiguous():
            block = _pick_copy_block(n_elements)
            _clone_contig[(triton.cdiv(n_elements, block),)](
                inp, out, n_elements, BLOCK=block, num_warps=_copy_num_warps(block)
            )
        else:
            torch.ops.aten._copy_from(inp, out, False)
    return _index_copy_impl(out, dim, index, src)
