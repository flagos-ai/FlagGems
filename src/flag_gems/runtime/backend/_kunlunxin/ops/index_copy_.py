import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


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
def _oob_flag_kernel(index, n, size_dim, flag, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(index + offsets, mask=mask, other=0)
    bad = tl.where(mask & ((values < 0) | (values >= size_dim)), 1, 0)
    tl.store(flag + pid, tl.minimum(tl.sum(bad, axis=0), 1))


def _has_out_of_bounds(index, size_dim):
    n = index.numel()
    if n == 0:
        return False
    block = 2048
    grid = (triton.cdiv(n, block),)
    flag = torch.empty(grid[0], dtype=torch.int32, device=index.device)
    _oob_flag_kernel[grid](index, n, size_dim, flag, BLOCK=block)
    if grid[0] == 1:
        return int(flag[0].item()) > 0
    return int(flag.max().item()) > 0


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
    BLOCK: tl.constexpr,
):
    """Flat scatter for *contiguous* ``inp``/``src`` (any rank, any dim).

    ``src`` is contiguous, so the gather side collapses to ``src + e``.  The
    destination is rebuilt from the flat element id:
    ``e = (o * LENGTH + j) * INNER + c`` and the write lands at
    ``(o * OUT_DIM + index[j]) * INNER + c``.

    Two TritonXPU-specific decisions matter here:

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
    """
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    li = LENGTH * INNER
    o = e // li
    r = e - o * li
    j = r // INNER
    c = r - j * INNER
    if NEED_MASK:
        m = e < total
        dst = (o * OUT_DIM + tl.load(index + j)) * INNER + c
        tl.store(inp + dst, tl.load(src + e, mask=m, other=0), mask=m)
    else:
        dst = (o * OUT_DIM + tl.load(index + j)) * INNER + c
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
    if index.numel() > 0:
        assert not _has_out_of_bounds(
            index, inp.size(dim)
        ), "0 <= index < self.size(dim)"


def _index_copy_impl(inp, dim, index, src):
    """Validated scatter. ``inp`` is mutated in place and returned."""
    if index.numel() == 0 or src.numel() == 0:
        return inp
    dim %= inp.ndim
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
