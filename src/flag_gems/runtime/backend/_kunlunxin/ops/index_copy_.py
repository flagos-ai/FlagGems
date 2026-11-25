import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy
from .copy import copy_ as _copy_into

_TRANS_MIN_ELEMENTS = 1 << 16  # fewer elements cannot pay for 3 extra launches
_TRANS_MAX_INNER = 16  # above this the flat scatter already has wide runs
_TRANS_MIN_STRIP = 64  # a shorter strip cannot be padded profitably
_STRIP_ALIGN = 64


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


def _strided_rows(buf, perm_shape, strip, row_stride):
    """View ``buf`` (rows x row_stride) as ``perm_shape`` (rows x *tail)."""
    if row_stride == strip:
        return buf.view(perm_shape)
    strides = [row_stride]
    acc = 1
    for size in reversed(perm_shape[1:]):
        strides.insert(1, acc)
        acc *= size
    return buf.as_strided(perm_shape, strides)


def _copy_permuted(src_view, dst_view):
    """Move a permuted view through the vendor DMA engine.

    False means tle cannot express the layout; the caller then abandons the
    transposed pipeline (without touching ``dst``) and uses the flat scatter.
    """
    return tle_copy(src_view, dst_view)


def _scatter_rows(dst_rows, index, src_rows, length, out_dim, strip, block):
    n_elements = src_rows.numel()
    _index_copy_flat[(triton.cdiv(n_elements, block),)](
        dst_rows,
        index,
        src_rows,
        n_elements,
        INNER=strip,
        LENGTH=length,
        OUT_DIM=out_dim,
        NEED_MASK=(n_elements % block) != 0,
        BLOCK=block,
        num_warps=8,
    )


def _index_copy_transposed(dst, dim, index, src):
    """Scatter with ``dim`` moved to the front so each index owns one strip.

    Returns False -- without touching ``dst`` -- when the copy engine cannot
    express one of the transposes; the caller then falls back to the flat
    scatter.
    """
    rank = dst.ndim
    perm = [dim] + [i for i in range(rank) if i != dim]
    length = src.shape[dim]
    out_dim = dst.shape[dim]
    strip = src.numel() // length
    strip_pad = (strip + _STRIP_ALIGN - 1) // _STRIP_ALIGN * _STRIP_ALIGN
    src_perm = src.permute(perm)
    dst_perm = dst.permute(perm)

    src_buf = torch.empty((length, strip_pad), dtype=src.dtype, device=src.device)
    dst_buf = torch.empty((out_dim, strip_pad), dtype=dst.dtype, device=dst.device)
    src_view = _strided_rows(src_buf, src_perm.shape, strip, strip_pad)
    dst_view = _strided_rows(dst_buf, dst_perm.shape, strip, strip_pad)
    if not _copy_permuted(src_perm, src_view):
        return False
    if not _copy_permuted(dst_perm, dst_view):
        return False

    block = 8192 if strip_pad <= 4096 else 16384
    _scatter_rows(dst_buf, index, src_buf, length, out_dim, strip_pad, block)
    return _copy_permuted(dst_view, dst_perm)


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
        outer = 1
        for size in inp.shape[:dim]:
            outer *= size

        if (
            n_elements >= _TRANS_MIN_ELEMENTS
            and inner < _TRANS_MAX_INNER
            and outer * inner >= _TRANS_MIN_STRIP
            and _index_copy_transposed(inp, dim, index, src)
        ):
            return inp

        block = _pick_scatter_block(n_elements)
        _scatter_rows(
            inp,
            index,
            src,
            src.shape[dim],
            inp.shape[dim],
            inner,
            block,
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
    if inp.numel() > 0 and not tle_copy(inp, out):
        # Layouts tle cannot move (e.g. a 1-byte transposed source) go through
        # the gems copy_ op, which keeps its own Triton pointwise fallback.
        _copy_into(out, inp)
    return _index_copy_impl(out, dim, index, src)
