"""row_stack_out: out= variant of row_stack (== vstack == dim-0 concat after atleast_2d).

Pure data movement implemented with Triton copy kernels; the host wrapper only does
metadata reads, out.resize_ state maintenance and kernel launches.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["dst_base", "n_elem"])
def _flat_copy_kernel(dst_ptr, src_ptr, dst_base, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + dst_base + offs, val, mask=mask)


@triton.jit(do_not_specialize=["n_elem"])
def _flat_copy_vec_kernel(
    dst_ptr, src_ptr, dst_base, n_elem, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    """Same copy, but with offsets that are provably contiguous and aligned.

    Only launched when the destination byte offset, the source pointer and the
    out pointer are all 16-byte aligned, so the vector accesses Triton derives
    from these hints are legal.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    if EVEN:
        val = tl.load(src_ptr + offs)
        tl.store(dst_ptr + dst_base + offs, val)
    else:
        mask = offs < n_elem
        val = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + dst_base + offs, val, mask=mask)


@triton.jit(
    do_not_specialize=[
        "row_base",
        "n_elem",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
    ]
)
def _strided_copy_kernel(
    dst_ptr,
    src_ptr,
    row_base,
    n_elem,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    t0,
    t1,
    t2,
    t3,
    t4,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    inner = d1 * d2 * d3 * d4
    row = idx // inner
    rem = idx % inner
    p = d2 * d3 * d4
    c1 = rem // p
    rem = rem % p
    p = d3 * d4
    c2 = rem // p
    rem = rem % p
    c3 = rem // d4
    c4 = rem % d4
    soff = row * s0 + c1 * s1 + c2 * s2 + c3 * s3 + c4 * s4
    doff = (row + row_base) * t0 + c1 * t1 + c2 * t2 + c3 * t3 + c4 * t4
    val = tl.load(src_ptr + soff, mask=mask)
    tl.store(dst_ptr + doff, val, mask=mask)


@triton.jit
def _multi_copy_eq_kernel(
    dst_ptr, s0, s1, s2, s3, n, BLOCK: tl.constexpr, K: tl.constexpr
):
    """Equal-sized contiguous inputs, one launch: the destination base of the
    program is simply sel*n, so only the pointers and the common length are
    needed (fewer launch arguments than the general fused kernel)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    sel = tl.program_id(1)
    sp = s0
    if K > 1:
        sp = tl.where(sel == 1, s1, sp)
    if K > 2:
        sp = tl.where(sel == 2, s2, sp)
    if K > 3:
        sp = tl.where(sel == 3, s3, sp)
    val = tl.load(sp + offs)
    tl.store(dst_ptr + sel.to(tl.int64) * n + offs, val)


@triton.jit(do_not_specialize=["b0", "b1", "b2", "b3", "n0", "n1", "n2", "n3"])
def _multi_copy_vec_kernel(
    dst_ptr,
    s0,
    s1,
    s2,
    s3,
    b0,
    b1,
    b2,
    b3,
    n0,
    n1,
    n2,
    n3,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """All inputs copied by one launch: program_id(1) selects the source.

    Same alignment preconditions as _flat_copy_vec_kernel (host-gated).
    """
    pid = tl.program_id(0)
    sel = tl.program_id(1)
    sp = s0
    sp = tl.where(sel == 1, s1, sp)
    sp = tl.where(sel == 2, s2, sp)
    sp = tl.where(sel == 3, s3, sp)
    base = b0
    base = tl.where(sel == 1, b1, base)
    base = tl.where(sel == 2, b2, base)
    base = tl.where(sel == 3, b3, base)
    n = n0
    n = tl.where(sel == 1, n1, n)
    n = tl.where(sel == 2, n2, n)
    n = tl.where(sel == 3, n3, n)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    if EVEN:
        val = tl.load(sp + offs)
        tl.store(dst_ptr + base + offs, val)
    else:
        mask = offs < n
        val = tl.load(sp + offs, mask=mask)
        tl.store(dst_ptr + base + offs, val, mask=mask)


_FLAT_BLOCK = 4096
_STRIDED_BLOCK = 1024
_MAX_TRAIL = 4
_overlaps = torch._C._overlaps


def _byte_range(t):
    span = 0
    for size, stride in zip(t.shape, t.stride()):
        if size > 1:
            span += (size - 1) * stride
    start = t.data_ptr()
    return start, start + (span + 1) * t.element_size()


def _ranges_intersect(out, t):
    o_start, o_end = _byte_range(out)
    t_start, t_end = _byte_range(t)
    return t_start < o_end and o_start < t_end


def _suggest_memory_format(t):
    """Mirror ATen's TensorImpl::suggest_memory_format for our rank limits."""
    nd = t.ndim
    if nd == 4 and t.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    if nd == 5 and t.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    return torch.contiguous_format


def _format_strides(shape, fmt):
    """Strides ATen's allocator uses for ``shape`` in the derived format."""
    if fmt is torch.channels_last and len(shape) == 4:
        _, c, h, w = shape
        return (c * h * w, 1, w * c, c)
    if fmt is torch.channels_last_3d and len(shape) == 5:
        _, c, d, h, w = shape
        return (c * d * h * w, 1, h * w * c, w * c, c)
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


def _derived_memory_format(tensors):
    """ATen's cat_compute_output_memory_format: every input must agree."""
    fmt = None
    for t in tensors:
        cur = _suggest_memory_format(t)
        if cur is torch.contiguous_format:
            return torch.contiguous_format
        if fmt is not None and fmt is not cur:
            return torch.contiguous_format
        fmt = cur
    return fmt if fmt is not None else torch.contiguous_format


def row_stack_out(tensors=None, *args, out=None, **kwargs):
    logger.debug("GEMS_ILUVATAR ROW_STACK_OUT")
    if tensors is None:
        tensors = kwargs.get("tensors")
    if out is None and args:
        out = args[0]
    if out is None:
        raise TypeError("row_stack_out() missing required keyword argument: 'out'")

    if len(tensors) == 0:
        raise RuntimeError("row_stack expected a non-empty TensorList")

    odtype = out.dtype
    odev = out.get_device()

    # ---- pass 1: validation, shape normalization, prefix row accounting ----
    info = []
    total_rows = 0
    tail0 = None
    cast_needed = False
    for t in tensors:
        if t.get_device() != odev:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got tensors is on "
                f"{t.device}, different from other tensors on {out.device}"
            )
        if _overlaps(out, t) and _ranges_intersect(out, t):
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and the "
                "written-to tensor refer to a single memory location. Please clone() "
                "the tensor before performing the operation."
            )
        if t.dtype is not odtype:
            cast_needed = True
        nd = t.ndim
        numel = t.numel()
        if nd == 0:
            rows = 1
            tail = (1,)
        elif nd == 1:
            rows = 1
            tail = (numel,)
        else:
            sh = t.shape
            rows = sh[0]
            tail = tuple(sh[1:])
        if tail0 is None:
            tail0 = tail
        elif tail != tail0:
            raise RuntimeError(
                "Sizes of tensors must match except in dimension 0. Expected size "
                f"{tail0} but got size {tail} for tensor number {len(info)} in the list."
            )
        info.append((t, rows, numel))
        total_rows += rows

    if cast_needed:
        for t in tensors:
            if not torch.can_cast(t.dtype, odtype):
                raise TypeError(
                    "torch.cat(): input types can't be cast to the desired output type "
                    f"{odtype}"
                )

    # ---- reference semantics: out is resized in place and returned ----
    target = (total_rows,) + tail0
    if out.shape != target:
        out.resize_(target)
        fmt = _derived_memory_format(tensors)
        if fmt is not torch.contiguous_format:
            # ATen re-imposes the format derived from the inputs when resizing.
            out.as_strided_(target, _format_strides(target, fmt))

    dptr = out.view(torch.uint8) if odtype is torch.bool else out
    out_contig = out.is_contiguous()
    esize = out.element_size()
    k = len(info)
    out_aligned = (dptr.data_ptr() & 15) == 0

    # ---- fast path: one launch copies every input ----
    if out_contig and 0 < k <= 4 and odtype is not torch.bool and out_aligned:
        dt0 = info[0][0].dtype
        srcs = []
        bases = []
        numels = []
        base = 0
        fuse = True
        for t, rows, numel in info:
            if (
                numel == 0
                or t.dtype is not dt0
                or not t.is_contiguous()
                or (t.data_ptr() & 15)
                or ((base * esize) & 15)
            ):
                fuse = False
                break
            srcs.append(t)
            bases.append(base)
            numels.append(numel)
            base += numel
        if fuse:
            first = numels[0]
            if first % _FLAT_BLOCK == 0 and all(x == first for x in numels):
                while len(srcs) < 4:
                    srcs.append(srcs[0])
                _multi_copy_eq_kernel[(first // _FLAT_BLOCK, k)](
                    dptr,
                    srcs[0],
                    srcs[1],
                    srcs[2],
                    srcs[3],
                    first,
                    BLOCK=_FLAT_BLOCK,
                    K=k,
                    num_warps=4,
                )
                return out
            even = first % _FLAT_BLOCK == 0 and all(n == first for n in numels)
            if even:
                n_tiles = first // _FLAT_BLOCK
            else:
                n_tiles = max((n + _FLAT_BLOCK - 1) // _FLAT_BLOCK for n in numels)
            while len(srcs) < 4:
                srcs.append(srcs[0])
                bases.append(bases[0])
                numels.append(numels[0])
            _multi_copy_vec_kernel[(n_tiles, k)](
                dptr,
                srcs[0],
                srcs[1],
                srcs[2],
                srcs[3],
                bases[0],
                bases[1],
                bases[2],
                bases[3],
                numels[0],
                numels[1],
                numels[2],
                numels[3],
                BLOCK=_FLAT_BLOCK,
                EVEN=even,
                num_warps=4,
            )
            return out

    base = 0
    row_off = 0
    for t, rows, numel in info:
        if numel:
            src = t if t.dtype is not torch.bool else t.view(torch.uint8)
            if out_contig and src.is_contiguous():
                if (
                    out_aligned
                    and ((base * esize) & 15) == 0
                    and (src.data_ptr() & 15) == 0
                ):
                    _flat_copy_vec_kernel[((numel + _FLAT_BLOCK - 1) // _FLAT_BLOCK,)](
                        dptr,
                        src,
                        base,
                        numel,
                        BLOCK=_FLAT_BLOCK,
                        EVEN=(numel % _FLAT_BLOCK) == 0,
                        num_warps=4,
                    )
                else:
                    _flat_copy_kernel[((numel + _FLAT_BLOCK - 1) // _FLAT_BLOCK,)](
                        dptr, src, base, numel, BLOCK=_FLAT_BLOCK, num_warps=4
                    )
            else:
                _launch_strided(dptr, src, row_off, rows, numel, out_contig, out)
        base += numel
        row_off += rows
    return out


def _launch_strided(dptr, src, row_off, rows, numel, dst_contig, out):
    """Cold path: arbitrary strides for the source (and optionally destination)."""
    nd = src.ndim
    if nd >= 3:
        dims = [rows] + list(src.shape[1:])
        sstr = list(src.stride())
    elif nd == 2:
        dims = [rows, src.shape[1]]
        sstr = [src.stride(0), src.stride(1)]
    elif nd == 1:
        dims = [1, numel]
        sstr = [0, src.stride(0)]
    else:
        dims = [1, 1]
        sstr = [0, 0]
    if len(dims) - 1 > _MAX_TRAIL:
        raise NotImplementedError("row_stack_out supports at most 5 dims")
    dims = dims + [1] * (_MAX_TRAIL + 1 - len(dims))
    sstr = sstr + [0] * (_MAX_TRAIL + 1 - len(sstr))

    if dst_contig:
        dstr = [1] * (_MAX_TRAIL + 1)
        inner = 1
        for i in range(_MAX_TRAIL, 0, -1):
            dstr[i] = inner
            inner *= dims[i]
        dstr[0] = inner
    else:
        ostr = list(out.stride())
        ostr = ostr + [0] * (_MAX_TRAIL + 1 - len(ostr))
        dstr = ostr[: _MAX_TRAIL + 1]

    _strided_copy_kernel[((numel + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK,)](
        dptr,
        src,
        row_off,
        numel,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        sstr[0],
        sstr[1],
        sstr[2],
        sstr[3],
        sstr[4],
        dstr[0],
        dstr[1],
        dstr[2],
        dstr[3],
        dstr[4],
        BLOCK=_STRIDED_BLOCK,
        num_warps=4,
    )
