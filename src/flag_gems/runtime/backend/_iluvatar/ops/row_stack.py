"""row_stack (alias of vstack) implemented as a Triton multi-source flat copy.

Semantics reproduced (torch.vstack / torch.row_stack):
  * every 1-D input of shape (N,) becomes one row (1, N);
  * inputs are concatenated along dim 0, trailing shape must match;
  * the output dtype is the promotion of all input dtypes;
  * the output *memory format* mirrors ATen's cat_compute_output_memory_format:
    channels_last / channels_last_3d when every input is contiguous in that
    format, plain contiguous otherwise.  Dim 0 is the outermost dim in all
    three layouts, so element offsets are simply cumulative numels whether the
    destination is contiguous or channels_last.

Two single-launch kernels:

  _stack_copy_even_kernel (fast path) is used when every input element count is
  a multiple of BLOCK.  Programs are then segment aligned - the first nb0
  programs belong to input 0, the next nb1 to input 1, and so on - so each
  program resolves its slot with a few program-uniform comparisons and executes
  exactly one mask-free, contiguous load/store pair.  Destination offsets are
  host-computed scalars, which keeps their 16-byte divisibility visible to the
  compiler, so the accesses vectorize.

  _stack_copy_kernel (fallback) keeps the masked multi-slot form for ragged
  element counts and for more than _MAX_SLOTS inputs (chunked launches).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_SLOTS = 4
_BLOCK = 1024


@triton.jit
def _stack_copy_even_kernel(
    p0,
    p1,
    p2,
    p3,
    out,
    o0,
    o1,
    o2,
    o3,
    e0,
    e1,
    e2,
    BLOCK: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    if pid < e0:
        off = pid * BLOCK + idx
        tl.store(out + o0 + off, tl.load(p0 + off))
    elif pid < e0 + e1:
        off = (pid - e0) * BLOCK + idx
        tl.store(out + o1 + off, tl.load(p1 + off))
    elif pid < e0 + e1 + e2:
        off = (pid - e0 - e1) * BLOCK + idx
        tl.store(out + o2 + off, tl.load(p2 + off))
    else:
        off = (pid - e0 - e1 - e2) * BLOCK + idx
        tl.store(out + o3 + off, tl.load(p3 + off))


@triton.jit
def _stack_copy_kernel(
    p0,
    p1,
    p2,
    p3,
    out,
    dst_base,
    n0,
    n1,
    n2,
    total,
    BLOCK: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    dst = out + dst_base

    # slot 0: destination window [0, n0)
    m0 = offs < n0
    tl.store(dst + offs, tl.load(p0 + offs, mask=m0, other=0), mask=m0)

    # slot 1: destination window [n0, n0 + n1)
    o1 = n0
    m1 = (offs >= o1) & (offs < o1 + n1)
    tl.store(dst + offs, tl.load(p1 + (offs - o1), mask=m1, other=0), mask=m1)

    # slot 2: destination window [n0 + n1, + n2)
    o2 = o1 + n1
    m2 = (offs >= o2) & (offs < o2 + n2)
    tl.store(dst + offs, tl.load(p2 + (offs - o2), mask=m2, other=0), mask=m2)

    # slot 3: destination window [n0 + n1 + n2, total)
    o3 = o2 + n2
    m3 = (offs >= o3) & (offs < total)
    tl.store(dst + offs, tl.load(p3 + (offs - o3), mask=m3, other=0), mask=m3)


def _trailing(t):
    """Trailing (non-concatenated) shape of one input."""
    d = t.dim()
    return t.shape[1:] if d >= 2 else t.shape


def _out_memory_format(ts):
    """Mirror ATen's cat_compute_output_memory_format for the output tensor.

    This torch build has no Tensor.suggest_memory_format, so the format is
    derived from contiguity: a plain contiguous input forces contiguous (which
    matches ATen, whose suggest_memory_format returns Contiguous there), and
    channels_last / channels_last_3d is used only when every input is
    contiguous in that format and not plain contiguous.
    """
    d = ts[0].dim()
    if d == 4:
        fmt = torch.channels_last
    elif d == 5:
        fmt = torch.channels_last_3d
    else:
        return torch.contiguous_format
    for t in ts:
        if t.is_contiguous() or not t.is_contiguous(memory_format=fmt):
            return torch.contiguous_format
    return fmt


def row_stack(tensors, out=None):
    logger.debug("GEMS_ILUVATAR ROW_STACK")
    ts = tuple(tensors)
    k = len(ts)
    if k == 0:
        raise RuntimeError("row_stack expects a non-empty list of Tensors")

    t0 = ts[0]
    if not isinstance(t0, torch.Tensor):
        raise RuntimeError("row_stack expects a list of Tensors")
    if t0.dim() == 0:
        raise RuntimeError("zero-dimensional tensor cannot be concatenated")

    device = t0.device
    dtype = t0.dtype
    trailing = _trailing(t0)

    rows = 1 if t0.dim() == 1 else t0.shape[0]
    numels = [t0.numel()]
    for i in range(1, k):
        t = ts[i]
        if not isinstance(t, torch.Tensor):
            raise RuntimeError("row_stack expects a list of Tensors")
        if t.dim() == 0:
            raise RuntimeError("zero-dimensional tensor cannot be concatenated")
        if t.device != device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least "
                "two devices"
            )
        if _trailing(t) != trailing:
            raise RuntimeError(
                "Sizes of tensors must match except in dimension 0. "
                "Got {} and {} in dimension 1".format(
                    tuple(trailing), tuple(_trailing(t))
                )
            )
        dtype = torch.promote_types(dtype, t.dtype)
        rows += 1 if t.dim() == 1 else t.shape[0]
        numels.append(t.numel())

    shape = (rows,) + tuple(trailing)
    if out is None:
        fmt = _out_memory_format(ts)
        if fmt == torch.contiguous_format:
            out = torch.empty(shape, dtype=dtype, device=device)
        else:
            out = torch.empty(shape, dtype=dtype, device=device, memory_format=fmt)
    elif tuple(out.shape) != tuple(shape):
        raise RuntimeError(
            "row_stack: out tensor has shape {}, expected {}".format(
                tuple(out.shape), tuple(shape)
            )
        )

    total = 0
    for n in numels:
        total += n
    if total == 0:
        return out

    # fast path: every input is a whole number of blocks, so programs can be
    # segment aligned and each program copies one mask-free contiguous run
    if k <= _MAX_SLOTS:
        even = True
        for n in numels:
            if n % _BLOCK:
                even = False
                break
        if even:
            ne = [0, 0, 0, 0]
            off = [0, 0, 0, 0]
            acc = 0
            for i in range(k):
                ne[i] = numels[i] // _BLOCK
                off[i] = acc
                acc += numels[i]
            _stack_copy_even_kernel[(ne[0] + ne[1] + ne[2] + ne[3],)](
                ts[0],
                ts[1] if k > 1 else out,
                ts[2] if k > 2 else out,
                ts[3] if k > 3 else out,
                out,
                off[0],
                off[1],
                off[2],
                off[3],
                ne[0],
                ne[1],
                ne[2],
            )
            return out

    # fallback: masked multi-slot kernel, chunked over _MAX_SLOTS sources
    i = 0
    base = 0
    while i < k:
        counts = numels[i : i + _MAX_SLOTS]
        c = len(counts)
        ctotal = 0
        for n in counts:
            ctotal += n
        p0 = ts[i]
        p1 = ts[i + 1] if c > 1 else out
        p2 = ts[i + 2] if c > 2 else out
        p3 = ts[i + 3] if c > 3 else out
        n0 = counts[0]
        n1 = counts[1] if c > 1 else 0
        n2 = counts[2] if c > 2 else 0
        if ctotal > 0:
            _stack_copy_kernel[((ctotal + _BLOCK - 1) // _BLOCK,)](
                p0, p1, p2, p3, out, base, n0, n1, n2, ctotal, BLOCK=_BLOCK
            )
        base += ctotal
        i += _MAX_SLOTS

    return out
