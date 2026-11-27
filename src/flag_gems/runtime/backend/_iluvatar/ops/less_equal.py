"""Element-wise less-than-or-equal comparison (torch.less_equal) in Triton."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_BLOCK = 1024
_SMALL_N = 1 << 16

# Bounded fixed-source search over elements-per-thread x warps.  A 64-lane warp
# means threads = 64 * num_warps, so BLOCK/threads is elements per lane.
_TUNE_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=8),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
]


@triton.jit
def _le_flat_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    b = tl.load(b_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, (a <= b).to(tl.uint8), mask=mask)


_le_flat_tuned = triton.autotune(configs=_TUNE_CONFIGS, key=["n"])(_le_flat_kernel)


@triton.jit
def _le_strided_kernel(
    a_ptr, b_ptr, out_ptr, n, shp, ast, bst, nd, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs.to(tl.int64)
    aoff = tl.zeros([BLOCK], dtype=tl.int64)
    boff = tl.zeros([BLOCK], dtype=tl.int64)
    for d in range(0, nd):
        sz = tl.load(shp + d).to(tl.int64)
        idx = rem % sz
        rem = rem // sz
        aoff = aoff + idx * tl.load(ast + d).to(tl.int64)
        boff = boff + idx * tl.load(bst + d).to(tl.int64)
    a = tl.load(a_ptr + aoff, mask=mask, other=0)
    b = tl.load(b_ptr + boff, mask=mask, other=0)
    tl.store(out_ptr + offs, (a <= b).to(tl.uint8), mask=mask)


def _as_u8(t):
    """Zero-copy byte reinterpretation of bool tensors for the kernel."""
    if t.dtype is torch.bool:
        try:
            return t.view(torch.uint8)
        except Exception:
            return t
    return t


def _scalar_tensor(v, other):
    """Materialize a python scalar operand with torch's wrapped-number dtype."""
    try:
        dt = torch.result_type(other, v)
    except Exception:
        dt = None
    t = torch.as_tensor(v, device=other.device if torch.is_tensor(other) else None)
    if dt is not None and t.dtype != dt:
        try:
            t = t.to(dt)
        except Exception:
            pass
    return t


def _collapse(shape, astr, bstr):
    """Merge/drop dims; returns innermost-first list of (size, astride, bstride)."""
    dims = []
    for i in range(len(shape) - 1, -1, -1):
        size = shape[i]
        if size == 1:
            continue
        a = astr[i]
        b = bstr[i]
        if dims:
            psize, pa, pb = dims[-1]
            if a == pa * psize and b == pb * psize:
                dims[-1] = (psize * size, pa, pb)
                continue
        dims.append((size, a, b))
    return dims


def less_equal(A, B):
    logger.debug("GEMS_ILUVATAR LESS_EQUAL")
    if not torch.is_tensor(A):
        A = _scalar_tensor(A, B)
    if not torch.is_tensor(B):
        B = _scalar_tensor(B, A)

    shape = torch.broadcast_shapes(A.shape, B.shape)
    out = torch.empty(shape, dtype=torch.bool, device=A.device)
    n = out.numel()
    if n == 0:
        return out

    a8 = _as_u8(A)
    b8 = _as_u8(B)
    out8 = out.view(torch.uint8)

    if A.shape == B.shape and A.is_contiguous() and B.is_contiguous():
        if n <= _SMALL_N:
            grid = (triton.cdiv(n, _BLOCK),)
            _le_flat_kernel[grid](a8, b8, out8, n, BLOCK=_BLOCK, num_warps=4)
        else:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
            _le_flat_tuned[grid](a8, b8, out8, n)
        return out

    ae = a8.expand(shape)
    be = b8.expand(shape)
    dims = _collapse(shape, ae.stride(), be.stride())
    if not dims:
        dims = [(1, 0, 0)]
    nd = len(dims)
    dev = A.device
    shp = torch.tensor([d[0] for d in dims], dtype=torch.int32, device=dev)
    ast = torch.tensor([d[1] for d in dims], dtype=torch.int32, device=dev)
    bst = torch.tensor([d[2] for d in dims], dtype=torch.int32, device=dev)
    grid = (triton.cdiv(n, _BLOCK),)
    _le_strided_kernel[grid](
        ae, be, out8, n, shp, ast, bst, nd, BLOCK=_BLOCK, num_warps=4
    )
    return out
