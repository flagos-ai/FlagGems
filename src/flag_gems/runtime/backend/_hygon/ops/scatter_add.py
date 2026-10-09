import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# scatter_add(inp, dim, index, src) -> out
#
# Reference semantics: out = inp.clone(); out.scatter_add_(dim, index, src)
#   i.e. out is a copy of inp, then for every position p of src:
#        out[..., index[p], ...] += src[p]   (along axis `dim`)
#
# Strategy:
#   kernel 1: elementwise copy inp -> out (general strides, so any layout works)
#   kernel 2: per src/index element, atomic_add(src_value) into the output
#             position selected by index along `dim`.
#
# Two scatter paths:
#   * fast path (_scatter2d_kernel): used when `dim` is the LAST axis and both
#     index/src have unit stride along it.  A 2D grid (j-chunk x row) makes the
#     per-element body just two coalesced loads + one atomic: no divmod chain.
#     When J is an exact multiple of BLOCK the bounds mask is dropped and the
#     atomics use relaxed semantics (no end-of-kernel waitcnt/wbinvl).
#   * general path (_scatter_kernel): any other layout.  Iterates in natural
#     row-major flat order of `index` (coalesced), decomposes the flat index
#     into per-dim coordinates via a constexpr divmod chain and accumulates
#     real tensor strides.
# ---------------------------------------------------------------------------

_COPY_BLOCK = 1024
_COPY_WARPS = 4
_SCATTER_BLOCK = 1024
_SCATTER_WARPS = 4


@triton.jit
def _copy_kernel(
    inp_ptr,
    out_ptr,
    numel,
    NDIM: tl.constexpr,
    TAILS: tl.constexpr,
    ISTR: tl.constexpr,
    OSTR: tl.constexpr,
    CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    f = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = f < numel
    if CONTIG:
        ioff = f
        ooff = f
    else:
        rem = f
        ioff = tl.zeros([BLOCK], dtype=tl.int64)
        ooff = tl.zeros([BLOCK], dtype=tl.int64)
        for k in tl.static_range(NDIM):
            c = rem // TAILS[k]
            rem -= c * TAILS[k]
            ioff += c * ISTR[k]
            ooff += c * OSTR[k]
    val = tl.load(inp_ptr + ioff, mask=mask)
    tl.store(out_ptr + ooff, val, mask=mask)


@triton.jit
def _scatter2d_kernel(
    out_ptr,
    idx_ptr,
    src_ptr,
    J,
    NDIM: tl.constexpr,
    REST_TAILS: tl.constexpr,
    IX_RSTR: tl.constexpr,
    SR_RSTR: tl.constexpr,
    OU_RSTR: tl.constexpr,
    IX_STR: tl.constexpr,
    SR_STR: tl.constexpr,
    OU_STR: tl.constexpr,
    BLOCK: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1).to(tl.int64)
    rem = pid_y
    bix = 0
    bsr = 0
    bou = 0
    for k in tl.static_range(NDIM - 1):
        c = rem // REST_TAILS[k]
        rem -= c * REST_TAILS[k]
        bix += c * IX_RSTR[k]
        bsr += c * SR_RSTR[k]
        bou += c * OU_RSTR[k]
    jj = pid_x.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if NO_MASK:
        idx = tl.load(idx_ptr + bix + jj * IX_STR)
        val = tl.load(src_ptr + bsr + jj * SR_STR)
        ooff = bou + idx.to(tl.int64) * OU_STR
        tl.atomic_add(out_ptr + ooff, val, sem="relaxed")
    else:
        mask = jj < J
        idx = tl.load(idx_ptr + bix + jj * IX_STR, mask=mask)
        val = tl.load(src_ptr + bsr + jj * SR_STR, mask=mask)
        ooff = bou + idx.to(tl.int64) * OU_STR
        tl.atomic_add(out_ptr + ooff, val, mask=mask)


@triton.jit
def _scatter_kernel(
    out_ptr,
    idx_ptr,
    src_ptr,
    numel,
    NDIM: tl.constexpr,
    DIM: tl.constexpr,
    TAILS: tl.constexpr,
    IXSTR: tl.constexpr,
    SRCSTR: tl.constexpr,
    OUTSTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    f = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = f < numel
    rem = f
    idx_off = tl.zeros([BLOCK], dtype=tl.int64)
    soff = tl.zeros([BLOCK], dtype=tl.int64)
    ooff = tl.zeros([BLOCK], dtype=tl.int64)
    for k in tl.static_range(NDIM):
        c = rem // TAILS[k]
        rem -= c * TAILS[k]
        idx_off += c * IXSTR[k]
        soff += c * SRCSTR[k]
        if k != DIM:
            ooff += c * OUTSTR[k]
    idx = tl.load(idx_ptr + idx_off, mask=mask)
    val = tl.load(src_ptr + soff, mask=mask)
    ooff += idx.to(tl.int64) * OUTSTR[DIM]
    tl.atomic_add(out_ptr + ooff, val, mask=mask)


def _tails(shape):
    """prod(shape[k+1:]) for every k, i.e. the size-stride used for divmod."""
    n = len(shape)
    t = [1] * n
    acc = 1
    for k in range(n - 1, -1, -1):
        t[k] = acc
        acc *= shape[k]
    return tuple(t)


def _fast_config(j):
    """Block/warp choice for the 2D-grid scatter path."""
    b = 128
    while b * 2 <= j and b < 4096:
        b *= 2
    w = max(1, min(16, b // 128))
    if b == 4096:
        # Debug sweeps: 16 warps helps the mid-size (J=8192, ~33M elem)
        # workloads; 8 warps is best for the huge rows (J=131072).
        w = 16 if j <= 16384 else 8
    return b, w


def run(inp, dim, index, src):
    ndim = inp.dim()

    if isinstance(dim, torch.Tensor):
        d = int(dim.item())
    else:
        d = int(dim)
    if d < 0:
        d += ndim

    # 0-dim inputs: view as 1-element 1-D tensors (zero-copy view).
    if ndim == 0:
        inp = inp.view(1)
        index = index.view(1)
        src = src.view(1)
        ndim = 1
        d = 0

    out = torch.empty_like(inp)

    numel_inp = inp.numel()
    numel_idx = index.numel()
    last = ndim - 1

    if numel_inp > 0:
        _copy_kernel[(triton.cdiv(numel_inp, _COPY_BLOCK),)](
            inp,
            out,
            numel_inp,
            NDIM=ndim,
            TAILS=_tails(tuple(inp.shape)),
            ISTR=tuple(inp.stride()),
            OSTR=tuple(out.stride()),
            CONTIG=inp.is_contiguous() and out.is_contiguous(),
            BLOCK=_COPY_BLOCK,
            num_warps=_COPY_WARPS,
        )

    if numel_idx > 0 and d == last and index.stride(d) == 1 and src.stride(d) == 1:
        J = index.size(d)
        rest = [k for k in range(ndim) if k != d]
        b, w = _fast_config(J)
        _scatter2d_kernel[(triton.cdiv(J, b), numel_idx // J)](
            out,
            index,
            src,
            J,
            NDIM=ndim,
            REST_TAILS=_tails(tuple(index.size(k) for k in rest)),
            IX_RSTR=tuple(index.stride(k) for k in rest),
            SR_RSTR=tuple(src.stride(k) for k in rest),
            OU_RSTR=tuple(out.stride(k) for k in rest),
            IX_STR=index.stride(d),
            SR_STR=src.stride(d),
            OU_STR=out.stride(d),
            BLOCK=b,
            NO_MASK=(J % b == 0),
            num_warps=w,
        )
    elif numel_idx > 0:
        _scatter_kernel[(triton.cdiv(numel_idx, _SCATTER_BLOCK),)](
            out,
            index,
            src,
            numel_idx,
            NDIM=ndim,
            DIM=d,
            TAILS=_tails(tuple(index.shape)),
            IXSTR=tuple(index.stride()),
            SRCSTR=tuple(src.stride()),
            OUTSTR=tuple(out.stride()),
            BLOCK=_SCATTER_BLOCK,
            num_warps=_SCATTER_WARPS,
        )

    return out


# Alias for FlagGems import convention
scatter_add = run
