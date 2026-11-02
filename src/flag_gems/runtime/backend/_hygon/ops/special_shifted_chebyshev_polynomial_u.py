import torch
import triton
import triton.language as tl

_MAX_ND = 8
_BLOCK = 2048
_NUM_WARPS = 8


@triton.jit
def _cheb_u_body(x, n):
    """U*_n(x) = U_n(2x - 1), elementwise, dynamic masked recurrence.

    U*_0 = 1, U*_1 = 4x - 2, U*_n = (4x - 2) * U*_{n-1} - U*_{n-2}; n < 0 -> 0.
    x is fp32 compute; n is int32 (already truncated toward zero).
    """
    a = 4.0 * x - 2.0
    nmax = tl.max(n, axis=0)
    npos = tl.maximum(n, 0)
    u1 = tl.where(n == 0, 1.0, a)
    u0 = u1 * 0.0 + 1.0
    for i in range(2, nmax + 1):
        nu = a * u1 - u0
        u0 = u1
        u1 = tl.where(i <= npos, nu, u1)
    return tl.where(n < 0, 0.0, u1)


@triton.jit
def _shifted_cheb_u_direct(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    X_IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    n = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
    if X_IS_FP64:
        xc = x
    else:
        xc = x.to(tl.float32)
    r = _cheb_u_body(xc, n)
    if X_IS_FP64:
        tl.store(out_ptr + offs, r, mask=mask)
    else:
        tl.store(out_ptr + offs, r.to(x.dtype), mask=mask, cache_modifier=".cs")


@triton.jit
def _shifted_cheb_u_bcast(
    x_ptr,
    n_ptr,
    out_ptr,
    dims_ptr,
    sx_ptr,
    sn_ptr,
    sp_ptr,
    numel,
    X_IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ar = tl.arange(0, 8)
    dims = tl.load(dims_ptr + ar)
    sx = tl.load(sx_ptr + ar)
    sn = tl.load(sn_ptr + ar)
    sp = tl.load(sp_ptr + ar)
    coord = (offs[:, None] // sp[None, :]) % dims[None, :]
    xo = tl.sum(coord * sx[None, :], axis=1)
    no = tl.sum(coord * sn[None, :], axis=1)
    x = tl.load(x_ptr + xo, mask=mask, other=0.0)
    n = tl.load(n_ptr + no, mask=mask, other=0).to(tl.int32)
    if X_IS_FP64:
        xc = x
    else:
        xc = x.to(tl.float32)
    r = _cheb_u_body(xc, n)
    if X_IS_FP64:
        tl.store(out_ptr + offs, r, mask=mask)
    else:
        tl.store(out_ptr + offs, r.to(x.dtype), mask=mask, cache_modifier=".cs")


def _bcast_strides(shape, t):
    nd = len(shape)
    tshape = t.shape
    tstride = t.stride()
    pad = nd - len(tshape)
    res = [0] * nd
    for d in range(nd):
        td = d - pad
        if td < 0:
            continue
        if tshape[td] == shape[d]:
            res[d] = int(tstride[td])
        else:
            res[d] = 0
    return res


def run(x, n):
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, dtype=torch.int64, device=x.device)
    xshape = x.shape
    nshape = n.shape
    if xshape == nshape:
        shape = xshape
        numel = x.numel()
        simple = x.is_contiguous() and n.is_contiguous()
    else:
        shape = torch.broadcast_shapes(xshape, nshape)
        numel = 1
        for s in shape:
            numel *= int(s)
        simple = False
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    if numel == 0:
        return out
    if simple:
        grid = (triton.cdiv(numel, _BLOCK),)
        _shifted_cheb_u_direct[grid](
            x,
            n,
            out,
            numel,
            X_IS_FP64=x.dtype == torch.float64,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    else:
        nd = len(shape)
        if nd > _MAX_ND:
            raise ValueError(f"more than {_MAX_ND} dims not supported")
        sx = _bcast_strides(shape, x)
        sn = _bcast_strides(shape, n)
        dims = [1] * _MAX_ND
        for d in range(nd):
            dims[d] = int(shape[d])
        sp = [1] * _MAX_ND
        acc = 1
        for d in range(_MAX_ND - 1, -1, -1):
            sp[d] = acc
            acc *= dims[d]
        dev = x.device
        dims_t = torch.tensor(dims, dtype=torch.int32, device=dev)
        sx_t = torch.tensor(sx, dtype=torch.int32, device=dev)
        sn_t = torch.tensor(sn, dtype=torch.int32, device=dev)
        sp_t = torch.tensor(sp, dtype=torch.int32, device=dev)
        grid = (triton.cdiv(numel, _BLOCK),)
        _shifted_cheb_u_bcast[grid](
            x,
            n,
            out,
            dims_t,
            sx_t,
            sn_t,
            sp_t,
            numel,
            X_IS_FP64=x.dtype == torch.float64,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out


# Alias for FlagGems import convention
special_shifted_chebyshev_polynomial_u = run
