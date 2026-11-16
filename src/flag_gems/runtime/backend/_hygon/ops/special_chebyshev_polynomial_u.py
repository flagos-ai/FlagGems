import torch
import triton
import triton.language as tl

# special_chebyshev_polynomial_u: out = U_n(x) (Chebyshev polynomial of the
# second kind), matching the FlagGems/torch test suite semantics:
#   * scalar n must satisfy 0 <= int(n) <= 5, otherwise ValueError
#   * tensor n (integer dtype, truncated) values must be in [0, 5],
#     otherwise ValueError
#   * explicit polynomial formulas for n = 0..5:
#       U_0 = 1, U_1 = 2x, U_2 = 4x^2-1, U_3 = 8x^3-4x,
#       U_4 = 16x^4-12x^2+1, U_5 = 32x^5-32x^3+6x
#   * non-floating x is computed in float32 and returned as float32
#   * output shape follows x (tensor n path is elementwise on equal shapes;
#     a general broadcast path is kept for robustness)

_BLOCK = 2048
_NUM_WARPS = 8


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _cheb_u_scalar_kernel(
    x_ptr, out_ptr, numel, CAST_F32: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(
        x_ptr + offs, mask=mask, cache_modifier=".cg", eviction_policy="evict_first"
    )
    if CAST_F32:
        x = x.to(tl.float32)
    if N == 0:
        r = x - x + 1.0
    elif N == 1:
        r = x + x
    elif N == 2:
        x2 = x * x
        r = 4.0 * x2 - 1.0
    elif N == 3:
        x2 = x * x
        x3 = x2 * x
        r = 8.0 * x3 - 4.0 * x
    elif N == 4:
        x2 = x * x
        x4 = x2 * x2
        r = 16.0 * x4 - 12.0 * x2 + 1.0
    else:  # N == 5
        x2 = x * x
        x3 = x2 * x
        x5 = x3 * x2
        r = 32.0 * x5 - 32.0 * x3 + 6.0 * x
    tl.store(out_ptr + offs, r, mask=mask, cache_modifier=".cs")


@triton.jit
def _cheb_u_minmax_kernel(n_ptr, max_ptr, min_ptr, n_numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_numel
    n = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
    m = tl.max(n, axis=0)
    mn = tl.min(n, axis=0)
    tl.atomic_max(max_ptr, m)
    tl.atomic_min(min_ptr, mn)


@triton.jit
def _cheb_u_tensor_kernel(
    x_ptr, n_ptr, out_ptr, numel, CAST_F32: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    if CAST_F32:
        x = x.to(tl.float32)
    n = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    u0 = x - x + 1.0
    u1 = x + x
    u2 = 4.0 * x2 - 1.0
    u3 = 8.0 * x3 - 4.0 * x
    u4 = 16.0 * x4 - 12.0 * x2 + 1.0
    u5 = 32.0 * x5 - 32.0 * x3 + 6.0 * x
    r = tl.where(
        n == 0,
        u0,
        tl.where(
            n == 1,
            u1,
            tl.where(n == 2, u2, tl.where(n == 3, u3, tl.where(n == 4, u4, u5))),
        ),
    )
    tl.store(out_ptr + offs, r, mask=mask)


@triton.jit
def _cheb_u_bcast_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    xb0,
    xb1,
    xb2,
    xb3,
    xb4,
    xb5,
    xb6,
    xb7,
    nb0,
    nb1,
    nb2,
    nb3,
    nb4,
    nb5,
    nb6,
    nb7,
    CAST_F32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs
    x_idx = tl.zeros_like(offs)
    n_idx = tl.zeros_like(offs)
    # mixed-radix decomposition, innermost dim (s7) first
    c = rem % s7
    rem = rem // s7
    x_idx += c * xb7
    n_idx += c * nb7
    c = rem % s6
    rem = rem // s6
    x_idx += c * xb6
    n_idx += c * nb6
    c = rem % s5
    rem = rem // s5
    x_idx += c * xb5
    n_idx += c * nb5
    c = rem % s4
    rem = rem // s4
    x_idx += c * xb4
    n_idx += c * nb4
    c = rem % s3
    rem = rem // s3
    x_idx += c * xb3
    n_idx += c * nb3
    c = rem % s2
    rem = rem // s2
    x_idx += c * xb2
    n_idx += c * nb2
    c = rem % s1
    rem = rem // s1
    x_idx += c * xb1
    n_idx += c * nb1
    c = rem % s0
    rem = rem // s0
    x_idx += c * xb0
    n_idx += c * nb0
    x = tl.load(x_ptr + x_idx, mask=mask)
    if CAST_F32:
        x = x.to(tl.float32)
    n = tl.load(n_ptr + n_idx, mask=mask, other=0).to(tl.int32)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    u0 = x - x + 1.0
    u1 = x + x
    u2 = 4.0 * x2 - 1.0
    u3 = 8.0 * x3 - 4.0 * x
    u4 = 16.0 * x4 - 12.0 * x2 + 1.0
    u5 = 32.0 * x5 - 32.0 * x3 + 6.0 * x
    r = tl.where(
        n == 0,
        u0,
        tl.where(
            n == 1,
            u1,
            tl.where(n == 2, u2, tl.where(n == 3, u3, tl.where(n == 4, u4, u5))),
        ),
    )
    tl.store(out_ptr + offs, r, mask=mask)


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------
def _broadcast_meta(sx, sn, x_stride, n_stride):
    nd = max(len(sx), len(sn))
    px = (1,) * (nd - len(sx)) + tuple(sx)
    pn = (1,) * (nd - len(sn)) + tuple(sn)
    out = []
    for a, b in zip(px, pn):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise ValueError("x and n are not broadcastable")
    out = tuple(out)
    xb = []
    for d in range(nd):
        j = d - (nd - len(sx))
        if j < 0 or px[d] != out[d]:
            xb.append(0)
        else:
            xb.append(x_stride[j])
    nb = []
    for d in range(nd):
        j = d - (nd - len(sn))
        if j < 0 or pn[d] != out[d]:
            nb.append(0)
        else:
            nb.append(n_stride[j])
    return out, tuple(xb), tuple(nb), nd


def _out_dtype(x):
    return x.dtype if x.dtype.is_floating_point else torch.float32


def run(x, n):
    if isinstance(n, (int, float)):
        nv = int(n)
        if nv < 0 or nv > 5:
            raise ValueError(f"n must be in [0, 5], got {n}")
        return _run_scalar(x, nv)
    if not isinstance(n, torch.Tensor):
        raise ValueError("n must be a scalar or a tensor")
    if n.device != x.device:
        n = n.to(x.device)
    if n.dim() == 0:
        # 0-dim tensor: treated as a scalar degree
        nv = int(n.item())
        if nv < 0 or nv > 5:
            raise ValueError(f"n must be in [0, 5], got {nv}")
        return _run_scalar(x, nv)
    return _run_tensor(x, n)


def _run_scalar(x, nv):
    numel = x.numel()
    out = torch.empty(x.shape, dtype=_out_dtype(x), device=x.device)
    if numel == 0:
        return out
    cast_f32 = not x.dtype.is_floating_point
    grid = (triton.cdiv(numel, _BLOCK),)
    _cheb_u_scalar_kernel[grid](
        x, out, numel, CAST_F32=cast_f32, N=nv, BLOCK=_BLOCK, num_warps=_NUM_WARPS
    )
    return out


def _run_tensor(x, n):
    # device-side min/max guard on the truncated n values
    max_buf = torch.full((1,), -(2**31), dtype=torch.int32, device=x.device)
    min_buf = torch.full((1,), 2**31 - 1, dtype=torch.int32, device=x.device)
    grid_m = (triton.cdiv(n.numel(), _BLOCK),)
    _cheb_u_minmax_kernel[grid_m](n, max_buf, min_buf, n.numel(), BLOCK=_BLOCK)
    maxv = int(max_buf.item())
    minv = int(min_buf.item())
    if maxv > 5 or minv < 0:
        raise ValueError(f"n values must be in [0, 5], got [{minv}, {maxv}]")

    sx = tuple(x.shape)
    sn = tuple(n.shape)
    if sx == sn:
        numel = x.numel()
        out = torch.empty(x.shape, dtype=_out_dtype(x), device=x.device)
        if numel == 0:
            return out
        cast_f32 = not x.dtype.is_floating_point
        grid = (triton.cdiv(numel, _BLOCK),)
        _cheb_u_tensor_kernel[grid](x, n, out, numel, CAST_F32=cast_f32, BLOCK=_BLOCK)
        return out

    out_shape, xb, nb, nd = _broadcast_meta(
        sx, sn, tuple(x.stride()), tuple(n.stride())
    )
    numel = 1
    for s in out_shape:
        numel *= s
    out = torch.empty(out_shape, dtype=_out_dtype(x), device=x.device)
    if numel == 0:
        return out
    s_pad = (1,) * (8 - nd) + out_shape
    xb_pad = (0,) * (8 - nd) + xb
    nb_pad = (0,) * (8 - nd) + nb
    cast_f32 = not x.dtype.is_floating_point
    grid = (triton.cdiv(numel, _BLOCK),)
    _cheb_u_bcast_kernel[grid](
        x,
        n,
        out,
        numel,
        s_pad[0],
        s_pad[1],
        s_pad[2],
        s_pad[3],
        s_pad[4],
        s_pad[5],
        s_pad[6],
        s_pad[7],
        xb_pad[0],
        xb_pad[1],
        xb_pad[2],
        xb_pad[3],
        xb_pad[4],
        xb_pad[5],
        xb_pad[6],
        xb_pad[7],
        nb_pad[0],
        nb_pad[1],
        nb_pad[2],
        nb_pad[3],
        nb_pad[4],
        nb_pad[5],
        nb_pad[6],
        nb_pad[7],
        CAST_F32=cast_f32,
        BLOCK=_BLOCK,
    )
    return out


# Alias for FlagGems import convention
special_chebyshev_polynomial_u = run
