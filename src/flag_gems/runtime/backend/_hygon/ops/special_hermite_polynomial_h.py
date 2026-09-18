import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# special_hermite_polynomial_h: physicist's Hermite polynomial H_n(x)
#   H_0(x) = 1, H_1(x) = 2x, H_k(x) = 2x*H_{k-1}(x) - 2*(k-1)*H_{k-2}(x)
# Contract (flaggems / torch.special.hermite_polynomial_h):
#   - x in {float32, float64}, n truncated to int, broadcast like torch
#   - n must be in [0, 9]: out-of-range n raises ValueError
#     ("only supports n in [0, 9]") for scalar / 1-element n
#   - output in promoted dtype (ints -> float32)
# The degree loop is statically unrolled to 9 (the validated upper bound).
# ---------------------------------------------------------------------------

_MAX_DEG = tl.constexpr(9)


@triton.jit
def _hermite_h_flat_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    x_stride,  # 0 if x is scalar-broadcast, 1 if elementwise
    n_stride,  # 0 if n is scalar-broadcast, 1 if elementwise
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs64 = offs.to(tl.int64)
    mask = offs < numel

    xv = tl.load(x_ptr + offs64 * x_stride, mask=mask, other=0).to(DTYPE)
    nv = tl.load(n_ptr + offs64 * n_stride, mask=mask, other=0).to(tl.int64)

    one = tl.full([BLOCK], 1.0, DTYPE)
    two_x = xv + xv
    zero = tl.zeros([BLOCK], DTYPE)
    res = tl.where(nv == 0, one, tl.where(nv == 1, two_x, zero))

    h_prev2 = one  # H_0
    h_prev = two_x  # H_1
    for k in tl.static_range(2, _MAX_DEG + 1):
        km1 = k - 1
        h_cur = (xv + xv) * h_prev - (km1 + km1) * h_prev2  # H_k
        res = tl.where(nv == k, h_cur, res)
        h_prev2 = h_prev
        h_prev = h_cur

    tl.store(out_ptr + offs64, res, mask=mask)


@triton.jit
def _hermite_h_nd_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    ND: tl.constexpr,
    OUT_SHAPE: tl.constexpr,
    XS: tl.constexpr,
    NS: tl.constexpr,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs64 = offs.to(tl.int64)
    mask = offs < numel

    # decompose flat index into broadcast offsets
    rem = offs64
    x_off = tl.zeros([BLOCK], tl.int64)
    n_off = tl.zeros([BLOCK], tl.int64)
    for i in tl.static_range(ND - 1, -1, -1):
        d = OUT_SHAPE[i]
        if d > 1:
            digit = rem % d
            rem = rem // d
        else:
            digit = tl.zeros([BLOCK], tl.int64)
        x_off += digit * XS[i]
        n_off += digit * NS[i]

    xv = tl.load(x_ptr + x_off, mask=mask, other=0).to(DTYPE)
    nv = tl.load(n_ptr + n_off, mask=mask, other=0).to(tl.int64)

    one = tl.full([BLOCK], 1.0, DTYPE)
    two_x = xv + xv
    zero = tl.zeros([BLOCK], DTYPE)
    res = tl.where(nv == 0, one, tl.where(nv == 1, two_x, zero))

    h_prev2 = one  # H_0
    h_prev = two_x  # H_1
    for k in tl.static_range(2, _MAX_DEG + 1):
        km1 = k - 1
        h_cur = (xv + xv) * h_prev - (km1 + km1) * h_prev2  # H_k
        res = tl.where(nv == k, h_cur, res)
        h_prev2 = h_prev
        h_prev = h_cur

    tl.store(out_ptr + offs64, res, mask=mask)


_DTYPE_MAP = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_BLOCK = 2048
_WARPS = 8


def _out_dtype(x_dtype, n_dtype):
    prom = torch.promote_types(x_dtype, n_dtype)
    if prom in _DTYPE_MAP:
        return prom
    return torch.float32  # all-int (or bool) inputs produce float32


def _bcast_strides(in_shape, out_shape):
    nd = len(out_shape)
    pad = nd - len(in_shape)
    in_padded = (1,) * pad + tuple(in_shape)
    # suffix products along the *input's own* padded shape
    suffix = [1] * nd
    for i in range(nd - 2, -1, -1):
        suffix[i] = suffix[i + 1] * in_padded[i + 1]
    strides = []
    for i in range(nd):
        if in_padded[i] == out_shape[i]:
            strides.append(suffix[i])
        else:
            strides.append(0)
    return tuple(strides)


def run(x, n):
    # --- n range validation (reference: flaggems special_hermite_polynomial_h) ---
    if isinstance(n, (int, float)):
        if int(n) < 0 or int(n) > _MAX_DEG.value:
            raise ValueError(
                f"special_hermite_polynomial_h only supports n in [0, {_MAX_DEG.value}], got n={n}"
            )
        n = torch.tensor(n, device=x.device)
    elif n.numel() == 1:
        n_int = int(n.item())
        if n_int < 0 or n_int > _MAX_DEG.value:
            raise ValueError(
                f"special_hermite_polynomial_h only supports n in [0, {_MAX_DEG.value}], got n={n_int}"
            )

    if isinstance(x, (int, float)):
        x = torch.tensor(x, device=n.device)

    out_dtype = _out_dtype(x.dtype, n.dtype)
    tl_dtype = _DTYPE_MAP[out_dtype]

    out_shape = tuple(torch.broadcast_shapes(x.shape, n.shape))
    numel = 1
    for d in out_shape:
        numel *= d
    out = torch.empty(out_shape, dtype=out_dtype, device=x.device)
    if numel == 0:
        return out

    grid = (triton.cdiv(numel, _BLOCK),)

    if x.numel() == numel and n.numel() == numel:
        # identical broadcastable shapes -> flat identity mapping
        _hermite_h_flat_kernel[grid](
            x,
            n,
            out,
            numel,
            1,
            1,
            BLOCK=_BLOCK,
            DTYPE=tl_dtype,
            num_warps=_WARPS,
        )
    elif len(out_shape) == 1:
        x_stride = 1 if x.numel() == numel else 0
        n_stride = 1 if n.numel() == numel else 0
        _hermite_h_flat_kernel[grid](
            x,
            n,
            out,
            numel,
            x_stride,
            n_stride,
            BLOCK=_BLOCK,
            DTYPE=tl_dtype,
            num_warps=_WARPS,
        )
    else:
        xs = _bcast_strides(x.shape, out_shape)
        ns = _bcast_strides(n.shape, out_shape)
        _hermite_h_nd_kernel[grid](
            x,
            n,
            out,
            numel,
            ND=len(out_shape),
            OUT_SHAPE=out_shape,
            XS=xs,
            NS=ns,
            BLOCK=_BLOCK,
            DTYPE=tl_dtype,
            num_warps=_WARPS,
        )
    return out


# Alias for FlagGems import convention
special_hermite_polynomial_h = run
