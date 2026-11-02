import torch
import triton
import triton.language as tl

_printed = False


def _log(msg):
    global _printed
    if not _printed:
        _printed = True
        print(msg, flush=True)


@triton.jit
def _legendre_flat_kernel(
    x_ptr,
    out_ptr,
    numel,
    nv_scalar,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fast path: scalar degree n, contiguous x, flat linear indexing.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
    else:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Legendre recurrence: P_0=1, P_1=x,
    # P_k = ((2k-1) x P_{k-1} - (k-1) P_{k-2}) / k
    p_prev2 = tl.full([BLOCK], 1.0, dtype=x.dtype)
    p_prev1 = x
    res = tl.where(nv_scalar == 0, p_prev2, p_prev1)
    res = tl.where(nv_scalar < 0, 0.0, res)
    for k in range(2, nv_scalar + 1):
        kf = k.to(x.dtype)
        c1 = (2.0 * kf - 1.0) / kf
        c2 = (kf - 1.0) / kf
        p_next = c1 * (x * p_prev1) - c2 * p_prev2
        p_prev2 = p_prev1
        p_prev1 = p_next
        res = tl.where(nv_scalar == k, p_prev1, res)
    if EVEN:
        tl.store(out_ptr + offs, res)
    else:
        tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _legendre_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    nv_scalar,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    NDIM: tl.constexpr,
    SCALAR_N: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    if FLAT:
        x_off = offs.to(tl.int64)
        n_off = offs.to(tl.int64)
    else:
        rem = offs.to(tl.int64)
        x_off = tl.zeros([BLOCK], dtype=tl.int64)
        n_off = tl.zeros([BLOCK], dtype=tl.int64)
        for d in tl.static_range(8):
            if d < NDIM:
                if d == 0:
                    sz = s0
                    xo = x0
                    no = n0
                elif d == 1:
                    sz = s1
                    xo = x1
                    no = n1
                elif d == 2:
                    sz = s2
                    xo = x2
                    no = n2
                elif d == 3:
                    sz = s3
                    xo = x3
                    no = n3
                elif d == 4:
                    sz = s4
                    xo = x4
                    no = n4
                elif d == 5:
                    sz = s5
                    xo = x5
                    no = n5
                elif d == 6:
                    sz = s6
                    xo = x6
                    no = n6
                else:
                    sz = s7
                    xo = x7
                    no = n7
                sz64 = sz.to(tl.int64)
                c = rem % sz64
                rem = rem // sz64
                x_off += c * xo.to(tl.int64)
                if not SCALAR_N:
                    n_off += c * no.to(tl.int64)

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    if SCALAR_N:
        nv = tl.full([BLOCK], nv_scalar, dtype=tl.int32)
        block_max = nv_scalar
    else:
        nv = tl.load(n_ptr + n_off, mask=mask, other=0).to(tl.int32)
        block_max = tl.max(nv, axis=0)

    p_prev2 = tl.full([BLOCK], 1.0, dtype=x.dtype)
    p_prev1 = x
    res = tl.where(nv == 0, p_prev2, p_prev1)
    res = tl.where(nv < 0, 0.0, res)
    for k in range(2, block_max + 1):
        kf = k.to(x.dtype)
        c1 = (2.0 * kf - 1.0) / kf
        c2 = (kf - 1.0) / kf
        p_next = c1 * (x * p_prev1) - c2 * p_prev2
        p_prev2 = p_prev1
        p_prev1 = p_next
        res = tl.where(nv == k, p_prev1, res)
    tl.store(out_ptr + offs, res, mask=mask)


def _prepare_dims(out_shape, x, n):
    ndim = len(out_shape)
    if ndim > 8:
        raise RuntimeError(f"legendre: unsupported rank {ndim}")
    s = [1] * 8
    xs = [0] * 8
    ns = [0] * 8
    # Kernel decomposes the flat out index with d=0 the innermost dim.
    # x/n dims align with out trailing dims: kernel dim d corresponds to
    # x dim (x.dim()-1-d) / n dim (n.dim()-1-d); missing or size-1 dims
    # are broadcast (stride 0).
    for d in range(ndim):
        s[d] = int(out_shape[ndim - 1 - d])
        xd = x.dim() - 1 - d
        if xd >= 0 and x.shape[xd] != 1:
            xs[d] = int(x.stride(xd))
        nd_ = n.dim() - 1 - d
        if nd_ >= 0 and n.shape[nd_] != 1:
            ns[d] = int(n.stride(nd_))
    return s, xs, ns


def _block_cfg(numel, xdt):
    if xdt == torch.float64:
        return 512, 4
    if numel >= (1 << 25):
        return 4096, 16
    return 1024, 4


def run(x, n):
    dev = x.device
    xdt = x.dtype

    is_tensor_n = isinstance(n, torch.Tensor)
    tensor_mode = is_tensor_n and n.numel() > 1

    if tensor_mode:
        out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(n.shape))
        out_dtype = (
            torch.promote_types(xdt, n.dtype) if n.dtype.is_floating_point else xdt
        )
        _log(
            f"[legendre] tensor-n x={tuple(x.shape)}/{xdt} n={tuple(n.shape)}/{n.dtype} "
            f"out={tuple(out_shape)}/{out_dtype}"
        )
        nv_scalar = 0
    else:
        out_shape = tuple(x.shape)
        out_dtype = xdt
        nv = int(n.reshape(-1)[0].item()) if is_tensor_n else int(n)
        _log(
            f"[legendre] scalar-n x={tuple(x.shape)}/{xdt} n={nv} type={type(n).__name__}"
        )
        nv_scalar = nv

    out = torch.empty(out_shape, dtype=out_dtype, device=dev)
    numel = out.numel()
    if numel == 0:
        return out

    if (not tensor_mode) and x.is_contiguous():
        BLOCK, warps = _block_cfg(numel, xdt)
        grid = (triton.cdiv(numel, BLOCK),)
        _legendre_flat_kernel[grid](
            x,
            out,
            numel,
            nv_scalar,
            EVEN=(numel % BLOCK == 0),
            BLOCK=BLOCK,
            num_warps=warps,
        )
        return out

    flat = x.is_contiguous() and n.is_contiguous() and tuple(n.shape) == tuple(x.shape)
    if flat:
        s = [1] * 8
        xs = [0] * 8
        ns = [0] * 8
        ndim = 0
    else:
        s, xs, ns = _prepare_dims(out_shape, x, n if tensor_mode else x)
        ndim = len(out_shape)

    BLOCK, warps = _block_cfg(numel, xdt)
    grid = (triton.cdiv(numel, BLOCK),)
    _legendre_kernel[grid](
        x,
        n if tensor_mode else x,
        out,
        numel,
        nv_scalar,
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
        xs[7],
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        NDIM=ndim,
        SCALAR_N=(not tensor_mode),
        FLAT=flat,
        BLOCK=BLOCK,
        num_warps=warps,
    )
    return out


# Alias for FlagGems import convention
special_legendre_polynomial_p = run
