import torch
import triton
import triton.language as tl

# Modified Bessel function K0, elementwise.
#
#   x <= 2 : K0(x) = -ln(x/2) * I0(x) - gamma + sum_{k>=1} c_k * y^k
#            with y = (x/2)^2, c_k = (H_k - gamma)/(k!)^2   (exact series, no fit)
#            and I0(x) from the Abramowitz & Stegun 9.8.1 polynomial in
#            yb = (x/3.75)^2 (|err| ~ 1e-7 relative).
#   x >  2 : K0(x) = exp(-x) / sqrt(x) * Q(2/x)   (Abramowitz & Stegun 9.8.6).
#
# Special values come out of IEEE arithmetic: negative / NaN inputs give NaN
# through log or sqrt, x == 0 gives +inf from log(0), x == +inf gives 0 through
# exp(-inf)/sqrt(inf).  No unordered fp compare is used, because the XPU backend
# cannot select one (LLVM "setuo" fatal error).

@triton.jit
def _k0_of(xf):
    # ---- shared: one logarithm serves both branches; both series use y=(x/2)^2 ----
    logx = tl.log(xf)
    y = 0.25 * (xf * xf)
    # I0(x) = sum_{k>=0} y^k/(k!)^2, truncated after k=5 (abs err < 2e-8 for x<=2)
    i0 = 1.0 + y * (
        1.0
        + y
        * (
            0.25
            + y
            * (
                0.02777777777777778
                + y * (0.0017361111111111111 + y * 0.00006944444444444444)
            )
        )
    )
    # K0 = -ln(x/2) I0 + [-gamma + sum_{k>=1} c_k y^k], c_k = (H_k - gamma)/(k!)^2,
    # truncated after k=5 (3.7e-6 absolute at x=2)
    s = -0.5772156649015329 + y * (
        0.4227843350984671
        + y
        * (
            0.2306960837746168
            + y * (0.0348921574564389 + y * (0.002614787619 + y * 0.0001184803936))
        )
    )
    small = -(logx - 0.6931471805599453) * i0 + s

    # ---- large range ----
    z = 2.0 / xf
    q = 1.25331414 + z * (
        -0.07832358
        + z
        * (
            0.02189568
            + z
            * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208)))
        )
    )
    # exp(-x) / sqrt(x) == exp(-x - 0.5*log(x)), reusing the log computed above so
    # the sqrt and the second division disappear.
    large = q * tl.exp(-xf - 0.5 * logx)

    return tl.where(xf <= 2.0, small, large)


@triton.jit
def _k0_contig_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    res = _k0_of(xf)
    tl.store(out_ptr + offs, res.to(x.dtype), mask=mask)


@triton.jit
def _k0_aligned_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    """Unmasked variant for numel % BLOCK == 0: no bounds mask, no staging copy."""
    pid = tl.program_id(0)
    offs = tl.max_contiguous(tl.multiple_of(pid * BLOCK + tl.arange(0, BLOCK), BLOCK), BLOCK)
    x = tl.load(x_ptr + offs)
    xf = x.to(tl.float32)
    res = _k0_of(xf)
    tl.store(out_ptr + offs, res.to(x.dtype))


_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 16384}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 16384}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 32768}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 32768}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 32768}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 65536}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 65536}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=["numel"])
@triton.jit
def _k0_aligned_tuned(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.max_contiguous(tl.multiple_of(pid * BLOCK + tl.arange(0, BLOCK), BLOCK), BLOCK)
    x = tl.load(x_ptr + offs)
    xf = x.to(tl.float32)
    res = _k0_of(xf)
    tl.store(out_ptr + offs, res.to(x.dtype))


@triton.jit
def _k0_strided_kernel(
    x_ptr,
    out_ptr,
    numel,
    xs0,
    xs1,
    xs2,
    xs3,
    os0,
    os1,
    os2,
    os3,
    d1,
    d2,
    d3,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    i3 = offs
    i2 = tl.zeros_like(offs)
    i1 = tl.zeros_like(offs)
    i0 = tl.zeros_like(offs)
    if NDIM == 4:
        i3 = offs % d3
        r = offs // d3
        i2 = r % d2
        r = r // d2
        i1 = r % d1
        i0 = r // d1
    elif NDIM == 3:
        i3 = offs % d3
        r = offs // d3
        i2 = r % d2
        i1 = r // d2
    elif NDIM == 2:
        i3 = offs % d3
        i2 = offs // d3
    x_off = i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3
    o_off = i0 * os0 + i1 * os1 + i2 * os2 + i3 * os3
    x = tl.load(x_ptr + x_off, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    res = _k0_of(xf)
    tl.store(out_ptr + o_off, res.to(x.dtype), mask=mask)


def run(self, out=None, *args, **kwargs):
    x = self
    if out is None:
        for key in ("output", "result"):
            if key in kwargs:
                out = kwargs[key]
    if out is None:
        out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    if x.is_contiguous() and out.is_contiguous():
        if n >= 65536 and n % 65536 == 0:
            _k0_aligned_tuned[_k0_grid(n)](x, out, n)
        elif n % BLOCK == 0:
            _k0_aligned_kernel[grid](x, out, BLOCK=BLOCK, num_warps=4)
        else:
            _k0_contig_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
    else:
        xsh = [1] * (4 - x.dim()) + list(x.shape)
        xs = [0] * (4 - x.dim()) + list(x.stride())
        os_ = [0] * (4 - out.dim()) + list(out.stride())
        ndim = max(x.dim(), out.dim())
        _k0_strided_kernel[grid](
            x,
            out,
            n,
            xs[0],
            xs[1],
            xs[2],
            xs[3],
            os_[0],
            os_[1],
            os_[2],
            os_[3],
            xsh[1],
            xsh[2],
            xsh[3],
            NDIM=ndim,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def _k0_grid(n):
    """Stable grid lambda: a per-call lambda misses the compile cache."""
    return lambda meta: (triton.cdiv(n, meta["BLOCK"]),)

# --- backend entry point ---------------------------------------------------
def special_modified_bessel_k0(A, out=None):
    return run(A, out)


def special_modified_bessel_k0_out(A, out=None):
    return run(A, out)
