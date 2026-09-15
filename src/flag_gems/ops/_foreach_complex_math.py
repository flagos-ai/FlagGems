"""Complex element-wise math for the unary foreach family.

PyTorch's ``_foreach_exp``/``_foreach_log``/``_foreach_sin``/... accept complex
tensors, but the single-tensor FlagGems operators do not -- they raise
``KeyError: 'complex64'`` from Triton's dtype table, so there is nothing to
reuse.  Each function below therefore takes the real and imaginary components
as separate values and returns the pair, matching what
``foreach_unary_c2c_kernel`` expects.

Formulas are the standard principal-branch definitions; they are written once
here and shared by the registration table rather than repeated per operator.
"""

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

# ``tl`` has no hyperbolic or two-argument-arctangent primitives; the repository
# routes those through the libdevice shim, so the complex formulas below use the
# same source as the single-tensor operators do.
_sinh = tl_extra_shim.sinh
_cosh = tl_extra_shim.cosh
_atan2 = tl_extra_shim.atan2


@triton.jit
def _hypot(re, im):
    return tl.sqrt(re * re + im * im)


@triton.jit
def c_neg(re, im):
    return -re, -im


@triton.jit
def c_exp(re, im):
    # exp(a+bi) = e^a (cos b + i sin b)
    r = tl.exp(re)
    return r * tl.cos(im), r * tl.sin(im)


@triton.jit
def c_expm1(re, im):
    r = tl.exp(re)
    return r * tl.cos(im) - 1.0, r * tl.sin(im)


@triton.jit
def c_log(re, im):
    # log(z) = ln|z| + i arg(z)
    return tl.log(_hypot(re, im)), _atan2(im, re)


@triton.jit
def c_log1p(re, im):
    return c_log(re + 1.0, im)


@triton.jit
def c_log2(re, im):
    lr, li = c_log(re, im)
    inv = 1.4426950408889634  # 1 / ln 2
    return lr * inv, li * inv


@triton.jit
def c_log10(re, im):
    lr, li = c_log(re, im)
    inv = 0.4342944819032518  # 1 / ln 10
    return lr * inv, li * inv


@triton.jit
def c_sqrt(re, im):
    # Principal square root via the half-angle form, which avoids cancellation
    # for negative real parts.
    m = _hypot(re, im)
    a = tl.sqrt(0.5 * (m + re))
    b = tl.sqrt(0.5 * (m - re))
    return a, tl.where(im < 0, -b, b)


@triton.jit
def c_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def c_div(ar, ai, br, bi):
    d = br * br + bi * bi
    return (ar * br + ai * bi) / d, (ai * br - ar * bi) / d


@triton.jit
def c_reciprocal(re, im):
    return c_div(1.0, 0.0, re, im)


@triton.jit
def c_rsqrt(re, im):
    sr, si = c_sqrt(re, im)
    return c_reciprocal(sr, si)


@triton.jit
def c_sin(re, im):
    # sin(a+bi) = sin a cosh b + i cos a sinh b
    return tl.sin(re) * _cosh(im), tl.cos(re) * _sinh(im)


@triton.jit
def c_cos(re, im):
    return tl.cos(re) * _cosh(im), -tl.sin(re) * _sinh(im)


@triton.jit
def c_sinh(re, im):
    return _sinh(re) * tl.cos(im), _cosh(re) * tl.sin(im)


@triton.jit
def c_cosh(re, im):
    return _cosh(re) * tl.cos(im), _sinh(re) * tl.sin(im)


@triton.jit
def c_tan(re, im):
    sr, si = c_sin(re, im)
    cr, ci = c_cos(re, im)
    return c_div(sr, si, cr, ci)


@triton.jit
def c_tanh(re, im):
    sr, si = c_sinh(re, im)
    cr, ci = c_cosh(re, im)
    return c_div(sr, si, cr, ci)


@triton.jit
def c_sigmoid(re, im):
    # 1 / (1 + exp(-z))
    er, ei = c_exp(-re, -im)
    return c_div(1.0, 0.0, 1.0 + er, ei)


@triton.jit
def c_asin(re, im):
    # asin(z) = -i log(iz + sqrt(1 - z^2))
    zr, zi = c_mul(re, im, re, im)
    sr, si = c_sqrt(1.0 - zr, -zi)
    # i*z = (-im, re)
    lr, li = c_log(-im + sr, re + si)
    # -i * (lr + i li) = li - i lr
    return li, -lr


@triton.jit
def c_acos(re, im):
    ar, ai = c_asin(re, im)
    return 1.5707963267948966 - ar, -ai


@triton.jit
def c_atan(re, im):
    # atan(z) = (i/2) log((i + z) / (i - z)).  The division has to happen before
    # the logarithm: subtracting two logarithms instead measured a constant pi
    # error in the real part wherever the two arguments straddle the branch cut.
    qr, qi = c_div(re, 1.0 + im, -re, 1.0 - im)
    lr, li = c_log(qr, qi)
    # (i/2) * (lr + i li) = -li/2 + i lr/2
    return -0.5 * li, 0.5 * lr
