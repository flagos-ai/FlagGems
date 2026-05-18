import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

atan2 = tl_extra_shim.atan2

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def log10_complex_func(real, imag):
    has_nan = (real != real) | (imag != imag)
    real_pos_inf = real == float("inf")
    real_neg_inf = real == -float("inf")
    imag_pos_inf = imag == float("inf")
    imag_neg_inf = imag == -float("inf")
    real_inf = real_pos_inf | real_neg_inf
    imag_inf = imag_pos_inf | imag_neg_inf
    abs_real = tl.where(real < 0, -real, real)
    abs_imag = tl.where(imag < 0, -imag, imag)
    large = tl.maximum(abs_real, abs_imag)
    small = tl.minimum(abs_real, abs_imag)
    safe_large = tl.where(large == 0.0, 1.0, large)
    ratio = tl.where(
        large == float("inf"),
        tl.where(small == float("inf"), 1.0, 0.0),
        small / safe_large,
    )
    log_abs = tl.log(safe_large) + 0.5 * tl.log(1.0 + ratio * ratio)
    log_abs = tl.where(large == 0.0, -float("inf"), log_abs)
    real_out = log_abs * 0.4342944819032518
    imag_out = atan2(imag, real) * 0.4342944819032518
    inf_angle = imag_out
    inf_angle = tl.where(imag_pos_inf & real_pos_inf, 0.34109407663345337, inf_angle)
    inf_angle = tl.where(imag_pos_inf & real_neg_inf, 1.0232822895050049, inf_angle)
    inf_angle = tl.where(imag_neg_inf & real_pos_inf, -0.34109407663345337, inf_angle)
    inf_angle = tl.where(imag_neg_inf & real_neg_inf, -1.0232822895050049, inf_angle)
    inf_angle = tl.where(imag_pos_inf & ~real_inf, 0.6821881532669067, inf_angle)
    inf_angle = tl.where(imag_neg_inf & ~real_inf, -0.6821881532669067, inf_angle)
    inf_angle = tl.where(
        real_neg_inf & ~imag_inf,
        tl.where(imag < 0, -1.3643763065338135, 1.3643763065338135),
        inf_angle,
    )
    inf_angle = tl.where(
        real_pos_inf & ~imag_inf, tl.where(imag < 0, -0.0, 0.0), inf_angle
    )
    imag_out = tl.where(real_inf | imag_inf, inf_angle, imag_out)
    real_out = tl.where(has_nan, float("nan"), real_out)
    imag_out = tl.where(has_nan, float("nan"), imag_out)
    return real_out, imag_out


def _log10_complex(A):
    if A.dtype == torch.complex32:
        raise NotImplementedError("\"log10_cuda\" not implemented for 'ComplexHalf'")
    real, imag = log10_complex_func(A.real, A.imag)
    return torch.complex(real, imag)


def log10(A):
    logger.debug("GEMS LOG10")
    if A.is_complex():
        return _log10_complex(A)
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    if A.is_complex():
        A.copy_(_log10_complex(A))
        return A
    return log10_func(A, out0=A)


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    if A.is_complex():
        if not out.dtype.is_complex:
            raise RuntimeError(
                f"result type {A.dtype} can't be cast to the desired output type "
                f"{out.dtype}"
            )
        out.copy_(_log10_complex(A))
        return out
    return log10_func(A, out0=out)
