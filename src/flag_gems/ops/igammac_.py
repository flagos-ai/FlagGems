import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_helper import use_tl_extra

logger = logging.getLogger(__name__)


@use_tl_extra
@triton.jit
def lgamma(x):
    pass


@use_tl_extra
@triton.jit
def exp(x):
    pass


@use_tl_extra
@triton.jit
def pow(x, y):
    pass


# Small value to determine convergence - must be constexpr for Triton
EPS = tl.constexpr(1e-12)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def igammac_func(a, x):
    # igammac(a, x) = Q(a, x) = Gamma(a, x) / Gamma(a)
    # where Gamma(a, x) is the upper incomplete gamma function
    # Using the relation: Q(a, x) = 1 - P(a, x) = 1 - gamma(a, x) / Gamma(a)
    # where gamma(a, x) is the lower incomplete gamma function

    # Convert to float32 for computation
    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    # Compute log(Gamma(a)) for normalization
    log_gamma_a = lgamma(a_f32)

    # Compute the lower incomplete gamma function using series expansion
    # gamma(a, x) = x^a * e^(-x) * sum_{n=0}^{inf} x^n / (a * (a+1) * ... * (a+n))
    # Using: sum = 1/a + x/(a*(a+1)) + x^2/(a*(a+1)*(a+2)) + ...
    # We'll unroll the first 25 terms for accuracy

    # First term: 1/a
    term = 1.0 / a_f32
    sum_val = term

    # Second term: x/(a+1)
    term = term * x_f32 / (a_f32 + 1.0)
    sum_val = sum_val + term

    # Third term: x^2/((a+1)*(a+2))
    term = term * x_f32 / (a_f32 + 2.0)
    sum_val = sum_val + term

    # Fourth term
    term = term * x_f32 / (a_f32 + 3.0)
    sum_val = sum_val + term

    # Fifth term
    term = term * x_f32 / (a_f32 + 4.0)
    sum_val = sum_val + term

    # Sixth term
    term = term * x_f32 / (a_f32 + 5.0)
    sum_val = sum_val + term

    # Seventh term
    term = term * x_f32 / (a_f32 + 6.0)
    sum_val = sum_val + term

    # Eighth term
    term = term * x_f32 / (a_f32 + 7.0)
    sum_val = sum_val + term

    # Ninth term
    term = term * x_f32 / (a_f32 + 8.0)
    sum_val = sum_val + term

    # Tenth term
    term = term * x_f32 / (a_f32 + 9.0)
    sum_val = sum_val + term

    # Eleventh term
    term = term * x_f32 / (a_f32 + 10.0)
    sum_val = sum_val + term

    # Twelfth term
    term = term * x_f32 / (a_f32 + 11.0)
    sum_val = sum_val + term

    # Thirteenth term
    term = term * x_f32 / (a_f32 + 12.0)
    sum_val = sum_val + term

    # Fourteenth term
    term = term * x_f32 / (a_f32 + 13.0)
    sum_val = sum_val + term

    # Fifteenth term
    term = term * x_f32 / (a_f32 + 14.0)
    sum_val = sum_val + term

    # Add more terms for better accuracy
    # Sixteenth term
    term = term * x_f32 / (a_f32 + 15.0)
    sum_val = sum_val + term

    # Seventeenth term
    term = term * x_f32 / (a_f32 + 16.0)
    sum_val = sum_val + term

    # Eighteenth term
    term = term * x_f32 / (a_f32 + 17.0)
    sum_val = sum_val + term

    # Nineteenth term
    term = term * x_f32 / (a_f32 + 18.0)
    sum_val = sum_val + term

    # Twentieth term
    term = term * x_f32 / (a_f32 + 19.0)
    sum_val = sum_val + term

    # Add eps to avoid log(0)
    sum_val = sum_val + EPS

    # Compute lower incomplete gamma in log space
    # gamma(a, x) = x^a * e^(-x) * sum
    log_gamma_lower = a_f32 * tl.log(x_f32) - x_f32 + tl.log(sum_val)

    # Compute P(a, x) = gamma(a, x) / Gamma(a)
    # In log space: log(P) = log(gamma) - log(Gamma)
    log_p = log_gamma_lower - log_gamma_a

    # P(a, x) = exp(log_P), clamped to [0, 1]
    p = tl.exp(log_p)

    # Clamp to avoid numerical issues
    p = tl.where(p > 1.0, 1.0, p)
    p = tl.where(p < 0.0, 0.0, p)

    # Q(a, x) = 1 - P(a, x)
    q = 1.0 - p

    # Handle edge case: x <= 0 => Q(a, x) = 1
    q = tl.where(x_f32 <= 0.0, 1.0, q)

    return q


def igammac(A, B):
    logger.debug("GEMS IGAMMAC")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return igammac_func(A, B)
    elif isinstance(A, torch.Tensor):
        return igammac_func(A, B)
    elif isinstance(B, torch.Tensor):
        return igammac_func(A, B)
    else:
        return torch.tensor(torch.igammac(torch.tensor(A), torch.tensor(B)).item())


def igammac_(A, B):
    logger.debug("GEMS IGAMMAC_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return igammac_func(A, B, out0=A)
    elif isinstance(A, torch.Tensor):
        return igammac_func(A, B, out0=A)
    else:
        raise ValueError("Unreachable.")
