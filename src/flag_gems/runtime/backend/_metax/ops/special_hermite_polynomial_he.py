import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim  # noqa: F401

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def hermite_polynomial_he_tensor_tensor(x, n):
    # Probabilist's Hermite polynomial He_n(x)
    # Recurrence: He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)
    # He_0(x) = 1
    # He_1(x) = x
    # He_2(x) = x^2 - 1
    # He_3(x) = x^3 - 3*x
    # He_4(x) = x^4 - 6*x^2 + 3
    # He_5(x) = x^5 - 10*x^3 + 15*x

    n_int = n.to(tl.int32)
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    x4 = x2 * x2
    x5 = x4 * x_f32

    # Base case n = 0
    result_0 = tl.where(n_int == 0, 1.0, 0.0)

    # Base case n = 1
    result_1 = tl.where(n_int == 1, x_f32, 0.0)

    # He_2 = x^2 - 1
    he_2 = x2 - 1.0

    # He_3 = x^3 - 3*x
    he_3 = x3 - 3.0 * x_f32

    # He_4 = x^4 - 6*x^2 + 3
    he_4 = x4 - 6.0 * x2 + 3.0

    # He_5 = x^5 - 10*x^3 + 15*x
    he_5 = x5 - 10.0 * x3 + 15.0 * x_f32

    # He_6 through He_10: recurrence He_{n+1} = x*He_n - n*He_{n-1}
    he_6 = x_f32 * he_5 - 5.0 * he_4
    he_7 = x_f32 * he_6 - 6.0 * he_5
    he_8 = x_f32 * he_7 - 7.0 * he_6
    he_9 = x_f32 * he_8 - 8.0 * he_7
    he_10 = x_f32 * he_9 - 9.0 * he_8

    # Combine all cases
    result = result_0
    result = tl.where(n_int == 1, result_1, result)
    result = tl.where(n_int == 2, he_2, result)
    result = tl.where(n_int == 3, he_3, result)
    result = tl.where(n_int == 4, he_4, result)
    result = tl.where(n_int == 5, he_5, result)
    result = tl.where(n_int == 6, he_6, result)
    result = tl.where(n_int == 7, he_7, result)
    result = tl.where(n_int == 8, he_8, result)
    result = tl.where(n_int == 9, he_9, result)
    result = tl.where(n_int == 10, he_10, result)
    # For n > 10, approximate with He_{11}
    result = tl.where(n_int >= 11, x_f32 * he_10 - 10.0 * he_9, result)

    return result


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def hermite_polynomial_he_tensor_scalar(x, n):
    # Probabilist's Hermite polynomial He_n(x) with scalar n
    n_int = n.to(tl.int32)
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    x4 = x2 * x2
    x5 = x4 * x_f32

    # Base case n = 0
    result_0 = tl.where(n_int == 0, 1.0, 0.0)

    # Base case n = 1
    result_1 = tl.where(n_int == 1, x_f32, 0.0)

    # He_2 = x^2 - 1
    he_2 = x2 - 1.0

    # He_3 = x^3 - 3*x
    he_3 = x3 - 3.0 * x_f32

    # He_4 = x^4 - 6*x^2 + 3
    he_4 = x4 - 6.0 * x2 + 3.0

    # He_5 = x^5 - 10*x^3 + 15*x
    he_5 = x5 - 10.0 * x3 + 15.0 * x_f32

    # He_6 through He_10: recurrence He_{n+1} = x*He_n - n*He_{n-1}
    he_6 = x_f32 * he_5 - 5.0 * he_4
    he_7 = x_f32 * he_6 - 6.0 * he_5
    he_8 = x_f32 * he_7 - 7.0 * he_6
    he_9 = x_f32 * he_8 - 8.0 * he_7
    he_10 = x_f32 * he_9 - 9.0 * he_8

    # Combine all cases
    result = result_0
    result = tl.where(n_int == 1, result_1, result)
    result = tl.where(n_int == 2, he_2, result)
    result = tl.where(n_int == 3, he_3, result)
    result = tl.where(n_int == 4, he_4, result)
    result = tl.where(n_int == 5, he_5, result)
    result = tl.where(n_int == 6, he_6, result)
    result = tl.where(n_int == 7, he_7, result)
    result = tl.where(n_int == 8, he_8, result)
    result = tl.where(n_int == 9, he_9, result)
    result = tl.where(n_int == 10, he_10, result)
    # For n > 10, approximate with He_{11}
    result = tl.where(n_int >= 11, x_f32 * he_10 - 10.0 * he_9, result)

    return result


def hermite_polynomial_he(x, n):
    logger.debug("METAX GEMS HERMITE_POLYNOMIAL_HE")

    if isinstance(n, torch.Tensor):
        return hermite_polynomial_he_tensor_tensor(x, n)
    else:
        # n is a scalar
        return hermite_polynomial_he_tensor_scalar(x, n)
