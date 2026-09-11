import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, "INT_TO_FLOAT"), (1, "INT_TO_FLOAT")],
)
@triton.jit
def shifted_chebyshev_polynomial_w(x, n):
    """Compute shifted Chebyshev polynomial of the second kind W_n(x).

    The shifted Chebyshev polynomial of the second kind is defined as:
    W_n(x) = U_n(2x - 1) where U_n is the Chebyshev polynomial of the second kind.
    Using y = 2x - 1:
    W_0 = 1
    W_1 = 2y + 1 = 4x - 1
    W_n = 2y * W_{n-1} - W_{n-2}
    """
    # Convert to float for computation
    x_f = x.to(tl.float32)
    n_i = n.to(tl.int32)

    two_x_minus_1 = 2.0 * x_f - 1.0

    # W_0 = 1
    w0 = tl.zeros_like(x_f) + 1.0

    # W_1 = 4x - 1
    w1 = 4.0 * x_f - 1.0

    # W_2 = 2(2x-1) * W_1 - W_0
    #     = 2(2x-1)(4x-1) - 1 = 16x^2 - 12x + 1
    w2 = 16.0 * x_f * x_f - 12.0 * x_f + 1.0

    # W*_3(x) = 2(2x-1) * W*_2(x) - W*_1(x)
    w3 = 2.0 * two_x_minus_1 * w2 - w1

    # W*_4(x) = 2(2x-1) * W*_3(x) - W*_2(x)
    w4 = 2.0 * two_x_minus_1 * w3 - w2

    # W*_5(x) = 2(2x-1) * W*_4(x) - W*_3(x)
    w5 = 2.0 * two_x_minus_1 * w4 - w3

    # W*_6(x) = 2(2x-1) * W*_5(x) - W*_4(x)
    w6 = 2.0 * two_x_minus_1 * w5 - w4

    # W*_7(x) = 2(2x-1) * W*_6(x) - W*_5(x)
    w7 = 2.0 * two_x_minus_1 * w6 - w5

    # W*_8(x) = 2(2x-1) * W*_7(x) - W*_6(x)
    w8 = 2.0 * two_x_minus_1 * w7 - w6

    # W*_9(x) = 2(2x-1) * W*_8(x) - W*_7(x)
    w9 = 2.0 * two_x_minus_1 * w8 - w7

    # W*_10(x) = 2(2x-1) * W*_9(x) - W*_8(x)
    w10 = 2.0 * two_x_minus_1 * w9 - w8

    # W*_11(x) = 2(2x-1) * W*_10(x) - W*_9(x)
    w11 = 2.0 * two_x_minus_1 * w10 - w9

    # W*_12(x) = 2(2x-1) * W*_11(x) - W*_10(x)
    w12 = 2.0 * two_x_minus_1 * w11 - w10

    # W*_13(x) = 2(2x-1) * W*_12(x) - W*_11(x)
    w13 = 2.0 * two_x_minus_1 * w12 - w11

    # W*_14(x) = 2(2x-1) * W*_13(x) - W*_12(x)
    w14 = 2.0 * two_x_minus_1 * w13 - w12

    # W*_15(x) = 2(2x-1) * W*_14(x) - W*_13(x)
    w15 = 2.0 * two_x_minus_1 * w14 - w13

    # W*_16(x) = 2(2x-1) * W*_15(x) - W*_14(x)
    w16 = 2.0 * two_x_minus_1 * w15 - w14

    # W*_17(x) = 2(2x-1) * W*_16(x) - W*_15(x)
    w17 = 2.0 * two_x_minus_1 * w16 - w15

    # W*_18(x) = 2(2x-1) * W*_17(x) - W*_16(x)
    w18 = 2.0 * two_x_minus_1 * w17 - w16

    # W*_19(x) = 2(2x-1) * W*_18(x) - W*_17(x)
    w19 = 2.0 * two_x_minus_1 * w18 - w17

    # W*_20(x) = 2(2x-1) * W*_19(x) - W*_18(x)
    w20 = 2.0 * two_x_minus_1 * w19 - w18

    # Combine results based on n value using sequential tl.where for clarity
    result = w20
    result = tl.where(n_i == 0, w0, result)
    result = tl.where(n_i == 1, w1, result)
    result = tl.where(n_i == 2, w2, result)
    result = tl.where(n_i == 3, w3, result)
    result = tl.where(n_i == 4, w4, result)
    result = tl.where(n_i == 5, w5, result)
    result = tl.where(n_i == 6, w6, result)
    result = tl.where(n_i == 7, w7, result)
    result = tl.where(n_i == 8, w8, result)
    result = tl.where(n_i == 9, w9, result)
    result = tl.where(n_i == 10, w10, result)
    result = tl.where(n_i == 11, w11, result)
    result = tl.where(n_i == 12, w12, result)
    result = tl.where(n_i == 13, w13, result)
    result = tl.where(n_i == 14, w14, result)
    result = tl.where(n_i == 15, w15, result)
    result = tl.where(n_i == 16, w16, result)
    result = tl.where(n_i == 17, w17, result)
    result = tl.where(n_i == 18, w18, result)
    result = tl.where(n_i == 19, w19, result)
    result = tl.where(n_i == 20, w20, result)

    return result


def special_shifted_chebyshev_polynomial_w(x: torch.Tensor, n: torch.Tensor):
    """Shifted Chebyshev polynomial of the second kind.

    Computes the shifted Chebyshev polynomial of the second kind W_n(x).

    Args:
        x: Input tensor
        n: Degree of the polynomial (tensor)

    Returns:
        Tensor with the computed polynomial values
    """
    logger.debug("METAX GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_W")
    if torch.any(n > 20):
        raise ValueError(
            "n must be <= 20, got values up to {}. "
            "The unrolled kernel only supports n <= 20.".format(int(n.max().item()))
        )
    return shifted_chebyshev_polynomial_w(x, n)
