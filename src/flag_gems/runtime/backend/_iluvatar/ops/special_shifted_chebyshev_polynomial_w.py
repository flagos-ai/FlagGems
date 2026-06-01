import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_w_kernel(x, n):
    """
    Compute shifted Chebyshev polynomial of the fourth kind W*_n(x).

    The shifted Chebyshev polynomial of the fourth kind is defined as:
    W*_n(x) = U_n(2x - 1) where U_n is the Chebyshev polynomial of the second kind.

    Using recurrence relation:
    W*_0(x) = 1
    W*_1(x) = 2x - 1
    W*_{n+1}(x) = 2(2x - 1) * W*_n(x) - W*_{n-1}(x)
    """
    n_int = n.to(tl.int32)

    # Handle edge cases
    if n_int == 0:
        return tl.constexpr(1.0)
    elif n_int == 1:
        return 2.0 * x - 1.0

    # Compute using recurrence relation
    # W*_0(x) = 1
    w_prev2 = tl.constexpr(1.0)
    # W*_1(x) = 2x - 1
    w_prev1 = 2.0 * x - 1.0

    # Iterate from 2 to n
    # Using a loop unrolling approach for efficiency
    for i in range(2, 20):  # Max n=19 for reasonable performance
        if n_int <= i:
            break
        w_current = 2.0 * (2.0 * x - 1.0) * w_prev1 - w_prev2
        w_prev2 = w_prev1
        w_prev1 = w_current

    return w_prev1


def shifted_chebyshev_polynomial_w(x: torch.Tensor, n: torch.Tensor):
    logger.debug("ILUVATAR GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_W")
    return shifted_chebyshev_polynomial_w_kernel(x, n)


def shifted_chebyshev_polynomial_w_out(
    x: torch.Tensor, n: torch.Tensor, out: torch.Tensor
):
    logger.debug("ILUVATAR GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_W_OUT")
    return shifted_chebyshev_polynomial_w_kernel(x, n, out)
