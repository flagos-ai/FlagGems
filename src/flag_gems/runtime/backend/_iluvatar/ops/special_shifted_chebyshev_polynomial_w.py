import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_w_kernel(x, n):
    """
    Compute shifted Chebyshev polynomial of the second kind W_n(x).

    The shifted Chebyshev polynomial of the second kind is defined as:
    W_n(x) = U_n(2x - 1) where U_n is the Chebyshev polynomial of the second kind.
    Using y = 2x - 1:
    W_0 = 1
    W_1 = 2y + 1 = 4x - 1
    W_n = 2y * W_{n-1} - W_{n-2}
    """
    n_int = n.to(tl.int32)

    # Handle edge cases
    if n_int == 0:
        return tl.constexpr(1.0)
    elif n_int == 1:
        return 4.0 * x - 1.0

    # Compute using recurrence relation
    # W_0 = 1
    w_prev2 = tl.constexpr(1.0)
    # W_1 = 4x - 1
    w_prev1 = 4.0 * x - 1.0

    # Iterate from 2 to n
    # Using a loop unrolling approach for efficiency
    for i in range(2, 20):  # Max n=19 for reasonable performance
        if n_int < i:
            break
        w_current = 2.0 * (2.0 * x - 1.0) * w_prev1 - w_prev2
        w_prev2 = w_prev1
        w_prev1 = w_current

    return w_prev1


def special_shifted_chebyshev_polynomial_w(x: torch.Tensor, n: torch.Tensor):
    logger.debug("ILUVATAR GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_W")
    if torch.any(n > 19):
        raise ValueError(
            "n must be <= 19, got values up to {}. "
            "The iluvatar backend only supports n <= 19.".format(int(n.max().item()))
        )
    return shifted_chebyshev_polynomial_w_kernel(x, n)
