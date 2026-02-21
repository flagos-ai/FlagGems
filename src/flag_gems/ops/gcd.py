import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gcd_func(x, y):
    a = tl.abs(x)
    b = tl.abs(y)

    # Euclidean algorithm: gcd(a, b) = gcd(b, a % b)
    # Continue for 64 iterations (enough for int64)
    for _ in range(64):
        b_nonzero = b != 0

        # Safe modulo: replace b with 1 when it's 0 to avoid division by zero
        # The result will be masked out anyway
        b_safe = tl.where(b_nonzero, b, 1)

        r = a % b_safe

        new_a = tl.where(b_nonzero, b, a)
        new_b = tl.where(b_nonzero, r, 0)

        a = new_a
        b = new_b

    return a


def gcd(A, B):
    """
    Computes the greatest common divisor (GCD) of two integer tensors.

    Args:
        A: First input tensor (must be integer type: int8, int16, int32, int64, uint8)
        B: Second input tensor (must be integer type: int8, int16, int32, int64, uint8)

    Returns:
        Tensor containing element-wise GCD of A and B

    Raises:
        TypeError: If inputs are not integer tensors (e.g., float, complex, bool)

    Supported dtypes:
        - torch.int8, torch.int16, torch.int32, torch.int64
        - torch.uint8

    Not supported:
        - Float types: torch.float16, torch.float32, torch.float64, torch.bfloat16
        - Complex types: torch.complex32, torch.complex64, torch.complex128
        - Bool type: torch.bool
        - Quantized types: torch.qint8, torch.quint8, etc.
    """
    logger.debug("GEMS GCD")

    valid_int_types = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }

    if A.dtype not in valid_int_types or B.dtype not in valid_int_types:
        raise TypeError(
            f"GCD does not support non-integer tensors. "
            f"Got A.dtype={A.dtype}, B.dtype={B.dtype}. "
            f"GCD is only defined for integers."
        )

    return gcd_func(A, B)
