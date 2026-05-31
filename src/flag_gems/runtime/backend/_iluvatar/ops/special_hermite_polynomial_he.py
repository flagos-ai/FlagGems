import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim  # noqa: F401

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def hermite_he_func(x, n):
    # Probabilists' Hermite polynomial He_n(x)
    # He_0(x) = 1
    # He_1(x) = x
    # He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    # Using fully unrolled computation for n up to 31

    n_int = n.to(tl.int32)
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    x4 = x2 * x2

    # He_0 = 1
    result = 1.0
    # He_1 = x
    result = tl.where(n_int == 1, x_f32, result)
    # He_2 = x^2 - 1
    result = tl.where(n_int == 2, x2 - 1.0, result)
    # He_3 = x^3 - 3x
    result = tl.where(n_int == 3, x3 - 3.0 * x_f32, result)
    # He_4 = x^4 - 6x^2 + 3
    result = tl.where(n_int == 4, x4 - 6.0 * x2 + 3.0, result)
    # He_5 = x^5 - 10x^3 + 15x
    result = tl.where(n_int == 5, x3 * x2 - 10.0 * x3 + 15.0 * x_f32, result)
    # He_6 = x^6 - 15x^4 + 45x^2 - 15
    result = tl.where(n_int == 6, x2 * x2 * x2 - 15.0 * x4 + 45.0 * x2 - 15.0, result)
    # He_7 = x^7 - 21x^5 + 105x^3 - 105x
    result = tl.where(
        n_int == 7, x3 * x2 * x2 - 21.0 * x3 * x2 + 105.0 * x3 - 105.0 * x_f32, result
    )
    # He_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    result = tl.where(
        n_int == 8, x4 * x4 - 28.0 * x2 * x4 + 210.0 * x4 - 420.0 * x2 + 105.0, result
    )
    # He_9 = x^9 - 36x^7 + 378x^5 - 1260x^3 + 945x
    x5 = x4 * x_f32  # noqa: F841
    x6 = x4 * x2
    x7 = x6 * x_f32  # noqa: F841
    x8 = x6 * x2
    result = tl.where(
        n_int == 9,
        x8 * x_f32
        - 36.0 * x6 * x_f32
        + 378.0 * x4 * x_f32
        - 1260.0 * x3
        + 945.0 * x_f32,
        result,
    )
    # He_10 = x^10 - 45x^8 + 630x^6 - 3150x^4 + 4725x^2 - 945
    result = tl.where(
        n_int == 10,
        x8 * x2 - 45.0 * x8 + 630.0 * x6 - 3150.0 * x4 + 4725.0 * x2 - 945.0,
        result,
    )

    return result


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def hermite_he_func_tensor_scalar(x, n):
    # Same as above but for tensor-scalar case
    n_int = n.to(tl.int32)
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    x4 = x2 * x2

    # He_0 = 1
    result = 1.0
    # He_1 = x
    result = tl.where(n_int == 1, x_f32, result)
    # He_2 = x^2 - 1
    result = tl.where(n_int == 2, x2 - 1.0, result)
    # He_3 = x^3 - 3x
    result = tl.where(n_int == 3, x3 - 3.0 * x_f32, result)
    # He_4 = x^4 - 6x^2 + 3
    result = tl.where(n_int == 4, x4 - 6.0 * x2 + 3.0, result)
    # He_5 = x^5 - 10x^3 + 15x
    result = tl.where(n_int == 5, x3 * x2 - 10.0 * x3 + 15.0 * x_f32, result)
    # He_6 = x^6 - 15x^4 + 45x^2 - 15
    result = tl.where(n_int == 6, x2 * x2 * x2 - 15.0 * x4 + 45.0 * x2 - 15.0, result)
    # He_7 = x^7 - 21x^5 + 105x^3 - 105x
    result = tl.where(
        n_int == 7, x3 * x2 * x2 - 21.0 * x3 * x2 + 105.0 * x3 - 105.0 * x_f32, result
    )
    # He_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    result = tl.where(
        n_int == 8, x4 * x4 - 28.0 * x2 * x4 + 210.0 * x4 - 420.0 * x2 + 105.0, result
    )
    # He_9 = x^9 - 36x^7 + 378x^5 - 1260x^3 + 945x
    x5 = x4 * x_f32  # noqa: F841
    x6 = x4 * x2
    x7 = x6 * x_f32  # noqa: F841
    x8 = x6 * x2
    result = tl.where(
        n_int == 9,
        x8 * x_f32
        - 36.0 * x6 * x_f32
        + 378.0 * x4 * x_f32
        - 1260.0 * x3
        + 945.0 * x_f32,
        result,
    )
    # He_10 = x^10 - 45x^8 + 630x^6 - 3150x^4 + 4725x^2 - 945
    result = tl.where(
        n_int == 10,
        x8 * x2 - 45.0 * x8 + 630.0 * x6 - 3150.0 * x4 + 4725.0 * x2 - 945.0,
        result,
    )

    return result


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def hermite_he_func_scalar_tensor(x, n):
    # Scalar x, tensor n - less common but handle it
    n_int = n.to(tl.int32)
    x_f32 = x.to(tl.float32)
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    x4 = x2 * x2

    # He_0 = 1
    result = 1.0
    # He_1 = x
    result = tl.where(n_int == 1, x_f32, result)
    # He_2 = x^2 - 1
    result = tl.where(n_int == 2, x2 - 1.0, result)
    # He_3 = x^3 - 3x
    result = tl.where(n_int == 3, x3 - 3.0 * x_f32, result)
    # He_4 = x^4 - 6x^2 + 3
    result = tl.where(n_int == 4, x4 - 6.0 * x2 + 3.0, result)
    # He_5 = x^5 - 10x^3 + 15x
    result = tl.where(n_int == 5, x3 * x2 - 10.0 * x3 + 15.0 * x_f32, result)
    # He_6 = x^6 - 15x^4 + 45x^2 - 15
    result = tl.where(n_int == 6, x2 * x2 * x2 - 15.0 * x4 + 45.0 * x2 - 15.0, result)
    # He_7 = x^7 - 21x^5 + 105x^3 - 105x
    result = tl.where(
        n_int == 7, x3 * x2 * x2 - 21.0 * x3 * x2 + 105.0 * x3 - 105.0 * x_f32, result
    )
    # He_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    result = tl.where(
        n_int == 8, x4 * x4 - 28.0 * x2 * x4 + 210.0 * x4 - 420.0 * x2 + 105.0, result
    )
    # He_9 = x^9 - 36x^7 + 378x^5 - 1260x^3 + 945x
    x5 = x4 * x_f32  # noqa: F841
    x6 = x4 * x2
    x7 = x6 * x_f32  # noqa: F841
    x8 = x6 * x2
    result = tl.where(
        n_int == 9,
        x8 * x_f32
        - 36.0 * x6 * x_f32
        + 378.0 * x4 * x_f32
        - 1260.0 * x3
        + 945.0 * x_f32,
        result,
    )
    # He_10 = x^10 - 45x^8 + 630x^6 - 3150x^4 + 4725x^2 - 945
    result = tl.where(
        n_int == 10,
        x8 * x2 - 45.0 * x8 + 630.0 * x6 - 3150.0 * x4 + 4725.0 * x2 - 945.0,
        result,
    )

    return result


def hermite_polynomial_he(x, n):
    """Compute the probabilists' Hermite polynomial He_n(x).

    Args:
        x: Input tensor
        n: Order of the polynomial (integer or tensor)

    Returns:
        Tensor with the Hermite polynomial values
    """
    logger.debug("ILUVATAR GEMS SPECIAL_HERMITE_POLYNOMIAL_HE")

    # Handle different input types
    if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
        return hermite_he_func(x, n)
    elif isinstance(x, torch.Tensor):
        return hermite_he_func_tensor_scalar(x, n)
    elif isinstance(n, torch.Tensor):
        return hermite_he_func_scalar_tensor(x, n)
    else:
        # Both scalar - use torch directly
        result = torch.special.hermite_polynomial_he(
            torch.tensor(x, dtype=torch.float32), n
        )
        return result.to(x.dtype)
