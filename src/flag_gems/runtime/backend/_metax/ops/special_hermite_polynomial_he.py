import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.utils import pointwise_dynamic, tl_extra_shim  # noqa: F401

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def _hermite_he_recurrence(x, n):
    # He_0(x) = 1, He_1(x) = x, He_k(x) = x*He_{k-1}(x) - (k-1)*He_{k-2}(x).
    # The caller validates n to be in [0, 10], so every degree is evaluated and
    # the one matching n is selected.
    n_i32 = n.to(tl.int32)
    he_0 = 1.0
    he_1 = x
    he_2 = x * he_1 - 1.0 * he_0
    he_3 = x * he_2 - 2.0 * he_1
    he_4 = x * he_3 - 3.0 * he_2
    he_5 = x * he_4 - 4.0 * he_3
    he_6 = x * he_5 - 5.0 * he_4
    he_7 = x * he_6 - 6.0 * he_5
    he_8 = x * he_7 - 7.0 * he_6
    he_9 = x * he_8 - 8.0 * he_7
    he_10 = x * he_9 - 9.0 * he_8

    result = he_10
    result = tl.where(n_i32 == 9, he_9, result)
    result = tl.where(n_i32 == 8, he_8, result)
    result = tl.where(n_i32 == 7, he_7, result)
    result = tl.where(n_i32 == 6, he_6, result)
    result = tl.where(n_i32 == 5, he_5, result)
    result = tl.where(n_i32 == 4, he_4, result)
    result = tl.where(n_i32 == 3, he_3, result)
    result = tl.where(n_i32 == 2, he_2, result)
    result = tl.where(n_i32 == 1, he_1, result)
    result = tl.where(n_i32 == 0, he_0, result)
    return result


# Each kernel has two variants. Both evaluate the recurrence in a single dtype:
# the _fp64 one computes in float64 for full accuracy, while the default one
# computes in float32 so it stays usable on devices without float64 support.
# The output is stored into the buffer implied by input promotion either way.


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_hermite_polynomial_he_tensor_tensor(x, n):
    return _hermite_he_recurrence(x.to(tl.float32), n)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_hermite_polynomial_he_tensor_tensor_fp64(x, n):
    return _hermite_he_recurrence(x.to(tl.float64), n)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_hermite_polynomial_he_tensor_scalar(x, n):
    return _hermite_he_recurrence(x.to(tl.float32), n)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_hermite_polynomial_he_tensor_scalar_fp64(x, n):
    return _hermite_he_recurrence(x.to(tl.float64), n)


def special_hermite_polynomial_he(x, n):
    logger.debug("GEMS_METAX HERMITE_POLYNOMIAL_HE")

    # Validate n is in supported range [0, 10]
    if isinstance(n, torch.Tensor):
        n_int = n.to(torch.int32)
        n_min = n_int.min().item()
        n_max = n_int.max().item()
        if n_min < 0 or n_max > 10:
            raise ValueError(
                f"special_hermite_polynomial_he only supports n in [0, 10], "
                f"got n in [{n_min}, {n_max}]"
            )
    elif isinstance(n, (int, float)):
        if int(n) < 0 or int(n) > 10:
            raise ValueError(
                f"special_hermite_polynomial_he only supports n in [0, 10], got n={n}"
            )

    if runtime_device.support_fp64:
        tensor_tensor = special_hermite_polynomial_he_tensor_tensor_fp64
        tensor_scalar = special_hermite_polynomial_he_tensor_scalar_fp64
    else:
        tensor_tensor = special_hermite_polynomial_he_tensor_tensor
        tensor_scalar = special_hermite_polynomial_he_tensor_scalar

    if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
        return tensor_tensor(x, n)
    elif isinstance(x, torch.Tensor):
        # n is a scalar
        return tensor_scalar(x, n)
    elif isinstance(n, torch.Tensor):
        # x is a scalar - materialize it as a tensor with n's shape and dtype,
        # then reuse the tensor-tensor kernel. The kernel requires equal-shape
        # inputs; full_like also yields an output dtype matching the native
        # operator (fp64 output when n is fp64).
        return tensor_tensor(torch.full_like(n, float(x)), n)
    else:
        # Both scalar - compute via the recurrence in plain Python, then wrap
        # the result in a tensor (no torch compute API).
        xi = float(x)
        deg = int(n)
        he_nm1, he_n = 1.0, xi
        for k in range(1, deg):
            he_nm1, he_n = he_n, xi * he_n - k * he_nm1
        return torch.tensor(he_nm1 if deg == 0 else he_n)
