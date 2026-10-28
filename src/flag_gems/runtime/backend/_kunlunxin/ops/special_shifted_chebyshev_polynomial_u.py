import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@triton.jit
def _cheb_u(x_f32, n_f32):
    # Shifted Chebyshev polynomial of the second kind:
    #   U*_n(x) = U_n(2x - 1),  y = 2x - 1
    #   U*_0 = 1,  U*_1 = 2y,  U*_{k+1} = 2y * U*_k - U*_{k-1}
    y = x_f32 * 2.0 - 1.0
    two_y = 2.0 * y
    # Degree selection: monotone `n >= k` chain.  ATen truncates n toward zero
    # (static_cast<int64_t>(n)), so this reproduces n < 0 -> 0.0, -1 < n < 1 ->
    # U*_0, n = 3.7 -> U*_3 and n = NaN -> 0.0.  Both arms of the first select
    # are literals so that a non-finite x cannot poison U*_0
    # (x = +-inf / NaN with n = 0 must give 1.0).
    res = tl.where(n_f32 > -1.0, 1.0, 0.0)  # U*_0
    res = tl.where(n_f32 >= 1.0, two_y, res)  # U*_1
    ukm2 = 1.0
    ukm1 = two_y
    # k = 2..9
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 2.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 3.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 4.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 5.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 6.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 7.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 8.0, uk, res)
    ukm2 = ukm1
    ukm1 = uk
    uk = two_y * ukm1 - ukm2
    res = tl.where(n_f32 >= 9.0, uk, res)
    return res


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def cheb_u_kernel(x, n):
    return _cheb_u(x.to(tl.float32), n.to(tl.float32)).to(x.dtype)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def cheb_u_kernel_scalar_n(x, n):
    return _cheb_u(x.to(tl.float32), n.to(tl.float32)).to(x.dtype)


def special_shifted_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_U")
    if x.dtype not in (torch.float32,):
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if not isinstance(n, torch.Tensor):
        return cheb_u_kernel_scalar_n(x, n)
    return cheb_u_kernel(x, n)


def special_shifted_chebyshev_polynomial_u_(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_U_")
    if x.dtype not in (torch.float32,):
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if not isinstance(n, torch.Tensor):
        return cheb_u_kernel_scalar_n(x, n, out0=x)
    return cheb_u_kernel(x, n, out0=x)
