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
def _legendre_p(xf, nf):
    res = tl.where(nf > -1.0, 1.0, 0.0)
    res = tl.where(nf >= 1.0, xf, res)
    pkm1 = 1.0
    pk = xf
    pkp1 = (3.0 * xf * pk - 1.0 * pkm1) * 0.5
    res = tl.where(nf >= 2.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (5.0 * xf * pk - 2.0 * pkm1) * (1.0 / 3.0)
    res = tl.where(nf >= 3.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (7.0 * xf * pk - 3.0 * pkm1) * 0.25
    res = tl.where(nf >= 4.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (9.0 * xf * pk - 4.0 * pkm1) * 0.2
    res = tl.where(nf >= 5.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (11.0 * xf * pk - 5.0 * pkm1) * (1.0 / 6.0)
    res = tl.where(nf >= 6.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (13.0 * xf * pk - 6.0 * pkm1) * (1.0 / 7.0)
    res = tl.where(nf >= 7.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (15.0 * xf * pk - 7.0 * pkm1) * 0.125
    res = tl.where(nf >= 8.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (17.0 * xf * pk - 8.0 * pkm1) * (1.0 / 9.0)
    res = tl.where(nf >= 9.0, pkp1, res)
    pkm1 = pk
    pk = pkp1
    pkp1 = (19.0 * xf * pk - 9.0 * pkm1) * 0.1
    res = tl.where(nf >= 10.0, pkp1, res)
    return res


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def legendre_polynomial_p_kernel(x, n):
    return _legendre_p(x.to(tl.float32), n.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def legendre_polynomial_p_kernel_scalar_n(x, n):
    return _legendre_p(x.to(tl.float32), n.to(tl.float32))


def special_legendre_polynomial_p(x: torch.Tensor, n) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LEGENDRE_POLYNOMIAL_P")
    if x.dtype != torch.float32:
        raise TypeError("special_legendre_polynomial_p only supports torch.float32")
    if not isinstance(n, torch.Tensor):
        return legendre_polynomial_p_kernel_scalar_n(x, n)
    return legendre_polynomial_p_kernel(x, n)
