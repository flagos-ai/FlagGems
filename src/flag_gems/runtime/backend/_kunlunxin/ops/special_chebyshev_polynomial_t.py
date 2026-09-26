# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems.ops.special_chebyshev_polynomial_t")

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
def _cheb_t(xf, nf):
    two_x = xf + xf
    res = tl.where(nf > -1.0, 1.0, 0.0)
    res = tl.where(nf >= 1.0, xf, res)
    tkm1 = 1.0
    tk = xf
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 2.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 3.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 4.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 5.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 6.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 7.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 8.0, tkp1, res)
    tkm1 = tk
    tk = tkp1
    tkp1 = tl.fma(two_x, tk, -tkm1)
    res = tl.where(nf >= 9.0, tkp1, res)
    return res


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def chebyshev_polynomial_t_kernel(x, n):
    return _cheb_t(x.to(tl.float32), n.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def chebyshev_polynomial_t_kernel_scalar_n(x, n):
    return _cheb_t(x.to(tl.float32), n.to(tl.float32))


def _check_dtype(x):
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "special_chebyshev_polynomial_t only supports "
            f"float32/float64, got {x.dtype}"
        )


def special_chebyshev_polynomial_t(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    _check_dtype(x)
    if not isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_kernel_scalar_n(x, n)
    return chebyshev_polynomial_t_kernel(x, n)


def special_chebyshev_polynomial_t_out(x, n, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT")
    _check_dtype(x)
    if not isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_kernel_scalar_n(x, n, out0=out)
    return chebyshev_polynomial_t_kernel(x, n, out0=out)


import sys as _sys  # noqa: E402

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    for _name, _fn in (
        ("special_chebyshev_polynomial_t", special_chebyshev_polynomial_t),
        ("special_chebyshev_polynomial_t_out", special_chebyshev_polynomial_t_out),
    ):
        setattr(_generic_ops_module, _name, _fn)
