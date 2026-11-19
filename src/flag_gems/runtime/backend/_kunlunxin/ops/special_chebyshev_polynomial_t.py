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


# Fast path used when every lane shares the same degree: the degree becomes a
# constexpr, so the recurrence is unrolled to exactly ND steps and the ten
# `tl.where` degree selections disappear (they cost more than the recurrence
# itself here, since each select is a full pass over the tile). The recurrence
# is written as `a * b - c` instead of `tl.fma`: `tl.fma` lowers to a ~3x
# slower sequence on this backend.
_UNIFORM_BLOCK = 8192
# Largest degree the slow kernel's unrolled recurrence covers.
_UNIFORM_MAX_DEGREE = 9


@triton.jit
def _cheb_t_uniform_kernel(
    x_ptr, out_ptr, numel, ND: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xf = tl.load(x_ptr + offs, mask=mask)
    if ND < 0:
        res = tl.zeros(xf.shape, tl.float32)
    elif ND == 0:
        res = tl.full(xf.shape, 1.0, tl.float32)
    else:
        two_x = xf + xf
        tkm1 = tl.full(xf.shape, 1.0, tl.float32)
        tk = xf
        for k in tl.static_range(2, ND + 1):
            tkp1 = two_x * tk - tkm1
            tkm1 = tk
            tk = tkp1
        res = tk
    tl.store(out_ptr + offs, res, mask=mask)


def _uniform_degree(n):
    """Return the degree as an int when it is a single integral value."""
    if isinstance(n, torch.Tensor):
        if n.dim() != 0:
            return None
        n = n.item()
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return None
    if n != int(n):
        return None
    return int(n)


def _launch_uniform(x, deg, out):
    numel = x.numel()
    if numel != 0:
        nd = max(min(deg, 9), -1)
        grid = (triton.cdiv(numel, _UNIFORM_BLOCK),)
        _cheb_t_uniform_kernel[grid](x, out, numel, ND=nd, BLOCK=_UNIFORM_BLOCK)
    return out


def _fast_path_ok(x, out=None):
    # The slow kernel unrolls a fixed ten degrees, so it is only correct for
    # |n| <= 9; keep every larger degree on that path so this fast path does not
    # change any result (the wrong value for n >= 10 is a pre-existing bug).
    if x.dtype != torch.float32 or not x.is_contiguous():
        return False
    if out is not None and (
        out.dtype != x.dtype or out.shape != x.shape or not out.is_contiguous()
    ):
        return False
    return True


def _check_dtype(x):
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "special_chebyshev_polynomial_t only supports "
            f"float32/float64, got {x.dtype}"
        )


def special_chebyshev_polynomial_t(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    _check_dtype(x)
    if _fast_path_ok(x):
        deg = _uniform_degree(n)
        if deg is not None and deg <= _UNIFORM_MAX_DEGREE:
            return _launch_uniform(x, deg, torch.empty_like(x))
    if not isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_kernel_scalar_n(x, n)
    return chebyshev_polynomial_t_kernel(x, n)


def special_chebyshev_polynomial_t_out(x, n, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT")
    _check_dtype(x)
    if _fast_path_ok(x, out):
        deg = _uniform_degree(n)
        if deg is not None and deg <= _UNIFORM_MAX_DEGREE:
            return _launch_uniform(x, deg, out)
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
