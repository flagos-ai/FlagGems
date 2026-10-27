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
#
# Kunlunxin (XPU) override of special_shifted_chebyshev_polynomial_v.
#
# Two independent defects were measured in the previous version of this file on
# this backend (2026-08-30, XPU2):
#
# 1. Correctness, silent.  The degree was selected with a float *equality* chain
#    (`tl.where(nf == k, vk, ...)`, nf = n.to(tl.float32)).  ATen truncates n
#    toward zero (`static_cast<int64_t>(n)`, ATen/native/Math.h:3854), so any
#    non-integral n must give V_trunc(n); the equality chain matches no level at
#    all and returns 0.0.  Measured against a CPU fp64 ATen oracle: n = 2.7
#    returned 0.0 where ATen gives V_2, and on the *benchmark* input
#    distribution (benchmark/base.py:743 feeds inp2 = randn(float32) as n)
#    100% of the outputs were 0.0 while ATen zeroes only ~16% -> 882064/1048576
#    elements wrong at [1024,1024], max abs error 1.16e4.  The equality chain is
#    replaced by a monotone `nf >= k` chain, which reproduces truncation exactly
#    at the same select cost.  Sibling `_w` already truncates via
#    `n.to(tl.int32)`; `_v` was the only member of the family with this defect.
#
# 2. Throughput.  The file imported the *generic* `flag_gems.utils`
#    pointwise_dynamic while passing a *vendor* `CodeGenConfig`.  The two
#    `CodeGenConfig` classes are distinct, so the generic codegen silently drops
#    the four XPU-only fields (buffer_size_limit / isCloseVectorization /
#    kunlunAutoGrid / unroll_num) and the remaining five happen to equal its own
#    defaults -- i.e. `config=config_` was a complete no-op.  The import is now
#    paired with the vendor codegen (`..utils.pointwise_dynamic`).
#
# Also fixed while rewriting the select chain: V_0 was built as `xf * 0.0 + 1.0`,
# which yields NaN for a non-finite x and then poisons the whole recurrence
# through the `-v0` term.  ATen returns 1.0 for x = +-inf / NaN at n = 0 and
# +inf for x = +-inf at n = 2.  Both arms of the first select are now literals.
#
# Deliberately *kept* (not an ATen match, but the documented contract of this
# operator in this repository, asserted by
# tests/test_special_shifted_chebyshev_polynomial_v.py:52 and :68):
# degrees outside [0, 15] return 0.0.  ATen instead evaluates V_16, V_100, ...
# The trailing `nf < 16.0` gate keeps that contract and, as a side effect, makes
# n = +-inf / NaN return 0.0, which *is* what ATen does.
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
def _cheb_v(xf, nf):
    # V*_0 = 1, V*_1 = 4x - 3, V*_{k+1} = (4x - 2) * V*_k - V*_{k-1}
    coefficient = 4.0 * xf - 2.0
    v1 = 4.0 * xf - 3.0
    # trunc(n) < 0 -> 0.0.  Both arms are literals so that a non-finite x cannot
    # leak into V_0 (x = +-inf / NaN with n = 0 must give 1.0).
    res = tl.where(nf > -1.0, 1.0, 0.0)  # V_0
    res = tl.where(nf >= 1.0, v1, res)  # V_1
    vkm1 = 1.0
    vk = v1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 2.0, vkp1, res)  # V_2
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 3.0, vkp1, res)  # V_3
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 4.0, vkp1, res)  # V_4
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 5.0, vkp1, res)  # V_5
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 6.0, vkp1, res)  # V_6
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 7.0, vkp1, res)  # V_7
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 8.0, vkp1, res)  # V_8
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 9.0, vkp1, res)  # V_9
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 10.0, vkp1, res)  # V_10
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 11.0, vkp1, res)  # V_11
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 12.0, vkp1, res)  # V_12
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 13.0, vkp1, res)  # V_13
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 14.0, vkp1, res)  # V_14
    vkm1 = vk
    vk = vkp1
    vkp1 = tl.fma(coefficient, vk, -vkm1)
    res = tl.where(nf >= 15.0, vkp1, res)  # V_15
    # Repository contract: degree outside [0, 15] -> 0.0 (also covers
    # n = +-inf / NaN, where ATen does return 0.0).
    return tl.where(nf < 16.0, res, 0.0)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def _shifted_v(x, n):
    return _cheb_v(x.to(tl.float32), n.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def _shifted_v_scalar_n(x, n):
    return _cheb_v(x.to(tl.float32), n.to(tl.float32))


def special_shifted_chebyshev_polynomial_v(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_V")
    if x.dtype != torch.float32:
        raise ValueError(f"Unsupported dtype {x.dtype}, only float32 is supported")
    if not isinstance(n, torch.Tensor):
        return _shifted_v_scalar_n(x, n)
    return _shifted_v(x, n)
