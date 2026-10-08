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
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    dtypes=[None, int],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit(do_not_specialize=["n"])
def special_legendre_polynomial_p_forward(x, n):
    # Compute the Legendre polynomial P_n(x) using Bonnet's recurrence relation.
    # P_0 = 1, P_1 = x, (n+1)*P_{n+1} = (2n+1)*x*P_n - n*P_{n-1}.
    # Loop bound is the runtime scalar n (vs. the generic implementation's
    # static 254-iteration unroll), so n=3 costs exactly 2 iterations.
    # NOTE: the vendor pointwise_dynamic codegen embeds this source inside a
    # synthetic triple-quoted wrapper, so the body must not contain any
    # triple-quote sequences at all (including in comments). Keep comments
    # single- or double-quote free of adjacent quote pairs.
    n_val = n.to(tl.int32)
    x_f32 = x.to(tl.float32)

    # Broadcast scalar constants to the tile shape via x (pointwise_dynamic
    # loads x as a block, so tl.full((1,), ...) / tl.zeros(1, ...) cannot
    # broadcast against it). Same idiom as special_chebyshev_polynomial_w.
    one = 1.0 + 0.0 * x_f32  # P_0 = 1
    zero = 0.0 + 0.0 * x_f32  # for n < 0 -> 0

    # Handle n < 0 case - return 0
    result = tl.where(n_val < 0, zero, x_f32)

    # P_0(x) = 1
    result = tl.where(n_val == 0, one, result)

    # P_1(x) = x
    result = tl.where(n_val == 1, x_f32, result)

    # For n > 1, use Bonnet's recurrence relation
    # i * P_i(x) = (2i-1) * x * P_{i-1}(x) - (i-1) * P_{i-2}(x)
    # We compute iteratively from P_0 and P_1. Same formula and iteration
    # order as the generic implementation; the only difference is the loop is
    # bounded by the runtime value of n (no mask/where inside the loop, and
    # zero iterations for n <= 1).
    p_prev2 = one  # P_0
    p_prev1 = x_f32  # P_1

    for i in tl.range(2, n_val + 1):
        p_curr = ((2.0 * i - 1.0) * x_f32 * p_prev1 - (i - 1.0) * p_prev2) / i
        p_prev2 = p_prev1
        p_prev1 = p_curr

    # Final result
    result = tl.where(n_val > 1, p_prev1, result)

    return result.to(x.dtype)


def special_legendre_polynomial_p(x: torch.Tensor, n) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LEGENDRE_POLYNOMIAL_P")
    assert x.dtype == torch.float32, "only float32 supported"
    # n arrives either as a Python int (n_scalar overload, the common case) or
    # as a 0-dim/1-element tensor (n_tensor overload); normalize to a host
    # scalar so the kernel receives it as a runtime loop bound.
    if isinstance(n, torch.Tensor):
        n = int(n.item())
    return special_legendre_polynomial_p_forward(x, n)
