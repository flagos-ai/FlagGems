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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def legendre_polynomial_p_kernel(
    x_ptr, n_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    n = tl.load(n_ptr).to(tl.int32)

    one = 1.0 + 0.0 * x
    zero = 0.0 + 0.0 * x
    result = tl.where(n < 0, zero, x)
    result = tl.where(n == 0, one, result)
    result = tl.where(n == 1, x, result)

    prev2 = one
    prev1 = x
    for degree in tl.static_range(2, 256):
        use_degree = n >= degree
        current = tl.where(
            use_degree,
            ((2.0 * degree - 1.0) * x * prev1 - (degree - 1.0) * prev2) / degree,
            prev1,
        )
        prev2 = tl.where(use_degree, prev1, prev2)
        prev1 = current

    result = tl.where(n > 1, prev1, result)
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def special_legendre_polynomial_p(x: torch.Tensor, n) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LEGENDRE_POLYNOMIAL_P")
    if x.dtype != torch.float32:
        raise TypeError("special_legendre_polynomial_p only supports torch.float32")
    if isinstance(n, torch.Tensor):
        if n.numel() != 1 or n.device != x.device:
            raise ValueError("n must be a scalar tensor on the input device")
        n_tensor = n.to(dtype=torch.int64)
    else:
        if not isinstance(n, int):
            raise TypeError("n must be an int or scalar tensor")
        n_tensor = torch.empty((), dtype=torch.int64, device=x.device)
        n_tensor.fill_(n)

    x_contiguous = x.contiguous()
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    n_elements = x_contiguous.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        legendre_polynomial_p_kernel[grid](
            x_contiguous, n_tensor, out, n_elements, BLOCK_SIZE=256
        )
    return out
