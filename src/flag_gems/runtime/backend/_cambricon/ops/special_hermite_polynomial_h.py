# Copyright 2026, The FlagOS Contributors.
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

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@triton.jit
def _hermite_step(x, h_nm1, h_n, n):
    return 2.0 * x * h_n - 2.0 * n * h_nm1


@triton.jit
def _hermite_tensor_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    n_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    M: tl.constexpr,
):
    grid_0 = tl.num_programs(0)
    pid = tl.program_id(0)
    while pid < M:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        n_offsets = tl.full((BLOCK_SIZE,), 0, tl.int64) if n_is_scalar else offsets
        n_int = tl.load(n_ptr + n_offsets, mask=mask, other=0).to(tl.int32)

        h0 = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        h1 = 2.0 * x
        result = tl.where(n_int == 0, h0, h1)
        h_nm1 = h0
        h_n = h1

        h_np1 = _hermite_step(x, h_nm1, h_n, 1)
        result = tl.where(n_int == 2, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 2)
        result = tl.where(n_int == 3, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 3)
        result = tl.where(n_int == 4, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 4)
        result = tl.where(n_int == 5, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 5)
        result = tl.where(n_int == 6, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 6)
        result = tl.where(n_int == 7, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 7)
        result = tl.where(n_int == 8, h_np1, result)
        h_nm1 = h_n
        h_n = h_np1
        h_np1 = _hermite_step(x, h_nm1, h_n, 8)
        result = tl.where(n_int == 9, h_np1, result)

        tl.store(out_ptr + offsets, result, mask=mask)
        pid += grid_0


@triton.jit
def _hermite_scalar_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    degree: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    M: tl.constexpr,
):
    grid_0 = tl.num_programs(0)
    pid = tl.program_id(0)
    while pid < M:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        result = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        if degree == 1:
            result = 2.0 * x
        elif degree >= 2:
            h_nm1 = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
            h_n = 2.0 * x
            for k in tl.static_range(1, 9):
                h_np1 = _hermite_step(x, h_nm1, h_n, k)
                if degree == k + 1:
                    result = h_np1
                h_nm1 = h_n
                h_n = h_np1

        tl.store(out_ptr + offsets, result, mask=mask)
        pid += grid_0


def _check_degree_range(n):
    if isinstance(n, torch.Tensor):
        n_int = n.to(torch.int32)
        n_min = n_int.min().item()
        n_max = n_int.max().item()
        if n_min < 0 or n_max > 9:
            raise ValueError(
                "special_hermite_polynomial_h only supports n in [0, 9], "
                f"got n in [{n_min}, {n_max}]"
            )
    elif isinstance(n, (int, float)):
        if int(n) < 0 or int(n) > 9:
            raise ValueError(
                f"special_hermite_polynomial_h only supports n in [0, 9], got n={n}"
            )
    else:
        raise ValueError("Second argument must be a tensor or scalar")


def hermite_polynomial_h_func(x, n):
    if n.numel() == 1:
        out_shape = x.shape
        n_is_scalar = True
    else:
        x, n = torch.broadcast_tensors(x, n)
        out_shape = x.shape
        n_is_scalar = False

    x_c = x.contiguous()
    n_c = n.contiguous()
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out_c = out.contiguous()
    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    M = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = min(M, TOTAL_CORE_NUM)
    with torch_device_fn.device(x.device):
        _hermite_tensor_kernel[(grid,)](
            x_c, n_c, out_c, n_elements, n_is_scalar, BLOCK_SIZE=BLOCK_SIZE, M=M
        )
    return out_c.view(out_shape)


def hermite_polynomial_h_func_tensor_scalar(x, n):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n_elements = out.numel()
    if n_elements == 0:
        return out.view_as(x)

    BLOCK_SIZE = 1024
    M = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = min(M, TOTAL_CORE_NUM)
    with torch_device_fn.device(x.device):
        _hermite_scalar_kernel[(grid,)](
            x_c, out, n_elements, int(n), BLOCK_SIZE=BLOCK_SIZE, M=M
        )
    return out.view_as(x)


def special_hermite_polynomial_h(x, n):
    logger.debug("GEMS_CAMBRICON SPECIAL_HERMITE_POLYNOMIAL_H")
    if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
        _check_degree_range(n)
        return hermite_polynomial_h_func(x, n)
    elif isinstance(x, torch.Tensor):
        _check_degree_range(n)
        return hermite_polynomial_h_func_tensor_scalar(x, n)
    else:
        raise ValueError("First argument must be a tensor")
