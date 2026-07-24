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
from flag_gems.utils import libentry, libtuner

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 65536}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 131072}, num_stages=1, num_warps=1),
    ],
    key=["n_elements", "is_inplace"],
    restore_value=["output_ptr"],
)
@triton.jit(do_not_specialize=["negative_slope"])
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    is_inplace: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for off in range(block_start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        output = tl.where(x >= 0, x, x * negative_slope)
        tl.store(output_ptr + offsets, output, mask=mask)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_CAMBRICON LEAKY_RELU")
    if not A.is_contiguous():
        A = A.contiguous()
    output = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](
            A, output, n_elements, negative_slope, is_inplace=False
        )
    return output


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_CAMBRICON LEAKY_RELU_")
    if not A.is_contiguous():
        raise RuntimeError(
            "leaky_relu_ requires a contiguous tensor for in-place operation"
        )
    n_elements = A.numel()
    if n_elements == 0:
        return A
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, A, n_elements, negative_slope, is_inplace=True)
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_CAMBRICON LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu(A, negative_slope)
    if not A.is_contiguous():
        A = A.contiguous()
    n_elements = A.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, out, n_elements, negative_slope, is_inplace=False)
    return out
