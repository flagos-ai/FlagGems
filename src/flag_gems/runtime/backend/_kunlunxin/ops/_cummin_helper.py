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

from flag_gems.runtime.backend._kunlunxin.ops.cummin import (
    scan_then_fan,
    scan_then_fan_col,
    scan_then_fan_loop,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _copy_to_out_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    vals = tl.load(src + offset, mask=mask)
    tl.store(dst + offset, vals, mask=mask)


def _write_back(src, dst):
    if dst.is_contiguous():
        n_elements = src.numel()
        if n_elements == 0:
            return
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _copy_to_out_kernel[grid](
            src.reshape(-1), dst.reshape(-1), n_elements, BLOCK_SIZE
        )
    else:
        dst.copy_(src)


def _cummin_helper(input, values, indices, dim):
    """Kunlunxin override of the out-of-place ``aten::_cummin_helper``.

    Mirrors ``_cummax_helper``: reuses the vendor ``_kunlunxin.ops.cummin`` scan
    kernels and always uses the per-row loop kernel for non-1D scans, avoiding
    the generic ``scan_then_fan`` path whose indices are corrupted on XPU3.
    """
    logger.debug("GEMS_KUNLUNXIN CUMMIN_HELPER")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]

    if N == 1:
        # scan over a size-1 axis is the identity.
        if values.dtype == input.dtype:
            values.copy_(input)
        else:
            values.copy_(input.to(values.dtype))
        indices.zero_()
        return None

    input = input.contiguous()
    K = input.numel() // M // N

    compute_dtype = values.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if (
        values.is_contiguous()
        and indices.is_contiguous()
        and values.dtype == input.dtype
    ):
        out = values
        out_indices = indices
        if M == 1 and K == 1:
            scan_then_fan_col(input, out, out_indices, N, compute_dtype)
        elif K == 1:
            scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
        else:
            scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)
        return None

    out = torch.empty_like(input, dtype=values.dtype)
    out_indices = torch.empty_like(input, dtype=torch.int64)
    if M == 1 and K == 1:
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif K == 1:
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)

    _write_back(out, values)
    _write_back(out_indices, indices)
    return None
