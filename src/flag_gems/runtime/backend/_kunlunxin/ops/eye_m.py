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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
device_ = device


@libentry()
@triton.jit
def eye_diagonal_kernel(
    out_ptr,
    diag_len,
    stride,
    BLOCK: tl.constexpr,
):
    # The N x M identity matrix is dominated by zeros; only the min(N, M)
    # diagonal entries are 1. Instead of launching one program per 32x32 tile
    # over the whole matrix (launch-bound: ~0.001x on XPU), the buffer is bulk
    # zero-filled by torch.zeros (vendor memset, ~1000+ GB/s) and this kernel
    # writes only the diagonal ones. Diagonal element k of a contiguous (N, M)
    # tensor is at flat index k * M + k = k * (M + 1) = k * stride.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < diag_len
    tl.store(out_ptr + offs * stride, 1, mask=mask)


def _fill_diagonal(out, n, m):
    diag_len = min(n, m)
    if diag_len <= 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(diag_len, BLOCK),)
    with torch_device_fn.device(out.device):
        eye_diagonal_kernel[grid](out, diag_len, m + 1, BLOCK)
    return out


def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye(n, m): bulk zero-fill + diagonal write.
    """
    logger.debug("GEMS_KUNLUNXIN EYE_M")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye_m.")

    out = torch.zeros(
        (n, m), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )
    return _fill_diagonal(out, n, m)
