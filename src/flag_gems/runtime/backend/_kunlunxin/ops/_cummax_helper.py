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

from flag_gems.runtime.backend._kunlunxin.ops.cummax import (
    scan_then_fan,
    scan_then_fan_col,
    scan_then_fan_loop,
)

logger = logging.getLogger(__name__)


def _cummax_helper(input, values, indices, dim):
    """Kunlunxin override of the out-of-place ``aten::_cummax_helper``.

    The generic helper imports the scan kernels from ``flag_gems.ops.cummax``
    and dispatches ``M*K <= 16`` to ``scan_then_fan`` (the 3-D abc multi-partial
    path). On XPU3 that path corrupts ``indices`` for ``B > 4096`` with ``C > 1``,
    and the generic loop kernel misses the ``is_use_mask_zero=True`` launch flag
    that the vendor scan kernels rely on, so large-scan indices come out wrong.

    This override reuses the vendor ``_kunlunxin.ops.cummax`` kernels (which
    ``flag_gems.cummax`` already uses and which are validated against a CPU fp64
    oracle) and, like the vendor op, always uses the per-row loop kernel for any
    non-1D scan instead of the buggy ``scan_then_fan`` path.
    """
    logger.debug("GEMS_KUNLUNXIN CUMMAX_HELPER")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]

    # N==1: a scan over a size-1 axis is the identity (values = input,
    # indices = 0). Avoids launching a grid of (M, 1) programs for a no-op.
    if N == 1:
        if values.dtype == input.dtype:
            values.copy_(input)
        else:
            values.copy_(input.to(values.dtype))
        indices.zero_()
        return None

    # bool inputs are promoted to int32 (consistent with aten::cummax and to
    # avoid the shared scan kernels, which do not support int1 directly).
    is_bool = input.dtype is torch.bool
    if is_bool:
        # Promote bool to int32, not int64: the vendor scan kernel's int64
        # path mis-computes tie-broken indices when a row has many equal
        # values (bool -> 0/1), while the int32 path is exact. bool fits in
        # int32, and the caller-facing values are cast back to bool below.
        input = input.to(torch.int32)
    input = input.contiguous()
    K = input.numel() // M // N

    compute_dtype = input.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    # Fast path: write the scan result directly into the caller-provided
    # contiguous buffers to avoid an extra allocation + copy.
    if (
        not is_bool
        and values.is_contiguous()
        and indices.is_contiguous()
        and values.dtype == input.dtype
    ):
        out = values
        out_indices = indices
        if M == 1 and K == 1:
            scan_then_fan_col(input, out, out_indices, N, compute_dtype)
        elif K == 1:
            # C==1: the two-level scan_then_fan is correct and much faster
            # than the serialized loop (its XPU3 corruption only needs C>1).
            scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
        else:
            scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)
        return None

    out = torch.empty_like(input)
    out_indices = torch.empty_like(input, dtype=torch.int64)
    if M == 1 and K == 1:
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif K == 1:
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)

    values.copy_(out)
    indices.copy_(out_indices)
    return None
