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
import math

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .cummax import (
    scan_part_max_abc_loop_key_kernel,
    scan_then_fan,
    scan_then_fan_col,
    scan_then_fan_loop,
)

logger = logging.getLogger(__name__)

# 16-bit dtypes whose loop-path scan benefits from a wider BLOCK on GCU400.
# (uint16 is left out on purpose: no CPU reference and no gate covers it, so
# it keeps the untouched shared path.)
_WIDE_BLOCK_DTYPES = (torch.int16, torch.float16, torch.bfloat16)


def _scan_then_fan_loop_tuned(inp, out, out_indices, A, B, C, dtype):
    """Loop-path scan with a GCU400-specific BLOCK policy for 16-bit dtypes.

    The shared launcher caps BLOCK at 4096. For 16-bit inputs that cap trips
    a narrow-load pathology in the multi-chunk regime: the 2B lanes stall the
    int64 key pipeline and the per-element cost inflates ~2.6x versus the
    fp32/int32 cells (measured at (1024, 65536): fp16 66.4ms @4096 vs
    23.2ms @16384, int16 62.8ms -> 23.7ms, 2.86x/2.65x; 32-bit dtypes are
    flat there, 1.05x, so they keep the shared 4096 path). BLOCK=16384
    keeps the int64 key vector at 128KB, well inside the 917504B DSM
    budget (compiled with 0 register spills).
    """
    if B > 4096 and inp.dtype in _WIDE_BLOCK_DTYPES:
        block_size = min(16384, triton.next_power_of_2(B))
        loop_num = math.ceil(B / block_size)
        grid = (min(A, 65535), min(C, 255))
        with torch_device_fn.device(inp.device):
            scan_part_max_abc_loop_key_kernel[grid](
                inp,
                out,
                out_indices,
                A,
                B,
                C,
                loop_num,
                block_size,
                inp.dtype in (torch.float16, torch.bfloat16, torch.float32),
                num_warps=1,
            )
        return
    scan_then_fan_loop(inp, out, out_indices, A, B, C, dtype)


def _cummax_helper(input, values, indices, dim):
    """Helper for ``cummax`` that writes the cumulative maximum and the
    index of each maximum value into the pre-allocated ``values`` and
    ``indices`` tensors along the given ``dim``.

    This is the out-of-place kernel used by ``aten::cummax`` /
    ``aten::cummax.out``.  When the caller-provided ``values`` / ``indices``
    buffers are contiguous (the common case), the scan kernels write directly
    into them; otherwise a contiguous staging buffer is used and copied back.
    """
    logger.debug("GEMS_ENFLAME CUMMAX_HELPER")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    # bool inputs are promoted to int64 (consistent with aten::cummax and to
    # avoid the shared scan kernels, which do not support int1 directly).
    is_bool = input.dtype is torch.bool
    if is_bool:
        input = input.to(torch.int64)
    input = input.contiguous()
    K = input.numel() // M // N

    compute_dtype = input.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    # Fast path: write the scan result directly into the caller-provided
    # contiguous buffers to avoid an extra allocation + copy. bool inputs
    # take the general path because their values are computed in the
    # promoted int64 dtype.
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
        elif M * K <= 16:
            scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
        else:
            _scan_then_fan_loop_tuned(input, out, out_indices, M, N, K, compute_dtype)
        return None

    # General path: compute into freshly allocated contiguous tensors (so the
    # scan kernels, which assume contiguous storage, work for any caller-provided
    # out layout/dtype), then copy the results back into the caller-provided
    # tensors, which may be non-contiguous or use the original (un-promoted)
    # value dtype.
    out = torch.empty_like(input)
    out_indices = torch.empty_like(input, dtype=torch.int64)

    if M == 1 and K == 1:
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif M * K <= 16:
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        _scan_then_fan_loop_tuned(input, out, out_indices, M, N, K, compute_dtype)

    values.copy_(out)
    indices.copy_(out_indices)
