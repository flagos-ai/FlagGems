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

from .cummin import (
    add_base_min_abc_kernel,
    add_base_min_kernel,
    scan_part_min_abc_kernel,
    scan_part_min_abc_loop_kernel,
    scan_part_min_kernel,
)

logger = logging.getLogger(__name__)


def _mask_kw(n_ele, BLOCK_SIZE):
    return {} if n_ele % BLOCK_SIZE == 0 else {"is_use_mask_zero": True}


def _scan_then_fan_col(inp, out, out_indices, n_ele, dtype, use_out_indices=False):
    """1D scan-then-fan; identical to the vendor host plus the mask-zero gate."""
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    need_partial = part_num >= 2
    if need_partial:
        partial_min = torch.empty(part_num, dtype=dtype, device=inp.device)
        partial_min_indices = torch.empty(
            part_num, dtype=torch.int64, device=inp.device
        )
    else:
        partial_min = None
        partial_min_indices = None

    mask_kw = _mask_kw(n_ele, BLOCK_SIZE)
    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_min_kernel[grid](
            inp,
            out,
            out_indices,
            out_indices,
            partial_min,
            partial_min_indices,
            n_ele,
            BLOCK_SIZE,
            need_partial,
            use_out_indices,
            **mask_kw,
        )

    if need_partial:
        _scan_then_fan_col(
            partial_min,
            partial_min,
            partial_min_indices,
            part_num,
            dtype,
            use_out_indices=True,
        )
        with torch_device_fn.device(inp.device):
            add_base_min_kernel[grid](
                out,
                out_indices,
                partial_min,
                partial_min_indices,
                n_ele,
                BLOCK_SIZE,
                **mask_kw,
            )


def _scan_then_fan_abc(inp, out, out_indices, A, B, C, dtype, use_out_indices=False):
    """3D scan-then-fan; identical to the vendor host plus the mask-zero gate."""
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    need_partial = part_num >= 2
    if need_partial:
        partial_min = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)
        partial_min_indices = torch.empty(
            A, part_num, C, dtype=torch.int64, device=inp.device
        )
    else:
        partial_min = None
        partial_min_indices = None

    mask_kw = _mask_kw(B, BLOCK_SIZE)
    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_min_abc_kernel[grid](
            inp,
            out,
            out_indices,
            out_indices,
            partial_min,
            partial_min_indices,
            B,
            C,
            part_num,
            BLOCK_SIZE,
            need_partial,
            use_out_indices,
            **mask_kw,
        )

    if need_partial:
        _scan_then_fan_abc(
            partial_min,
            partial_min,
            partial_min_indices,
            A,
            part_num,
            C,
            dtype,
            use_out_indices=True,
        )
        with torch_device_fn.device(inp.device):
            add_base_min_abc_kernel[grid](
                out,
                out_indices,
                partial_min,
                partial_min_indices,
                B,
                C,
                part_num,
                BLOCK_SIZE,
                **mask_kw,
            )


def _scan_then_fan_loop(inp, out, out_indices, A, B, C, dtype):
    """Loop host; identical to ``cummin.scan_then_fan_loop`` (which hardcodes
    ``is_use_mask_zero=True``) except that the flag is gated on alignment."""
    BLOCK_SIZE = 1024
    if B < 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    loop_num = math.ceil(B / BLOCK_SIZE)

    with torch_device_fn.device(inp.device):
        scan_part_min_abc_loop_kernel[(A, C)](
            inp,
            out,
            out_indices,
            B,
            C,
            loop_num,
            BLOCK_SIZE,
            **_mask_kw(B, BLOCK_SIZE),
        )


def _dispatch_scan(inp, out, out_indices, M, N, K, compute_dtype):
    if M == 1 and K == 1:
        _scan_then_fan_col(inp, out, out_indices, N, compute_dtype)
    elif M * K <= 16 or inp.dtype is torch.int64:
        # int64 must avoid the loop kernel: its tl.sum carry miscompiles for
        # 64-bit payloads (736972/819980 values wrong at (20, 40999, 1)).
        _scan_then_fan_abc(inp, out, out_indices, M, N, K, compute_dtype)
    else:
        # Same host as the vendor's, with the mask-zero flag gated on alignment.
        _scan_then_fan_loop(inp, out, out_indices, M, N, K, compute_dtype)


def _cummin_helper(input, values, indices, dim):
    logger.debug("GEMS_KUNLUNXIN CUMMIN_HELPER")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    # Empty tensors: nothing to scan, and numel() // M // N would divide by 0.
    if input.numel() == 0:
        return None

    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = input.contiguous()
    K = inp.numel() // M // N

    is_bool = inp.dtype is torch.bool
    if is_bool:
        # Zero-copy reinterpretation: bool is a single 0/1 byte and cummin over
        # {0, 1} stays in {0, 1}, so the int8 result is already a valid bool
        # buffer.  The loop kernel cannot be compiled for int1 at all.
        inp = inp.view(torch.int8)

    compute_dtype = inp.dtype
    if compute_dtype is torch.float16 or compute_dtype is torch.bfloat16:
        compute_dtype = torch.float32

    if (
        values.is_contiguous()
        and indices.is_contiguous()
        and values.dtype == input.dtype
        and indices.dtype is torch.int64
    ):
        out = values.view(torch.int8) if is_bool else values
        _dispatch_scan(inp, out, indices, M, N, K, compute_dtype)
        return None

    out = torch.empty_like(inp)
    out_indices = torch.empty_like(inp, dtype=torch.int64)
    _dispatch_scan(inp, out, out_indices, M, N, K, compute_dtype)
    values.copy_(out.view(torch.bool) if is_bool else out)
    indices.copy_(out_indices)
    return None
