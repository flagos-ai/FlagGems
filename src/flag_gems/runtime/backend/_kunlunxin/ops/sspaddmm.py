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
"""Kunlunxin XPU3 override for ``sspaddmm`` / ``sspaddmm_out``.

The generic implementation picks ``BLOCK_N = min(next_pow2(N), 1024)``.  On
XPU3 the row kernel keeps a loop-carried 1-D accumulator over the sparse row's
nonzeros, and ``TritonXPUUnrollControl`` only accepts a tile of **<=64 lanes or
exactly 1024 lanes**; 128/256/512 exhaust ``uni_sram`` at compile time
(``OutOfResources: OutOfResources: out of resource: uni_sram``).  Measured on
this kernel: N=128 fails, N<=64 compiles, 1024 compiles.  We therefore pick
``next_pow2(N)`` for N<=64 and a masked 1024-wide tile otherwise (measured
~16x faster than the 64-lane fallback on N=1024).

The kernel itself is reused verbatim from the generic implementation; only the
column-tile selection is backend-local.  The host composition below mirrors the
generic ``_sspaddmm_impl`` one-to-one (index/metadata preprocessing remains
native device work; the SpMM numeric contraction is the Triton row kernel).
"""

import logging

import torch
import triton

from flag_gems.ops.sspaddmm import _sspaddmm_row_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# XPU3-safe loop-carried 1-D accumulator widths: <=64 lanes, or exactly 1024.
_MAX_BLOCK_N = 64
_WIDE_BLOCK_N = 1024


def _pick_block_n(N, dtype):
    npow2 = triton.next_power_of_2(N) if N > 0 else 1
    # fp16 cannot lower a 1024-lane tile for this kernel (TritonXPUUnrollControl
    # / uni_sram); it stays on the <=64-lane path.
    if npow2 <= _MAX_BLOCK_N:
        return npow2
    if dtype == torch.float16:
        return _MAX_BLOCK_N  # 64-lane grid-tiled fallback
    return _WIDE_BLOCK_N


def _sspaddmm_impl(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
    assert input.is_sparse, "sspaddmm: input must be a sparse COO tensor"
    assert mat1.is_sparse, "sspaddmm: mat1 must be a sparse COO tensor"
    assert not mat2.is_sparse, "sspaddmm: mat2 must be a dense tensor"

    input = input.coalesce()
    mat1 = mat1.coalesce()

    n = mat1.shape[0]
    N = mat2.shape[1]
    dtype = input.dtype
    device = input.device

    # Keep the whole reduce/accumulate pipeline in a wider dtype for low
    # precision inputs, casting back to `dtype` only at the very end.
    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    acc_dtype = torch.float32 if is_low_prec else dtype

    mat2_c = mat2.contiguous()

    mat1_indices = mat1.indices()
    mat1_rows = mat1_indices[0]
    mat1_cols = mat1_indices[1].contiguous()
    mat1_vals = mat1.values().contiguous()

    # Group mat1 nonzeros by row (coalesced COO is sorted row-major).
    if mat1_rows.numel() == 0:
        active_rows = mat1_rows.new_empty((0,))
        counts = mat1_rows.new_empty((0,))
    else:
        active_rows, counts = torch.unique_consecutive(mat1_rows, return_counts=True)
    R = active_rows.numel()

    if R > 0:
        row_end = counts.cumsum(0)
        row_start = row_end - counts
        row_start = row_start.to(torch.int64).contiguous()
        row_end = row_end.to(torch.int64).contiguous()

        dense = torch.empty((R, N), dtype=acc_dtype, device=device)

        # Backend-local tile choice: see module docstring.
        BLOCK_N = _pick_block_n(N, dtype)
        grid = (R, triton.cdiv(N, BLOCK_N) if N > 0 else 1)

        with torch_device_fn.device(device):
            _sspaddmm_row_kernel[grid](
                row_start,
                row_end,
                mat1_cols,
                mat1_vals,
                mat2_c,
                dense,
                alpha,
                N,
                mat2_c.stride(0),
                mat2_c.stride(1),
                IS_FP64=dtype == torch.float64,
                BLOCK_N=BLOCK_N,
            )

        # Build indices for the mm contribution: every (active_row, col) pair.
        row_idx = active_rows.to(torch.int64).repeat_interleave(N)
        col_idx = torch.arange(N, device=device, dtype=torch.int64).repeat(R)
        mm_indices = torch.stack([row_idx, col_idx], dim=0)
        mm_values = dense.reshape(-1)
    else:
        mm_indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        mm_values = torch.empty((0,), dtype=acc_dtype, device=device)

    # beta * input contribution, scaled in the wider accumulation dtype.
    input_indices = input.indices().to(torch.int64)
    input_values = input.values().to(acc_dtype)
    beta_values = input_values * beta if beta != 1.0 else input_values

    out_indices = torch.cat([input_indices, mm_indices], dim=1)
    out_values = torch.cat([beta_values, mm_values], dim=0)

    result = torch.sparse_coo_tensor(
        out_indices, out_values, size=(n, N), dtype=acc_dtype, device=device
    ).coalesce()
    if acc_dtype != dtype:
        result = torch.sparse_coo_tensor(
            result.indices(),
            result.values().to(dtype),
            size=(n, N),
            dtype=dtype,
            device=device,
        ).coalesce()

    if out is not None:
        out.copy_(result)
        return out
    return result


def sspaddmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS_SSPADDMM")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sspaddmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_SSPADDMM_OUT")
    if out is None:
        raise TypeError("sspaddmm(): out must be provided for the out variant")
    return _sspaddmm_impl(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
