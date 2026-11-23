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

"""GCU400-specific cumulative_trapezoid.

The common implementation is a single kernel with "one program per row, tl.cumsum along the integration dimension".
This implementation keeps that path as a baseline switch and adds two optimizations:

* ``KERNEL_MODE=1``: combine the integration dimension and the inner dimension into a 2D tile, processing several rows
of adjacent
  inner elements at once so memory access is contiguous along inner and program count is reduced;
* ``FUSE_DX=1``: fold dx scaling into the scan's trapezoid term, removing the common implementation's extra
  elementwise multiply launch when dx != 1. ``DX_QUANTIZE`` reproduces the common implementation's
  "quantize to output dtype first, then multiply by dx" two-step rounding for elementwise equivalence.

All tuning switches are module-level constants rewritten by the Step 4 round script.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ===== Tuning constants rewritten by the Step 4 round script =====
BASELINE_MODE = 0
KERNEL_MODE = 2
BLOCK_I = 32
NUM_WARPS = 1
FUSE_DX = 1
DX_QUANTIZE = 1
ROWS = 8
ROWS_MODE = 1
MAX_VECTOR = 1024
BLOCK_LIMIT = 128
WARPS_MODE = 1
WARPS_SMALL = 1
WARPS_LARGE = 4
WARPS_SPLIT = 64
WARPS_1D = 1
# ========================================


@triton.jit
def _cumtrapz_baseline_kernel(
    y_ptr,
    out_ptr,
    N,
    inner,
    stride_row_in,
    stride_row_out,
    BLOCK: tl.constexpr,
):
    """Per-row scan of the common implementation, used as a comparable baseline."""
    pid = tl.program_id(0)
    o = pid // inner
    i = pid % inner
    in_base = o * stride_row_in + i
    out_base = o * stride_row_out + i

    k = tl.arange(0, BLOCK)
    mask = k < (N - 1)

    a = tl.load(y_ptr + in_base + k * inner, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + in_base + (k + 1) * inner, mask=mask, other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    tl.store(out_ptr + out_base + k * inner, cs, mask=mask)


@triton.jit
def _cumtrapz_1d_kernel(
    y_ptr,
    out_ptr,
    N,
    inner,
    stride_row_in,
    stride_row_out,
    DX: tl.constexpr,
    FUSE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-row scan, optionally folding dx into the kernel (including quantization emulation)."""
    pid = tl.program_id(0)
    o = pid // inner
    i = pid % inner
    in_base = o * stride_row_in + i
    out_base = o * stride_row_out + i

    k = tl.arange(0, BLOCK)
    mask = k < (N - 1)
    a = tl.load(y_ptr + in_base + k * inner, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + in_base + (k + 1) * inner, mask=mask, other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    if FUSE:
        if QUANTIZE:
            # The common implementation first writes the cumsum result into an output-dtype tensor, then does one
            # elementwise multiply by dx;
            # here both roundings are reproduced in registers so fusion does not change low-precision numeric semantics.
            cs = cs.to(OUT_DTYPE).to(tl.float32)
        cs = cs * DX
    tl.store(out_ptr + out_base + k * inner, cs, mask=mask)


@triton.jit
def _cumtrapz_2d_kernel(
    y_ptr,
    out_ptr,
    N,
    inner,
    stride_row_in,
    stride_row_out,
    DX: tl.constexpr,
    FUSE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """2D tile: rows are the integration dimension k, columns are contiguous inner elements.

    One program covers ``BLOCK_N * BLOCK_I`` output elements; the cumsum along the integration dimension
    is independent per column (axis=0), so it is mathematically equivalent to the per-row scan.
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    k = tl.arange(0, BLOCK_N)
    i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    valid = (k[:, None] < (N - 1)) & (i[None, :] < inner)

    # Input and output are both contiguous along inner; the row stride is determined by the integration length.
    in_base = y_ptr + pid_o * stride_row_in
    out_base = out_ptr + pid_o * stride_row_out
    in_ptrs = in_base + k[:, None] * inner + i[None, :]
    a = tl.load(in_ptrs, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(in_ptrs + inner, mask=valid, other=0.0).to(tl.float32)

    cs = tl.cumsum((a + b) * 0.5, axis=0)
    if FUSE:
        if QUANTIZE:
            cs = cs.to(OUT_DTYPE).to(tl.float32)
        cs = cs * DX

    tl.store(out_base + k[:, None] * inner + i[None, :], cs, mask=valid)


@triton.jit
def _cumtrapz_rows_kernel(
    y_ptr,
    out_ptr,
    N,
    inner,
    rows_total,
    DX: tl.constexpr,
    FUSE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    INNER_EQ1: tl.constexpr,
):
    """Merged multi-row scan: one program handles ROWS logical rows.

    The common implementation uses one program per logical row (one outer and inner combination), so small N
    creates many programs computing only a dozen elements each, with scheduling overhead dominating. Here ROWS rows are
    flattened into
    one 1D vector loaded at once, reshaped to [ROWS, BLOCK] for independent per-row cumsum, then written back.
    Only 1D masked load/store is used throughout, avoiding GCU400's 2D gather compilation limits.
    ``INNER_EQ1`` specializes the common "integration dimension last" case to division-free address computation.
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, ROWS * BLOCK)
    row = pid * ROWS + idx // BLOCK
    k = idx % BLOCK
    valid = (row < rows_total) & (k < (N - 1))

    if INNER_EQ1:
        in_off = row * N + k
        out_off = row * (N - 1) + k
        step = 1
    else:
        o = row // inner
        i = row % inner
        in_off = o * N * inner + k * inner + i
        out_off = o * (N - 1) * inner + k * inner + i
        step = inner

    a = tl.load(y_ptr + in_off, mask=valid, other=0.0).to(tl.float32)
    # With k <= N-2, in_off+step is still the last input element of the current logical row and does not spill into the
    # next row.
    b = tl.load(y_ptr + in_off + step, mask=valid, other=0.0).to(tl.float32)
    t = (a + b) * 0.5

    cs = tl.cumsum(tl.reshape(t, (ROWS, BLOCK)), axis=1)
    flat = tl.reshape(cs, (ROWS * BLOCK,))
    if FUSE:
        if QUANTIZE:
            flat = flat.to(OUT_DTYPE).to(tl.float32)
        flat = flat * DX

    tl.store(out_ptr + out_off, flat, mask=valid)


def _dtype_constexpr(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64
    return tl.float32


def cumulative_trapezoid(y, *, dx=1, dim=-1):
    logger.debug("GEMS_ENFLAME_GCU400 CUMULATIVE_TRAPEZOID")

    assert y.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"cumulative_trapezoid: unsupported dtype {y.dtype}"

    ndim = y.dim()
    d = dim if dim >= 0 else dim + ndim
    sizes = list(y.shape)
    N = sizes[d]

    outer = 1
    for s in sizes[:d]:
        outer *= s
    inner = 1
    for s in sizes[d + 1 :]:
        inner *= s

    out_sizes = list(sizes)
    out_sizes[d] = max(N - 1, 0)
    output = torch.empty(out_sizes, dtype=y.dtype, device=y.device)

    if N <= 1 or outer * inner == 0:
        return output if dx == 1 else output * dx

    yc = y.contiguous()
    stride_row_in = N * inner
    stride_row_out = (N - 1) * inner
    block_n = triton.next_power_of_2(max(N - 1, 1))

    if BASELINE_MODE:
        _cumtrapz_baseline_kernel[(outer * inner,)](
            yc,
            output,
            N,
            inner,
            stride_row_in,
            stride_row_out,
            BLOCK=block_n,
        )
        return output if dx == 1 else output * dx

    fuse_dx = bool(FUSE_DX and dx != 1)
    quantize = bool(DX_QUANTIZE)
    out_dtype = _dtype_constexpr(y.dtype)

    # Short integration dimensions benefit significantly from 1 warp by reducing scheduling overhead; long integration
    # dimensions (BLOCK >= 128)
    # degrade under 1 warp because a single program processes too many serial elements, so multiple warps are required.
    if WARPS_MODE == 1:
        warps_rows = WARPS_SMALL if block_n <= WARPS_SPLIT else WARPS_LARGE
        warps_other = WARPS_1D
    else:
        warps_rows = NUM_WARPS
        warps_other = NUM_WARPS

    if KERNEL_MODE == 2 and block_n <= BLOCK_LIMIT:
        # Total logical rows = outer * inner; the adaptive row count first shrinks by the target program count,
        # then MAX_VECTOR caps the 1D vector length per program to avoid
        # register spilling when ROWS*BLOCK is too large at big N (measured N=1024 with ROWS=4 is 10x slower).
        rows_total = outer * inner
        if ROWS_MODE == 1:
            rows_eff = max(1, min(ROWS, rows_total // 256))
        else:
            rows_eff = ROWS
        if MAX_VECTOR > 0 and rows_eff * block_n > MAX_VECTOR:
            rows_eff = max(1, MAX_VECTOR // block_n)
        _cumtrapz_rows_kernel[(triton.cdiv(rows_total, rows_eff),)](
            yc,
            output,
            N,
            inner,
            rows_total,
            float(dx),
            fuse_dx,
            quantize,
            out_dtype,
            rows_eff,
            block_n,
            inner == 1,
            num_warps=warps_rows,
        )
    elif KERNEL_MODE == 1:
        grid = (outer, triton.cdiv(inner, BLOCK_I))
        _cumtrapz_2d_kernel[grid](
            yc,
            output,
            N,
            inner,
            stride_row_in,
            stride_row_out,
            float(dx),
            fuse_dx,
            quantize,
            out_dtype,
            block_n,
            BLOCK_I,
            num_warps=warps_other,
        )
    else:
        _cumtrapz_1d_kernel[(outer * inner,)](
            yc,
            output,
            N,
            inner,
            stride_row_in,
            stride_row_out,
            float(dx),
            fuse_dx,
            quantize,
            out_dtype,
            block_n,
            num_warps=warps_other,
        )

    if fuse_dx:
        return output
    return output if dx == 1 else output * dx
