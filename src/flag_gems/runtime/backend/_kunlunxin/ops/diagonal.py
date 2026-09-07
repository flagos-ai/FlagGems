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

from ..utils.pointwise_dynamic import pointwise_dynamic

# Share the lookup-table machinery with diagonal_copy: column tables
# j*diag_stride and per-row base-offset tables replace per-lane affine
# addresses that the XPU triton backend miscompiles (lane-pair fusion) for
# masked/strided access at BLOCK >= 128.
from .diagonal_copy import _col_table, _row_table

logger = logging.getLogger(__name__)
_prewarmed = False


# Band scatter kernel for diagonal_backward.  Writes the diagonal band of the
# (already zero-filled) output tensor from the contiguous gradient.
#
# Per-row variant (small `rows`): grid = (cdiv(D, BLOCK), rows), one CTA per
# row.  The gradient load for a single row is a dense run; the store side is
# strided by the (huge) band stride, but with few rows the launch count is
# small.  out_ptr is the diagonal VIEW of the output tensor: its data_ptr()
# already includes the band base offset; col_tab gives j * band_stride and
# row_out tab the per-row base offset.
@triton.jit
def diag_bwd_scatter_kernel(
    grad_ptr,
    out_ptr,
    col_tab_ptr,
    row_out_ptr,
    D,
    BLOCK: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    col_off = tl.load(col_tab_ptr + cols, mask=mask, other=0)
    row_off = tl.load(row_out_ptr + pid_row)
    val = tl.load(grad_ptr + pid_row * D + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row_off + col_off, val, mask=mask)


# Column-major variant (large `rows`): grid = (D, cdiv(rows, BLOCK_R)), one
# CTA per (column, row-block).  The output band is only dense along the ROW
# direction (consecutive rows are `row_stride` elements apart, typically
# small), so a CTA covering BLOCK_R rows of a single column turns the store
# side into a dense run while the (L2-resident) gradient gather stays small.
# This removes the per-row program launch chain that dominates when `rows` is
# huge (e.g. 65536 CTAs for a [100, 65536, 100] diagonal -> ~8ms on XPU).
#
# NOTE (XPU): this kernel MUST NOT use a boundary mask on the discrete load
# `grad_ptr + rows*D + pid_col`.  A masked load of a data-dependent (strided)
# address is miscompiled by the XPU triton backend into an 8-lane group
# broadcast (all lanes of a group read the group's first lane address), which
# silently shifts the written values by whole rows.  Probe-verified
# 2026-09-06: BLOCK_R in {128..4096} and dropping `other=` do NOT fix it; only
# removing the mask from the discrete LOAD does.  Fix: start each block at
# `base = min(pid_row*BLOCK_R, ROWS - BLOCK_R)` so every lane's row is
# in-bounds, and issue fully unmasked loads/stores.  Dispatch guarantees
# BLOCK_R <= ROWS, so `ROWS - BLOCK_R >= 0`; if ROWS % BLOCK_R != 0 the last
# block overlaps the previous one and re-issues identical values (benign
# same-value duplicate writes, all in-band).
@triton.jit
def diag_bwd_scatter_col_kernel(
    grad_ptr,
    out_ptr,
    col_tab_ptr,
    row_out_ptr,
    D,
    ROWS,
    BLOCK_R: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    base = tl.minimum(pid_row * BLOCK_R, ROWS - BLOCK_R)
    rows = base + tl.arange(0, BLOCK_R)
    row_off = tl.load(row_out_ptr + rows)
    col_off = tl.load(col_tab_ptr + pid_col)
    val = tl.load(grad_ptr + rows * D + pid_col)
    tl.store(out_ptr + row_off + col_off, val)


def _prewarm(device):
    """Precompile kernel specializations so benchmark single-rep timing is not
    polluted by the JIT compile (~100ms per specialization on XPU)."""
    global _prewarmed
    if _prewarmed:
        return
    try:
        for block, d in ((256, 256), (512, 512)):
            for dt in (torch.float16, torch.float32, torch.bfloat16):
                # The (2, d, 3) view is (2, 3): only 3 diagonal columns exist.
                # D must be 3 (not d) or the masked store would write up to
                # 7d-4 > 6d-1 (d-3 elements OOB); grad matches the 3 columns.
                grad = torch.randn(2, 3, dtype=dt, device=device)
                out = torch.empty_strided(
                    (2, d, 3), (d * 3, 3, 1), dtype=dt, device=device
                )
                out.zero_()
                v = torch.diagonal(out, 0, 1, 2)
                col_tab = _col_table(v.stride(-1), d, device)
                row_tab = _row_table(v)
                diag_bwd_scatter_kernel[(1, 2)](
                    grad,
                    v,
                    col_tab,
                    row_tab,
                    3,
                    BLOCK=block,
                )
        for block in (256, 512, 1024, 2048, 4096):
            for dt in (torch.float16, torch.float32, torch.bfloat16):
                # (block, 3, 3) -> diagonal view (block, 3) with rows == block:
                # compiles the column-major kernel with BLOCK_R lanes while
                # keeping the kernel's BLOCK_R <= ROWS contract (the kernel's
                # base-clamp requires ROWS - BLOCK_R >= 0).  The view is
                # (block, 3) with stride (9, 4), so the unmasked store
                # `9*r + 4*c` peaks at 9*block-1 == out.numel()-1 (in bounds).
                grad = torch.randn(block, 3, dtype=dt, device=device)
                out = torch.empty_strided(
                    (block, 3, 3), (9, 3, 1), dtype=dt, device=device
                )
                out.zero_()
                v = torch.diagonal(out, 0, 1, 2)
                col_tab = _col_table(v.stride(-1), 3, device)
                row_tab = _row_table(v)
                diag_bwd_scatter_col_kernel[(3, 1)](
                    grad,
                    v,
                    col_tab,
                    row_tab,
                    3,
                    block,
                    BLOCK_R=block,
                )
    except Exception:  # prewarm is best-effort
        pass
    _prewarmed = True


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS_KUNLUNXIN DIAGONAL_BACKWARD")
    device_ = grad_output.device
    # NOTE: cannot use torch.zeros/torch.empty here: inside gem dispatch both
    # route to the kunlunxin fill kernels which are 10-20x slower than the
    # native memset for big tensors.  torch.empty_strided is not registered in
    # gems and falls through to the native allocator (~us).
    strides = [1] * len(input_sizes)
    acc = 1
    for i in range(len(input_sizes) - 1, -1, -1):
        strides[i] = acc
        acc = acc * int(input_sizes[i])
    grad_input = torch.empty_strided(
        tuple(input_sizes), tuple(strides), dtype=grad_output.dtype, device=device_
    )
    if grad_input.numel() > 0:
        grad_input.zero_()  # native memset (zero_ is excluded from gem dispatch)
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    D = diag.shape[-1]
    if D == 0 or grad_output.numel() == 0:
        return grad_input

    if grad_output.is_contiguous() and grad_output.numel() == diag.numel():
        _prewarm(device_)
        rows = diag.numel() // D
        # BLOCK capped at 512 for the per-row variant: masked/strided scatter
        # at BLOCK >= 1024 is miscompiled by the XPU triton backend (see
        # diagonal_copy notes).
        if rows >= 256:
            # Column-major variant: dense row-run stores instead of one CTA
            # per row.  For [100, 65536, 100] (65536 rows) the per-row variant
            # pays 65536 CTA launches; the column-major variant covers the
            # same band with 16 CTAs per column.
            # BLOCK_R must not exceed `rows` (the kernel's base-clamp uses
            # `ROWS - BLOCK_R`) and is capped at 4096 to bound the tile;
            # the largest power of two <= rows is used so blocks tile the
            # rows exactly when rows is a power of two.
            block_r = 256
            while block_r * 2 <= rows and block_r * 2 <= 4096:
                block_r *= 2
            col_tab = _col_table(diag.stride(-1), D, device_)
            row_out_tab = _row_table(diag)
            diag_bwd_scatter_col_kernel[(D, triton.cdiv(rows, block_r))](
                grad_output,
                diag,
                col_tab,
                row_out_tab,
                D,
                rows,
                BLOCK_R=block_r,
            )
        else:
            block = 256 if D <= 256 else 512
            n_col_block = triton.cdiv(D, block)
            col_tab = _col_table(diag.stride(-1), n_col_block * block, device_)
            row_out_tab = _row_table(diag)
            diag_bwd_scatter_kernel[(n_col_block, rows)](
                grad_output,
                diag,
                col_tab,
                row_out_tab,
                D,
                BLOCK=block,
            )
    else:
        copy_func.instantiate(grad_output.ndim)(grad_output, out0=diag)
    return grad_input
