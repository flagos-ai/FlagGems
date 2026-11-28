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
import triton.language as tl

_FILL_BLOCK = 8192
# When the number of output elements does not exceed this threshold, use single-kernel fusion (fill+patch merged) to
# save one launch.
# In pure kernel timing, fused and non-fused are roughly on par up to 128K elements, beyond which fused is slower
# (it must do div/mod for every element), but in real calls the fixed overhead of two launches outweighs this
# difference,
# so the threshold stays at 2M, switching back to fill+patch only for larger sizes.
_FUSED_MAX_OUT = 1 << 21
# When outer==1, use the spanfill single kernel (one pass over the whole output: copy grad into the span, write 0
# elsewhere).
# Works for any size with 4-byte dtypes (measured: (2^30,) fp32 1385→1326, [1024³]d0 1410→1328);
# for 2-byte dtypes on outputs ≥2GiB this kernel has a steady ~+30µs codegen tax (fill 672→703,
# interleaved retest over 4 rounds ±0.5µs), so only allow outputs ≤1GiB on this path (67MB-scale fp16 still wins in
# measurements:
# [4096²]d0 15.6→14.1, (64,512,512)d0 16.8→14.1).
_SPAN_BLOCK = 8192
_SPAN_MAX_BYTES = 1 << 30
# Total-element range for the int32-safe-domain version of the flat patch: in 32768..2^20, pb=4096 is uniformly better
# than
# the original flat (E-case patch 5.7→4.2µs, F-case 6.9→5.9, [1024³]d2 93→91.5);
# for smaller totals the original flat (more grid, smaller per-program work) is faster; for larger totals revert to the
# original flat to avoid a CTA tsunami.
_FLAT_I32_MIN = 1 << 15
_FLAT_I32_MAX = 1 << 20
logger = logging.getLogger(__name__)


@triton.jit
def _select_bwd_fill_kernel(out_ptr, n_out, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if WIDE:
        base0 = pid.to(tl.int64) * BLOCK
        step = nprog.to(tl.int64) * BLOCK
        for off0 in tl.range(base0, n_out, step):
            offs = off0 + tl.arange(0, BLOCK)
            tl.store(
                out_ptr + offs,
                tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
                mask=offs < n_out,
            )
    else:
        for off0 in tl.range(pid * BLOCK, n_out, nprog * BLOCK):
            offs = off0 + tl.arange(0, BLOCK)
            tl.store(
                out_ptr + offs,
                tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
                mask=offs < n_out,
            )


@triton.jit
def _select_bwd_patch_rowblock_kernel(
    grad_ptr,
    out_ptr,
    outer,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    npr = tl.num_programs(0)
    pid_b = tl.program_id(1)
    npb = tl.num_programs(1)
    for row in tl.range(pid_r, outer, npr):
        row64 = row.to(tl.int64)
        row_base = row64 * dim_stride
        for g0 in tl.range(pid_b * BLOCK, inner_size, npb * BLOCK):
            gidx = g0 + tl.arange(0, BLOCK)
            m = gidx < inner_size
            vals = tl.load(grad_ptr + row64 * inner_size + gidx, mask=m)
            tl.store(out_ptr + row_base + lo + gidx, vals, mask=m)


@triton.jit
def _select_bwd_patch_flat_kernel(
    grad_ptr,
    out_ptr,
    total,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    safe_outer = tl.where(mask, offs // inner_size, 0).to(tl.int64)
    inner = offs % inner_size
    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = safe_outer * dim_stride + lo + inner
    tl.store(out_ptr + out_offset, vals, mask=mask)


@triton.jit
def _select_bwd_patch_flat_i32_kernel(
    grad_ptr,
    out_ptr,
    total,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """int32-safe-domain version of the flat patch: the element-domain multiply-add stays int32
    end to end (dispatch guarantees total/n_out < 2^31, so outer*dim_stride+lo+inner <=
    n_out-1 < 2^31 cannot wrap; masked lanes already clamp outer to 0), with a single
    .to(tl.int64) at the final addressing step, avoiding the flat path whole-chain 64-bit
    vector multiply. inner_size/dim_stride are constexpr, so non-power-of-two division is
    still strength-reduced to multiply+shift. Measured (GCU400, do_bench kernel mode):
    (64,512,512) d1 patch 5.7->4.2us, d2 6.9->5.9us, [1024^3] d2 93->91.5us (1M scattered 4B floor).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    safe_outer = tl.where(mask, offs // inner_size, 0)
    inner = offs % inner_size
    off32 = safe_outer * dim_stride + lo + inner
    vals = tl.load(grad_ptr + offs, mask=mask)
    tl.store(out_ptr + off32.to(tl.int64), vals, mask=mask)


def _next_pow2(x):
    x = int(x)
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@triton.jit
def _select_bwd_spanfill_kernel(
    grad_ptr,
    out_ptr,
    n_out,
    inner_size: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Dedicated single kernel for outer==1 (dim0): one pass writes the whole output --
    [lo, lo+inner_size) is copied contiguously from grad and the rest is zeroed, saving the second

    fill+patch launch and the double write of the span.
    Each block classifies at scalar granularity (a uniform branch, no per-lane decode): fully
    outside the span is a pure zero write, fully inside a contiguous copy, and a straddling block
    is a range compare plus masked contiguous load/store.
    Measured on large outputs (do_bench kernel mode, GCU400): (2^30,) fp32 fill 1382us -> 1326us;
    [1024^3] d0 1410 -> 1328; [4096^2] d0 26.5 -> 25.1; (64,512,512) d0 28.2 -> 25.7 (n_out < 2^31).
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for b0 in tl.range(pid * BLOCK, n_out, nprog * BLOCK):
        offs = b0 + tl.arange(0, BLOCK)
        b_end = b0 + BLOCK
        if (b_end <= lo) or (b0 >= lo + inner_size):
            tl.store(
                out_ptr + offs,
                tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
                mask=offs < n_out,
            )
        elif (b0 >= lo) and (b_end <= lo + inner_size):
            m = offs < n_out
            v = tl.load(grad_ptr + (offs - lo), mask=m)
            tl.store(out_ptr + offs, v, mask=m)
        else:
            m = offs < n_out
            sel = (offs >= lo) & (offs < lo + inner_size)
            src = tl.where(sel, offs - lo, 0)
            v = tl.load(grad_ptr + src, mask=m & sel, other=0)
            tl.store(out_ptr + offs, v, mask=m)


@triton.jit
def _select_bwd_fused_kernel(
    grad_ptr,
    out_ptr,
    total_out,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Folds "zero the whole output + backfill the selected slice" into one output pass.

    Output positions offs split into (outer, inner): those inside [lo, lo+inner_size) read
    grad[outer*inner_size + inner-lo], everything else writes 0. The output is written once and
    one kernel launch is saved -- at small output sizes launch overhead dominates.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for base in tl.range(pid * BLOCK, total_out, nprog * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < total_out
        outer = offs // dim_stride
        inner = offs % dim_stride
        selected = (inner >= lo) & (inner < lo + inner_size)
        gather = mask & selected
        # Use address 0 as a placeholder for unselected positions to avoid out-of-bounds pointer arithmetic (a
        # masked-out load is not dereferenced)
        src = tl.where(gather, outer * inner_size + (inner - lo), 0)
        vals = tl.load(grad_ptr + src, mask=gather, other=0)
        tl.store(out_ptr + offs, vals, mask=mask)


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS_ENFLAME SELECT_BACKWARD")
    dim = int(dim)
    index = int(index)
    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    if out is None:
        out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1
    dim_stride = dim_size * inner_size
    n_out = outer_size * dim_stride
    if n_out == 0:
        return out

    grad_c = grad.contiguous()
    total = outer_size * inner_size
    lo = index * inner_size

    # Small outputs use single-kernel fusion: one pass fills the whole output (take grad for the selected span, write 0
    # elsewhere),
    # saving one launch versus "zero kernel + patch kernel" — measured 12.8us -> 4.8us at 64x64.
    # Large outputs still use fill+patch: fill is a pure streaming write and patch touches only the selected slice, so
    # the extra
    # read/write for the patch is negligible compared to writing the whole buffer, whereas the fused version must do
    # div/mod per element and is noticeably slower at large sizes.
    if n_out <= _FUSED_MAX_OUT:
        fused_block = max(256, min(_FILL_BLOCK, _next_pow2(n_out)))
        fused_grid = (min(48, triton.cdiv(n_out, fused_block)),)
        _select_bwd_fused_kernel[fused_grid](
            grad_c,
            out,
            n_out,
            inner_size,
            dim_stride,
            lo,
            fused_block,
            num_warps=4,
        )
        return out

    wide = n_out >= 2**31
    item_bytes = grad_c.element_size()
    # outer==1: fill the whole output in one pass with a single kernel (span copy + zero the rest), saving the second
    # launch.
    # See the _SPAN_MAX_OUT comment for the dispatch condition: 2-byte dtypes are limited to ≤1GiB (larger sizes have a
    # measured
    # codegen tax), while 4-byte dtypes win at any n_out < 2^31 in measurements.
    if (
        outer_size == 1
        and not wide
        and (item_bytes == 4 or n_out * item_bytes <= _SPAN_MAX_BYTES)
    ):
        span_grid = (min(48, triton.cdiv(n_out, _SPAN_BLOCK)),)
        _select_bwd_spanfill_kernel[span_grid](
            grad_c,
            out,
            n_out,
            inner_size,
            lo,
            _SPAN_BLOCK,
            num_warps=4,
        )
        return out

    fill_block = max(1024, min(_FILL_BLOCK, _next_pow2(n_out)))
    n_fill_blocks = triton.cdiv(n_out, fill_block)
    fill_grid = (min(48, n_fill_blocks),)
    _select_bwd_fill_kernel[fill_grid](out, n_out, fill_block, wide, num_warps=4)

    if total > 0:
        if inner_size >= 1024:
            rblock = min(8192, _next_pow2(inner_size))
            n_rblocks = triton.cdiv(inner_size, rblock)
            patch_grid = (min(48, outer_size), min(48, n_rblocks))
            _select_bwd_patch_rowblock_kernel[patch_grid](
                grad_c,
                out,
                outer_size,
                inner_size,
                dim_stride,
                lo,
                rblock,
                num_warps=4,
            )
        elif _FLAT_I32_MIN <= total <= _FLAT_I32_MAX and not wide:
            # int32-safe-domain flat patch, with pb=4096 measured best across the board (see kernel comment).
            pblock = 4096
            _select_bwd_patch_flat_i32_kernel[(triton.cdiv(total, pblock),)](
                grad_c,
                out,
                total,
                inner_size,
                dim_stride,
                lo,
                pblock,
                num_warps=4,
            )
        else:
            pblock = max(256, min(16384, _next_pow2(triton.cdiv(total, 96))))
            n_pblocks = triton.cdiv(total, pblock)
            patch_grid = (n_pblocks,)
            _select_bwd_patch_flat_kernel[patch_grid](
                grad_c,
                out,
                total,
                inner_size,
                dim_stride,
                lo,
                pblock,
                wide,
                num_warps=4,
            )
    return out
