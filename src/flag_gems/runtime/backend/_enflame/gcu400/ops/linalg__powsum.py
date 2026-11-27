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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

# GCU400 exposes 48 SPs; a persistent grid of at most one program per SP
# amortises per-program setup across the whole tensor.
_MAX_GRID = 48
# Largest element count handled by the fused single-block full reduction.
# Above it one block can no longer hide the load latency, so the two-stage
# persistent scheme wins (cutoff shared with gcu400/ops/vector_norm.py).
_SINGLE_PASS_MAX_NUMEL = 65536
# Cap on the row-tile accumulator BLOCK_M*BLOCK_N (fp32 elements, 128KB DSM)
# shared with gcu400/ops/vector_norm.py; keeps the peak tile + load buffers
# inside the 448KB per-program DSM budget.
_DIM_TILE_ELEMS = 32768


def _acc_torch_dtype(compute_dtype):
    return torch.float64 if compute_dtype == torch.float64 else torch.float32


@libentry()
@triton.jit(do_not_specialize=["M", "N", "ord"])
def powsum_dim_kernel(
    inp,
    out,
    M,
    N,
    ord,
    MODE: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    EXACT: tl.constexpr,
    N_FITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """Row-wise sum(|x|^ord) over the last dim of a contiguous [M, N] view.

    Persistent strided row tiles: program p handles tiles p, p+G, p+2G ...
    so a grid capped at the SP count still covers any M. Each tile iteration
    is pure elementwise work (load, term, accumulate); the only cross-lane
    op is one BLOCK_N-wide tree per row at tile end. M/N are typed by value
    (i32 unless the tensor exceeds 2^31 elements, then i64 automatically).
    """
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    inp_dtype = inp.type.element_ty
    acc_dtype = tl.float64 if inp_dtype == tl.float64 else tl.float32

    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M

        if MODE == 1:
            # |x|^0 == 1 for every element including NaN/Inf: the row sum is
            # simply the number of valid columns.
            tl.store(out + rows, tl.zeros([BLOCK_M], dtype=acc_dtype) + N, row_mask)
        else:
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
            if N_FITS:
                # Single tile covers the whole row: no inner loop at all.
                cols = tl.arange(0, BLOCK_N)[None, :]
                offsets = rows[:, None] * N + cols
                if IS_COMPLEX:
                    # The interleaved real/imag view doubles the flat index;
                    # widen before scaling so 2*offsets cannot wrap int32.
                    offsets = offsets.to(tl.int64)
                if EXACT:
                    if IS_COMPLEX:
                        real = tl.load(inp + 2 * offsets).to(acc_dtype)
                        imag = tl.load(inp + 2 * offsets + 1).to(acc_dtype)
                        value = tl.sqrt(real * real + imag * imag)
                    else:
                        value = tl.load(inp + offsets).to(acc_dtype)
                else:
                    mask = row_mask[:, None] & (cols < N)
                    if IS_COMPLEX:
                        real = tl.load(inp + 2 * offsets, mask, other=0.0).to(acc_dtype)
                        imag = tl.load(inp + 2 * offsets + 1, mask, other=0.0).to(
                            acc_dtype
                        )
                        value = tl.sqrt(real * real + imag * imag)
                    else:
                        value = tl.load(inp + offsets, mask, other=0.0).to(acc_dtype)
                if MODE == 0:
                    term = pow(tl.abs(value), ord.to(acc_dtype))
                    if EXACT:
                        acc += term
                    else:
                        acc += tl.where(mask, term, 0.0)
                elif MODE == 2:
                    acc += tl.abs(value)
                elif MODE == 3:
                    acc += value * value
                else:
                    acc += tl.abs(value * value * value)
            else:
                for start in range(0, N, BLOCK_N):
                    cols = start + tl.arange(0, BLOCK_N)[None, :]
                    offsets = rows[:, None] * N + cols
                    if IS_COMPLEX:
                        offsets = offsets.to(tl.int64)
                    if EXACT:
                        if IS_COMPLEX:
                            real = tl.load(inp + 2 * offsets).to(acc_dtype)
                            imag = tl.load(inp + 2 * offsets + 1).to(acc_dtype)
                            value = tl.sqrt(real * real + imag * imag)
                        else:
                            value = tl.load(inp + offsets).to(acc_dtype)
                    else:
                        mask = row_mask[:, None] & (cols < N)
                        if IS_COMPLEX:
                            real = tl.load(inp + 2 * offsets, mask, other=0.0).to(
                                acc_dtype
                            )
                            imag = tl.load(inp + 2 * offsets + 1, mask, other=0.0).to(
                                acc_dtype
                            )
                            value = tl.sqrt(real * real + imag * imag)
                        else:
                            value = tl.load(inp + offsets, mask, other=0.0).to(
                                acc_dtype
                            )
                    if MODE == 0:
                        term = pow(tl.abs(value), ord.to(acc_dtype))
                        if EXACT:
                            acc += term
                        else:
                            acc += tl.where(mask, term, 0.0)
                    elif MODE == 2:
                        acc += tl.abs(value)
                    elif MODE == 3:
                        acc += value * value
                    else:
                        acc += tl.abs(value * value * value)

            tl.store(out + rows, tl.sum(acc, axis=1), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "ord"])
def powsum_global_single(
    inp,
    out,
    M,
    ord,
    MODE: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-block full reduction for small tensors: one launch, one tree."""
    offsets = tl.arange(0, BLOCK)
    mask = offsets < M
    if MODE == 1:
        tl.store(out, tl.sum(mask.to(tl.float32)))
    else:
        inp_dtype = inp.type.element_ty
        acc_dtype = tl.float64 if inp_dtype == tl.float64 else tl.float32
        if IS_COMPLEX:
            real = tl.load(inp + 2 * offsets, mask, other=0.0).to(acc_dtype)
            imag = tl.load(inp + 2 * offsets + 1, mask, other=0.0).to(acc_dtype)
            value = tl.sqrt(real * real + imag * imag)
        else:
            value = tl.load(inp + offsets, mask, other=0.0).to(acc_dtype)
        if MODE == 0:
            tl.store(
                out, tl.sum(tl.where(mask, pow(tl.abs(value), ord.to(acc_dtype)), 0.0))
            )
        elif MODE == 2:
            tl.store(out, tl.sum(tl.abs(value)))
        elif MODE == 3:
            tl.store(out, tl.sum(value * value))
        else:
            tl.store(out, tl.sum(tl.abs(value * value * value)))


@libentry()
@triton.jit(do_not_specialize=["M", "ord"])
def powsum_global_partial(
    inp,
    mid,
    M,
    ord,
    MODE: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """Persistent partial full reduction (gcu400 native_norm scheme).

    Each program owns a strided set of full blocks and folds them into a
    register accumulator with pure elementwise FFMAs — no cross-lane traffic
    inside the hot loop, and the single masked tail block is owned by one
    program so the loop stays mask-free. Only one BLOCK-lane tree per
    program runs at the very end.
    """
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_full = M // BLOCK

    if MODE == 1:
        # Every valid element contributes exactly 1.
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for _chunk in tl.range(pid, num_full, num_prog, num_stages=num_stages):
            acc += 1.0
        if pid == num_full % num_prog:
            offsets = num_full * BLOCK + tl.arange(0, BLOCK)
            acc += (offsets < M).to(tl.float32)
        tl.store(mid + pid, tl.sum(acc))
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for chunk in tl.range(pid, num_full, num_prog, num_stages=num_stages):
            offset = chunk * BLOCK + tl.arange(0, BLOCK)
            if IS_COMPLEX:
                # The interleaved real/imag view doubles the flat index;
                # widen before scaling so 2*offset cannot wrap int32.
                offset = offset.to(tl.int64)
                real = tl.load(inp + 2 * offset).to(tl.float32)
                imag = tl.load(inp + 2 * offset + 1).to(tl.float32)
                value = tl.sqrt(real * real + imag * imag)
            else:
                value = tl.load(inp + offset).to(tl.float32)
            if MODE == 0:
                acc += pow(tl.abs(value), ord.to(tl.float32))
            elif MODE == 2:
                acc += tl.abs(value)
            elif MODE == 3:
                acc += value * value
            else:
                acc += tl.abs(value * value * value)

        # The single masked tail block is folded in by its owner, keeping the
        # hot loop free of any mask predicate.
        if pid == num_full % num_prog:
            offset = num_full * BLOCK + tl.arange(0, BLOCK)
            mask = offset < M
            if IS_COMPLEX:
                offset = offset.to(tl.int64)
                real = tl.load(inp + 2 * offset, mask, other=0.0).to(tl.float32)
                imag = tl.load(inp + 2 * offset + 1, mask, other=0.0).to(tl.float32)
                value = tl.sqrt(real * real + imag * imag)
            else:
                value = tl.load(inp + offset, mask, other=0.0).to(tl.float32)
            if MODE == 0:
                acc += tl.where(mask, pow(tl.abs(value), ord.to(tl.float32)), 0.0)
            elif MODE == 2:
                acc += tl.where(mask, tl.abs(value), 0.0)
            elif MODE == 3:
                acc += tl.where(mask, value * value, 0.0)
            else:
                acc += tl.where(mask, tl.abs(value * value * value), 0.0)

        tl.store(mid + pid, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def powsum_global_final(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offsets = tl.arange(0, BLOCK_MID)
    values = tl.load(mid + offsets, offsets < MID_SIZE, other=0.0)
    tl.store(out, tl.sum(values))


def _normalize_dims(dim, ndim):
    if dim is None or dim == [] or dim == ():
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    if ndim == 0:
        if any(d not in (-1, 0) for d in dim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [-1, 0])"
            )
        if len(dim) > 1:
            raise RuntimeError("dim 0 appears multiple times in the list of dims")
        return []
    if any(d < -ndim or d >= ndim for d in dim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}])"
        )
    dims = [d % ndim for d in dim]
    if len(set(dims)) != len(dims):
        raise RuntimeError("dim appears multiple times")
    return dims


def _output_dtype(inp, dtype):
    dtype = inp.dtype if dtype is None else dtype
    if dtype == torch.complex64:
        return torch.complex64, torch.float32
    if dtype == torch.complex128:
        return torch.complex128, torch.float64
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(f"linalg__powsum not implemented for {dtype}")
    return dtype, dtype


def _launch_global(inp, out, size, ord, mode, is_complex, compute_dtype):
    with torch_device_fn.device(inp.device):
        if size <= _SINGLE_PASS_MAX_NUMEL:
            powsum_global_single[(1,)](
                inp,
                out,
                size,
                ord,
                mode,
                is_complex,
                BLOCK=max(1024, min(triton.next_power_of_2(size), 32768)),
                num_warps=1,
            )
            return

        # Two-stage persistent scheme: wide chunks keep the per-program hot
        # loop pure elementwise FMA; fp32/complex cap at 32K elements so the
        # 128KB fp32 accumulator + load buffer stay inside the DSM budget.
        elem_size = inp.element_size()
        cap = 32768 if elem_size >= 4 else 65536
        target = triton.next_power_of_2((size + _MAX_GRID - 1) // _MAX_GRID)
        block = min(cap, max(2048, target))
        num_chunks = triton.cdiv(size, block)
        grid = min(num_chunks, _MAX_GRID)

        mid = torch.empty(
            [grid], dtype=_acc_torch_dtype(compute_dtype), device=inp.device
        )
        powsum_global_partial[(grid,)](
            inp,
            mid,
            size,
            ord,
            mode,
            is_complex,
            BLOCK=block,
            num_stages=2 if num_chunks > grid else 1,
            num_warps=4,
        )
        powsum_global_final[(1,)](
            mid, out, grid, BLOCK_MID=triton.next_power_of_2(grid), num_warps=1
        )


def _launch_dim(inp, out, rows, reduction_size, ord, mode, is_complex):
    # BLOCK_N must hug N: a 1024-wide tile on N=256 wastes 4x lanes in the
    # predicate, term ALU and per-row tree (measured 26.7 -> 65.7us on
    # (64,512,256) fp16). BLOCK_M then scales with the row count so small
    # tensors still spread over ~16 programs while huge ones take the widest
    # fp32-safe tile: the sweep winners BM=16/32/128 for M=256/1024/32768 all
    # match next_pow2(M/16) capped by the 128KB fp32 accumulator budget.
    block_n = min(triton.next_power_of_2(reduction_size), 2048)
    block_m = max(
        1,
        min(
            128,
            _DIM_TILE_ELEMS // block_n,
            triton.next_power_of_2(max(1, rows // 16)),
        ),
    )
    if is_complex:
        # The interleaved real/imag view keeps two fp32 tiles live; halve the
        # row tile to stay inside the 448KB per-program DSM budget.
        block_m = max(1, block_m // 2)
    num_tiles = triton.cdiv(rows, block_m)
    grid = min(num_tiles, _MAX_GRID)

    powsum_dim_kernel[(grid,)](
        inp,
        out,
        rows,
        reduction_size,
        ord,
        mode,
        is_complex,
        rows % block_m == 0 and reduction_size % block_n == 0,
        block_n >= reduction_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_stages=2 if num_tiles > grid else 1,
        num_warps=1,
    )


def linalg__powsum(inp, ord=2, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME LINALG__POWSUM")
    compute_dtype, output_dtype = _output_dtype(inp, dtype)
    if inp.dtype != compute_dtype and not (
        inp.is_complex() and not compute_dtype.is_complex
    ):
        inp = inp.to(compute_dtype)
    is_complex = inp.is_complex()

    dims = _normalize_dims(dim, inp.ndim)
    shape = list(inp.shape)
    reduced_shape = list(shape)
    for d in dims:
        reduced_shape[d] = 1
    out_shape = (
        reduced_shape
        if keepdim
        else [size for i, size in enumerate(shape) if i not in set(dims)]
    )

    if inp.numel() == 0:
        return torch.zeros(out_shape, dtype=output_dtype, device=inp.device)

    mode = 1 if ord == 0 else 2 if ord == 1 else 3 if ord == 2 else 4 if ord == 3 else 0
    with torch_device_fn.device(inp.device):
        if len(dims) == inp.ndim:
            ndim = inp.ndim
            inp = inp.contiguous()
            size = inp.numel()
            if is_complex:
                inp = torch.view_as_real(inp)
            out = torch.empty([], dtype=output_dtype, device=inp.device)
            _launch_global(inp, out, size, ord, mode, is_complex, compute_dtype)
            if keepdim:
                out = out.reshape([1] * ndim)
            return out

        ndim = inp.ndim
        # Innermost-dim reduction of a contiguous tensor is already in the
        # [M, N] layout the kernel wants; skip the permute + contiguous copy.
        if dims != [ndim - 1] or not inp.is_contiguous():
            inp = dim_compress(inp, dims)
        reduction_size = math.prod(shape[d] for d in dims)
        rows = inp.numel() // reduction_size
        if is_complex:
            inp = torch.view_as_real(inp)
        out = torch.empty(rows, dtype=output_dtype, device=inp.device)
        _launch_dim(inp, out, rows, reduction_size, ord, mode, is_complex)
    return out.reshape(out_shape)
