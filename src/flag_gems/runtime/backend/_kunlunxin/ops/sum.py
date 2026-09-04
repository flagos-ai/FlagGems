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

# from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# =============================================================================
# XPU correctness constraints (see harness/HARNESS_SUMMARY.md §2.5 / §3.6):
#   - `tl.sum` is only exact for tiles <= 8192 lanes (no buffer limit) or
#     == 32768 lanes WITH buffer_size_limit=2048; 16384/65536 miscompile.
#   - Masked loads (mask + other=0) are NOT reliable inside reductions for
#     tiles >= 32768 lanes, and unreliable INSIDE runtime loops; single-shot
#     masked tiles <= 8192 lanes ARE reliable (validated 2026-08-22).
#   - `tl.where(mask, x, 0)` inside a reduction miscompiles; avoid.
# All kernels below are built exclusively from: unmasked exact-size tiles,
# single-shot masked tiles <= 8192 lanes, static merges <= 8192 lanes.
# =============================================================================

# Flat (full-tensor) reduction chunk. 32768 lanes with buffer_size_limit=2048
# is the documented exact point on this XPU.
_FLAT_CHUNK = 32768
# Row-reduce tile width for the dim path (exact <=8192 safe tile).
_ROW_BN = 8192
_BLOCK_M = 128
_SMALL_M = 4096
_HUGE_N = 32768
_SMALL_BLOCK_M = 8
_TAIL_BLOCK_M = 64


def _resolve_acc_dtype(inp_dtype):
    """ATen reduction accumulate type: fp32 for 16-bit fp, fp64 for fp64,
    int64 for all integers/bool. 32-bit fp accumulates in fp32."""
    if inp_dtype is torch.float64:
        return torch.float64
    if inp_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    return torch.int64


def _resolve_out_dtype(inp_dtype, dtype):
    """torch.sum dtype=None semantics: int/bool -> int64, float -> input dtype."""
    if dtype is not None:
        return dtype
    if inp_dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32):
        return torch.int64
    return inp_dtype


@libentry()
@triton.jit
def sum_flat_core_kernel(inp, out, CHUNK: tl.constexpr):
    """Unmasked exact-size 1D chunk reduce. Grid = number of full chunks."""
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif (
        tl.constexpr(inp.dtype.element_ty == tl.float16)
        or tl.constexpr(inp.dtype.element_ty == tl.bfloat16)
    ) or tl.constexpr(inp.dtype.element_ty == tl.float32):
        cdtype = tl.float32
    else:
        cdtype = tl.int64
    pid = ext.program_id(0)
    off = pid * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(inp + off).to(cdtype)
    tl.store(out + pid, tl.sum(x))


@libentry()
@triton.jit
def _sum_flat_tail_kernel(inp, out, TL: tl.constexpr):
    """Sum of an already zero-padded (staged) buffer; fully unmasked,
    TL == next_pow2(tail) <= _FLAT_CHUNK."""
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif (
        tl.constexpr(inp.dtype.element_ty == tl.float16)
        or tl.constexpr(inp.dtype.element_ty == tl.bfloat16)
    ) or tl.constexpr(inp.dtype.element_ty == tl.float32):
        cdtype = tl.float32
    else:
        cdtype = tl.int64
    off = tl.arange(0, TL)
    x = tl.load(inp + off).to(cdtype)
    tl.store(out, tl.sum(x))


@libentry()
@triton.jit
def _sum_flat_tail_masked_kernel(inp, out, start, NTAIL, TL: tl.constexpr):
    """Single-shot masked tail (NTAIL <= 8192 lanes; validated 2026-08-22).
    start is a scalar flat offset."""
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif (
        tl.constexpr(inp.dtype.element_ty == tl.float16)
        or tl.constexpr(inp.dtype.element_ty == tl.bfloat16)
    ) or tl.constexpr(inp.dtype.element_ty == tl.float32):
        cdtype = tl.float32
    else:
        cdtype = tl.int64
    off = tl.arange(0, TL)
    x = tl.load(inp + start + off, mask=off < NTAIL, other=0).to(cdtype)
    tl.store(out, tl.sum(x))


@libentry()
@triton.jit
def _sum_flat_merge_kernel(mid, out, np, NLANES: tl.constexpr):
    """Single-shot masked merge of np partials (np <= 8192)."""
    if tl.constexpr(mid.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif (
        tl.constexpr(mid.dtype.element_ty == tl.float16)
        or tl.constexpr(mid.dtype.element_ty == tl.bfloat16)
    ) or tl.constexpr(mid.dtype.element_ty == tl.float32):
        cdtype = tl.float32
    else:
        cdtype = tl.int64
    off = tl.arange(0, NLANES)
    x = tl.load(mid + off, mask=off < np, other=0).to(cdtype)
    tl.store(out, tl.sum(x))


@libentry()
@triton.jit
def _sum_flat_group_kernel(mid, gsum, GCHUNK: tl.constexpr):
    """Unmasked group-reduce of a zero-padded partial buffer (compresses
    8192 partials per program)."""
    pid = ext.program_id(0)
    off = pid * GCHUNK + tl.arange(0, GCHUNK)
    x = tl.load(mid + off).to(mid.dtype.element_ty)
    tl.store(gsum + pid, tl.sum(x))


def _launch_sum_flat(inp, out, acc_dtype):
    """Two/three-stage exact flat reduction writing into `out` (0-d or (1,))."""
    if inp.ndim != 1:
        # The tail staging below slices by flat offset; operate on the flat view.
        # reshape (not view): a transposed input cannot be viewed flat.
        inp = inp.reshape(-1)
    M = inp.shape[0]
    if M == 0:
        zero_(out)
        return
    if M <= 8192:
        # Small input: single-shot masked tile (<=8192 lanes is a validated
        # reliable point on this XPU) - one launch instead of the staged copy.
        # Fixed 8192-lane tile: a single compiled kernel serves all M <= 8192.
        with torch_device_fn.device(inp.device):
            _sum_flat_merge_kernel[(1, 1, 1)](inp, out, M, 8192)
        return
    # CHUNK = 8192 keeps mid-M inputs (8K < M < 32K) on the cheap exact
    # tiles; CHUNK = 32768 (with buffer_size_limit=2048) halves the partial
    # count for large inputs. The tail is either a single-shot masked tile
    # (<= 8192 lanes, no staging) or a zero-padded staged tile.
    CH = 8192 if M < _FLAT_CHUNK else _FLAT_CHUNK
    nfull = M // CH
    tail = M % CH
    nb = nfull + (1 if tail else 0)
    mid = torch.empty((nb,), dtype=acc_dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        if nfull:
            sum_flat_core_kernel[(nfull, 1, 1)](inp, mid, CH, buffer_size_limit=2048)
        if tail and tail <= 8192:
            _sum_flat_tail_masked_kernel[(1, 1, 1)](
                inp,
                mid[nfull : nfull + 1],
                nfull * CH,
                tail,
                triton.next_power_of_2(tail),
            )
        elif tail:
            TL = triton.next_power_of_2(tail)
            staged = torch.zeros((TL,), dtype=inp.dtype, device=inp.device)
            torch.ops.aten._copy_from(inp[nfull * CH : M], staged[:tail], False)
            _sum_flat_tail_kernel[(1, 1, 1)](
                staged, mid[nfull : nfull + 1], TL, buffer_size_limit=2048
            )
        if nb <= 8192:
            _sum_flat_merge_kernel[(1, 1, 1)](mid, out, nb, triton.next_power_of_2(nb))
        else:
            # > 8192 partials (M > 268M): compress via 8192-wide unmasked
            # groups of a zero-padded partial buffer, then a small merge.
            g = (nb + 8191) // 8192
            padded = torch.zeros((g * 8192,), dtype=acc_dtype, device=inp.device)
            torch.ops.aten._copy_from(mid, padded[:nb], False)
            gsum = torch.empty((g,), dtype=acc_dtype, device=inp.device)
            _sum_flat_group_kernel[(g, 1, 1)](padded, gsum, 8192)
            _sum_flat_merge_kernel[(1, 1, 1)](gsum, out, g, triton.next_power_of_2(g))


@libentry()
@triton.jit
def _sum_row_full_kernel(
    inp, out, M, STRIDE, NW, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Row-reduce over cols [0, NW) with NW % BLOCK_N == 0 and row stride
    STRIDE (STRIDE >= NW; the trailing STRIDE-NW lanes carry the tail).
    Fully unmasked loads (rows clamped to [0, M-1]), reduce-OUTSIDE
    accumulation, masked row stores. Exact (validated 2026-08-22)."""
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif (
        tl.constexpr(inp.dtype.element_ty == tl.float16)
        or tl.constexpr(inp.dtype.element_ty == tl.bfloat16)
    ) or tl.constexpr(inp.dtype.element_ty == tl.float32):
        cdtype = tl.float32
    else:
        cdtype = tl.int64
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_c = tl.where(rows < M, rows, M - 1)
    inp = inp + rows_c * STRIDE
    out = out + rows
    row_mask = rows < M
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(cdtype)
        _sum += a
    tl.store(out, tl.sum(_sum, axis=1)[:, None], row_mask)


@libentry()
@triton.jit
def _sum_row_tail_kernel(
    inp,
    in_sum,
    out,
    M,
    N,
    N0,
    NTAIL,
    BLOCK_M: tl.constexpr,
    TL: tl.constexpr,
):
    """Row tail (rows' trailing NTAIL < 8192 lanes): out[r] = in_sum[r] +
    sum(inp[r, N0:N0+NTAIL]). Single-shot masked 1-D tiles per row
    (static-unrolled over rows; validated 2026-08-22)."""
    if tl.constexpr(inp.dtype.element_ty == tl.float64):
        cdtype = tl.float64
    elif (
        tl.constexpr(inp.dtype.element_ty == tl.float16)
        or tl.constexpr(inp.dtype.element_ty == tl.bfloat16)
    ) or tl.constexpr(inp.dtype.element_ty == tl.float32):
        cdtype = tl.float32
    else:
        cdtype = tl.int64
    pid = ext.program_id(0)
    off = tl.arange(0, TL)
    for ri in tl.static_range(BLOCK_M):
        row = pid * BLOCK_M + ri
        row_c = tl.minimum(row, M - 1)
        a = tl.load(inp + row_c * N + N0 + off, mask=off < NTAIL, other=0).to(cdtype)
        s = tl.sum(a) + tl.load(in_sum + row_c).to(cdtype)
        tl.store(out + row, s, row < M)


def _launch_sum_dim(inp, out, M, N):
    if M == 1:
        # Degenerate: whole tensor reduces to one element -> route to the
        # exact flat machinery (parallel over N).
        _launch_sum_flat(inp.view(-1), out, _resolve_acc_dtype(inp.dtype))
        return

    block_m = _BLOCK_M
    if M <= _SMALL_M and N >= _HUGE_N:
        block_m = _SMALL_BLOCK_M
    n0 = (N // _ROW_BN) * _ROW_BN
    tail = N - n0
    with torch_device_fn.device(inp.device):
        if tail == 0:
            _sum_row_full_kernel[(triton.cdiv(M, block_m), 1, 1)](
                inp, out, M, N, N, block_m, _ROW_BN, buffer_size_limit=2048
            )
        else:
            acc = _resolve_acc_dtype(inp.dtype)
            full = torch.empty((M,), dtype=acc, device=inp.device)
            _sum_row_full_kernel[(triton.cdiv(M, block_m), 1, 1)](
                inp, full, M, N, n0, block_m, _ROW_BN, buffer_size_limit=2048
            )
            _sum_row_tail_kernel[(triton.cdiv(M, _TAIL_BLOCK_M), 1, 1)](
                inp,
                full,
                out,
                M,
                N,
                n0,
                tail,
                _TAIL_BLOCK_M,
                triton.next_power_of_2(tail),
            )


def _prep_flat(inp, dtype):
    if dtype is None and inp.dtype is torch.bool:
        inp = inp.to(torch.int64)
    out_dtype = _resolve_out_dtype(inp.dtype, dtype)
    acc_dtype = _resolve_acc_dtype(inp.dtype)
    return inp, out_dtype, acc_dtype


def sum(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SUM")
    inp, out_dtype, acc_dtype = _prep_flat(inp, dtype)
    out = torch.empty([], dtype=out_dtype, device=inp.device)
    _launch_sum_flat(inp, out, acc_dtype)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN SUM_OUT")
    inp, _, acc_dtype = _prep_flat(inp, dtype)
    _launch_sum_flat(inp, out, acc_dtype)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SUM_DIM")
    out_dtype = _resolve_out_dtype(inp.dtype, dtype)

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is None or dim == []:
            out_shape = [1] * len(out_shape) if keepdim else []
        else:
            dims = dim if isinstance(dim, (list, tuple)) else [dim]
            if keepdim:
                for d in dims:
                    out_shape[d % inp.ndim] = 1
            else:
                for d in sorted(dims, key=lambda x: x % inp.ndim, reverse=True):
                    out_shape.pop(d % inp.ndim)
        out = torch.empty(out_shape, dtype=out_dtype, device=inp.device)
        zero_(out)
        return out

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=out_dtype, device=inp.device)

    _launch_sum_dim(inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN SUM_DIM_OUT")

    if inp.numel() == 0:
        dims = (
            dim
            if isinstance(dim, (list, tuple))
            else ([dim] if dim is not None else [])
        )
        if keepdim:
            for d in dims:
                pass  # out shape already correct from caller
        zero_(out)
        return out

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out.resize_(shape)
    _launch_sum_dim(inp, out, M, N)
    if not keepdim:
        # Compute squeezed shape and resize in-place
        out_shape = [s for i, s in enumerate(shape) if i not in dim]
        out.resize_(out_shape)
    return out
